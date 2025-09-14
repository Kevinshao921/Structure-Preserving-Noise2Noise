import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision.transforms import ToTensor, ToPILImage

from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from numpy.fft import fft2, ifft2

from bm3d import bm3d

from time import time

import copy

from scipy.ndimage import median_filter

from skimage.transform import resize

import pandas as pd

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

print("device:", device)

INPUT_FOLDER = r"INPUT" 
OUTPUT_FOLDER = r"OUTPUT"
SUPPORTED_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png']

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_image_files(folder_path):
    image_files = []
    seen_files = set() 
    
    for extension in SUPPORTED_EXTENSIONS:
        files = glob.glob(os.path.join(folder_path, extension))
        for file in files:
            abs_path = os.path.abspath(file)
            if abs_path not in seen_files:
                image_files.append(file)
                seen_files.add(abs_path)
        
        files = glob.glob(os.path.join(folder_path, extension.upper()))
        for file in files:
            abs_path = os.path.abspath(file)
            if abs_path not in seen_files:
                image_files.append(file)
                seen_files.add(abs_path)
    
    return sorted(image_files)

def add_noise_gaussian(x, noise_level): 
    noisy = x + torch.normal(0, noise_level, x.shape, device=x.device)
    return torch.clamp(noisy, 0, 1)

def add_noise_poisson(x, peak):
    x_cpu = x.cpu()
    scaled = x_cpu * peak
    noisy_cpu = torch.poisson(scaled) / peak
    return torch.clamp(noisy_cpu.to(x.device), 0, 1)

def add_noise_salt_pepper(x, noise_level):
    noisy = x.clone()
    b,c,h,w = noisy.shape
    rand = torch.rand((b,c,h,w), device = x.device)
    noisy[rand < (noise_level / 2)] = 0.0
    noisy[(rand >= (noise_level / 2)) & (rand < noise_level)] = 1.0
    return noisy

def add_noise(x, noise_param, current_noise_type):
    # 兼容旧接口：单一噪声
    if isinstance(current_noise_type, str):
        if current_noise_type == "gaussian":
            return add_noise_gaussian(x, noise_param)
        elif current_noise_type == "poisson":
            return add_noise_poisson(x, noise_param)
        elif current_noise_type == "sp":
            return add_noise_salt_pepper(x, noise_param)
        else:
            raise ValueError(f"Unsupported noise type: {current_noise_type}")
    # 新接口：多种噪声
    elif isinstance(current_noise_type, list):
        return add_mixed_noise(x, current_noise_type)
    else:
        raise ValueError("current_noise_type must be str or list")


def add_mixed_noise(x, noise_configs):
    """
    noise_configs: list of tuples (noise_type, noise_param)
        e.g. [('gaussian', 0.05), ('sp', 0.02)]
    """
    noisy = x.clone()
    for noise_type, param in noise_configs:
        if noise_type == 'gaussian':
            noisy = add_noise_gaussian(noisy, param)
        elif noise_type == 'poisson':
            noisy = add_noise_poisson(noisy, param)
        elif noise_type == 'sp':
            noisy = add_noise_salt_pepper(noisy, param)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
    return torch.clamp(noisy, 0, 1)


def ssim_numpy(img1, img2):
    if img1.ndim == 3:
        ssim_values = []
        for i in range(img1.shape[2]):
            ssim_val = structural_similarity(img1[:, :, i], img2[:, :, i], data_range = 1.0)
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        return structural_similarity(img1, img2, data_range = 1.0)

class Network(nn.Module):
    def __init__(self, n_chan, chan_embed=48):
        super(Network, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)
        
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)  
        return x

class L0Smoothing:
    def __init__(self, img_path: str = "",
                 param_lambda: Optional[float] = 2e-2,
                 param_kappa: Optional[float] = 2.0,
                 max_iter: Optional[int] = 10):
        
        self._lambda = param_lambda
        self._kappa = param_kappa
        self._img_path = img_path
        self._beta_max = 1e5
        self.max_iter = max_iter

    def psf2otf(self, psf, shape):
        otf = np.zeros(shape, dtype = np.complex64)
        otf[:psf.shape[0], :psf.shape[1]] = psf
        for axis, axis_size in enumerate(psf.shape):
            otf = np.roll(otf, -axis_size //2, axis = axis)
        return fft2(otf)
    
    def run(self, image):
        S = image.copy()
        if S.ndim == 2:
            S = S[..., np.newaxis]

        N, M, D = S.shape
        beta = 2*self._lambda
        
        psf_x = np.array([[-1,1]])
        otfx = self.psf2otf(psf_x, (N, M))
        psf_y = np.array([[-1],[1]])
        otfy = self.psf2otf(psf_y, (N, M))

        Normin1 = fft2(np.squeeze(S), axes = (0,1))
        Denormin2 = np.square(abs(otfx)) + np.square(abs(otfy))

        if D>1:
            Denormin2 = Denormin2[..., np.newaxis]
            Denormin2 = np.repeat(Denormin2, 3, axis = 2)

        for i in range(self.max_iter):
            if beta > self._beta_max:
                break

            Denormin = 1+beta*Denormin2
            
            h = np.diff(S, axis = 1)
            last_col = S[:,0:1, :] - S[:, -1:, :]
            h = np.hstack([h, last_col])

            v = np.diff(S, axis = 0)
            last_row = S[0:1, ...] - S[-1:, ...]
            v = np.vstack([v, last_row])

            grad = np.square(h) + np.square(v)
            if D>1:
                grad = np.sum(grad, axis = 2)
                idx = grad < (self._lambda / beta)
                idx = idx[..., np.newaxis]
                idx = np.repeat(idx,3,axis = 2)
            else:
                idx = grad< (self._lambda / beta)

            h[idx] = 0
            v[idx] = 0

            h_diff = -np.diff(h, axis = 1)
            first_col = h[:, -1:, :] - h[:, 0:1, :]
            h_diff = np.hstack([first_col, h_diff])

            v_diff = -np.diff(v, axis = 0)
            first_row = v[-1:,...] - v[0:1, ...]
            v_diff = np.vstack([first_row, v_diff])

            Normin2 = h_diff + v_diff
            Normin2 = beta*fft2(Normin2, axes = (0,1))

            FS  =(Normin1 + Normin2) / Denormin
            S = np.real(ifft2(FS, axes = (0,1)))

            if S.ndim<3:
                S = S[..., np.newaxis]
            beta*= self._kappa

        return S

def apply_median_filter(image_np, kernel_size = 5):
    if image_np.ndim == 3:
        filtered_np = np.zeros_like(image_np)
        for c in range(image_np.shape[2]):
            filtered_np[:, :, c] = median_filter(image_np[:, :, c], size = kernel_size)
    else:
        filtered_np = median_filter(image_np, size = kernel_size)
    return np.clip(filtered_np, 0, 1)

def safe_bm3d(noisy, sigma_psd):
    try:
        denoised = bm3d(noisy, sigma_psd=sigma_psd)

        if denoised is None:
            print("BM3D returned None, fallback to noisy")
            return noisy.copy()

        if np.isnan(denoised).any() or np.isinf(denoised).any():
            print("BM3D output invalid (NaN/Inf), fallback to noisy")
            return noisy.copy()
        
        if denoised.max() <= 1e-8:
            print("BM3D output nearly black, fallback to noisy")
            return noisy.copy()

        return denoised

    except Exception as e:
        print(f"BM3D failed: {e}, fallback to noisy")
        return noisy.copy()


def pair_downsampler(img):
    c = img.shape[1]
    
    filter1 = torch.tensor([[[[0, 0.5], [0.5, 0]]]], dtype=torch.float32, device=img.device)
    filter2 = torch.tensor([[[[0.5, 0], [0, 0.5]]]], dtype=torch.float32, device=img.device)
    
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter2 = filter2.repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)
    
    return output1, output2

def residual_loss(net, noisy_img):
    D1, D2 = pair_downsampler(noisy_img)
    
    denoised_D1 = D1 - net(D1)
    denoised_D2 = D2 - net(D2)
    
    loss1 = F.mse_loss(denoised_D1, D2) 
    loss2 = F.mse_loss(denoised_D2, D1) 
    
    return 0.5 * (loss1 + loss2)

def consistency_loss(net, noisy_img):
    D1, D2 = pair_downsampler(noisy_img)
    
    denoised_full = noisy_img - net(noisy_img)
    D1_full, D2_full = pair_downsampler(denoised_full)
    
    denoised_D1 = D1 - net(D1)
    denoised_D2 = D2 - net(D2)

    loss1 = F.mse_loss(denoised_D1, D1_full)
    loss2 = F.mse_loss(denoised_D2, D2_full)
    
    return 0.5 * (loss1 + loss2)

def total_loss(net, noisy_img, lambda_con=1):
    res_loss = residual_loss(net, noisy_img)
    con_loss = consistency_loss(net, noisy_img)
    return res_loss + lambda_con * con_loss

def total_loss_theta_z(net, noisy_img, z = None, beta = 0.0, lambda_con = 1):
    D1, D2 = pair_downsampler(noisy_img)

    denoised_D1 = D1 - net(D1)
    denoised_D2 = D2 - net(D2)

    res_loss = 0.5 * (F.mse_loss(denoised_D1, D2) + F.mse_loss(denoised_D2, D1))

    denoised_full = noisy_img - net(noisy_img)
    D1_full, D2_full = pair_downsampler(denoised_full)

    con_loss = 0.5 * (F.mse_loss(denoised_D1, D1_full) +
                      F.mse_loss(denoised_D2, D2_full))
    
    prior_loss = (z - (noisy_img - net(noisy_img))).pow(2).mean()

    total = res_loss + lambda_con * con_loss + (beta / 2) * prior_loss
    return total

def denoise(model, noisy_img):
    with torch.no_grad():
        noise_pred = model(noisy_img)
        denoised = noisy_img - noise_pred
        denoised = torch.clamp(denoised, 0, 1)
    return denoised

def to_float32(img):
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def train_step(model, optimizer, noisy_img):
    model.train()
    loss = total_loss(model, noisy_img)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def zero_shot_n2n_l0_joint(
    noisy_tensor,
    clean_tensor=None, 
    iterations=5,
    beta_init=0.1,
    beta_scale=2,
    lambda_l0=0.002,
    lambda_con=1,
    train_epochs=2000,
    verbose=False
):
    model = Network(noisy_tensor.shape[1]).to(device)
    z = torch.zeros_like(noisy_tensor).to(device)
    beta = beta_init

    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_is_post = False
    best_post_img = None  

    def eval_psnr(t_img):
        if clean_tensor is None:
            with torch.no_grad():
                return - total_loss(model, t_img).item()
        else:
            with torch.no_grad():
                out = denoise(model, t_img).clamp(0,1)
            out_np = out.squeeze(0).permute(1,2,0).cpu().numpy()
            gt_np  = clean_tensor.squeeze(0).permute(1,2,0).cpu().numpy()
            return peak_signal_noise_ratio(gt_np, out_np)

    for i in range(iterations):
        if verbose:
            print(f"[Joint] Iter {i+1}/{iterations}, beta={beta:.4f}")

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(train_epochs):
            loss = (residual_loss(model, noisy_tensor) + lambda_con * consistency_loss(model, noisy_tensor)) if i == 0 \
                   else total_loss_theta_z(model, noisy_tensor, z, beta, lambda_con)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        score_now = eval_psnr(noisy_tensor)
        if score_now > best_score:
            best_score = score_now
            best_state = copy.deepcopy(model.state_dict())
            best_is_post = False
            best_post_img = None

        with torch.no_grad():
            den0 = denoise(model, noisy_tensor).cpu().squeeze(0).permute(1,2,0).numpy()
        smoother_post = L0Smoothing(param_lambda=lambda_l0, max_iter=10)
        den0_post = np.clip(smoother_post.run(den0), 0, 1)
        if clean_tensor is not None:
            gt_np = clean_tensor.squeeze(0).permute(1,2,0).cpu().numpy()
            score_post = peak_signal_noise_ratio(gt_np, den0_post)
        else:
            den0_post_t = torch.from_numpy(den0_post).permute(2,0,1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                score_post = - total_loss(model, den0_post_t).item()

        if score_post > best_score:
            best_score = score_post
            best_state = copy.deepcopy(model.state_dict())
            best_is_post = True
            best_post_img = den0_post 

        smoother = L0Smoothing(param_lambda=lambda_l0, max_iter=10)
        z_np = np.clip(smoother.run(den0), 0, 1)
        z = torch.from_numpy(z_np).permute(2,0,1).unsqueeze(0).float().to(device)

        beta = min(beta * beta_scale, 10.0)

    model.load_state_dict(best_state)

    return model, best_is_post, best_post_img


def zero_shot_n2n_training(noisy_img, max_epochs=2000, patience=10, verbose=False):
    model = Network(noisy_img.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    best_loss = float('inf')
    counter = 0

    pbar = tqdm(range(max_epochs), desc="Training ZSN2N", leave=False) if verbose else range(max_epochs)
    
    for epoch in pbar:
        loss = train_step(model, optimizer, noisy_img)
        scheduler.step()

        if loss < best_loss - 1e-6:
            best_loss = loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break
    
    return model

def process_single_image(img_path, output_dir, image_name, noise_type):

    def _noise_label(nt):
        if isinstance(nt, str):
            return nt
        parts = []
        for typ, p in nt:
            if typ == 'gaussian':
                parts.append(f"gauss{p:.3f}")
            elif typ == 'poisson':
                parts.append(f"poiss{p:g}")
            elif typ == 'sp':
                parts.append(f"sp{p:.3f}")
            else:
                parts.append(f"{typ}{p}")
        return "mix_" + "_".join(parts)

    def _estimate_bm3d_sigma(nt) -> float:
 
        if isinstance(nt, str):
            if nt == 'gaussian':
                return 0.05
            elif nt == 'poisson':
                return 1.0 / np.sqrt(10) 
            elif nt == 'sp':
                return 0.05 
            else:
                return 0.05

        var_sum = 0.0
        for typ, p in nt:
            if typ == 'gaussian':
                var_sum += float(p) ** 2
            elif typ == 'poisson':
                var_sum += (1.0 / np.sqrt(float(p))) ** 2
            elif typ == 'sp':

                pass
        sigma = float(np.sqrt(var_sum)) if var_sum > 0 else 0.05
        return float(np.clip(sigma, 1e-3, 0.2))

    print(f"\nProcessing: {image_name} (Noise: {noise_type})")

    try:
        image = ToTensor()(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
        image_np = image.squeeze(0).permute(1,2,0).cpu().numpy()
        clean_img = image.clone()
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

    if isinstance(noise_type, str):
        if noise_type == 'gaussian':
            noise_param = 0.05
            bm3d_sigma = 0.05
        elif noise_type == 'poisson':
            noise_param = 10
            bm3d_sigma = 1.0 / np.sqrt(10)  # = sqrt(peak)/peak
        elif noise_type == 'sp':
            noise_param = 0.05
            bm3d_sigma = 0.05
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        noisy_img = add_noise(image, noise_param, noise_type)
        noise_tag = noise_type
    elif isinstance(noise_type, list):
        # [('gaussian',0.05), ('sp',0.02)]
        noisy_img = add_noise(image, None, noise_type)
        bm3d_sigma = _estimate_bm3d_sigma(noise_type)
        noise_tag = _noise_label(noise_type)
    else:
        raise ValueError("noise_type must be str or list[tuple] like [('gaussian',0.05),('sp',0.02)]")

    noisy_np = noisy_img.squeeze(0).permute(1,2,0).cpu().numpy()

    psnr_noisy = peak_signal_noise_ratio(image_np, noisy_np)
    ssim_noisy = ssim_numpy(image_np, noisy_np)

    results = {}
    results['image_name'] = image_name
    results['noise_type'] = noise_tag
    results['psnr_noisy'] = psnr_noisy

    # 1) L0 Smoothing
    print("  Running L0 Smoothing...")
    start_time = time()
    smoother = L0Smoothing(param_lambda=0.03, max_iter=10)
    smoothed_np = smoother.run(noisy_np)
    smoothed_np = np.clip(smoothed_np, 0, 1)
    time_l0 = time() - start_time

    psnr_smoothed = peak_signal_noise_ratio(to_float32(image_np), to_float32(smoothed_np))
    ssim_smoothed = ssim_numpy(to_float32(image_np), to_float32(smoothed_np))
    results['psnr_l0'] = psnr_smoothed
    results['ssim_l0'] = ssim_smoothed
    results['time_l0'] = time_l0

    # 2) Median Filter
    print("  Running Median Filter...")
    start_time = time()
    median_filtered_np = apply_median_filter(noisy_np, kernel_size=5)
    time_median = time() - start_time

    psnr_median = peak_signal_noise_ratio(image_np, median_filtered_np)
    ssim_median = ssim_numpy(image_np, median_filtered_np)
    results['psnr_median'] = psnr_median
    results['ssim_median'] = ssim_median
    results['time_median'] = time_median

    # 3) BM3D
    print(f"  Running BM3D (sigma≈{bm3d_sigma:.4f})...")
    start_time = time()
    try:
        bm3d_denoised_np = safe_bm3d(noisy_np, bm3d_sigma)
        bm3d_denoised_np = np.clip(bm3d_denoised_np, 0, 1)
        time_bm3d = time() - start_time

        psnr_bm3d = peak_signal_noise_ratio(image_np, bm3d_denoised_np)
        ssim_bm3d = ssim_numpy(image_np, bm3d_denoised_np)
        results['psnr_bm3d'] = psnr_bm3d
        results['ssim_bm3d'] = ssim_bm3d
        results['time_bm3d'] = time_bm3d
    except Exception as e:
        print(f"    BM3D failed: {e}")
        bm3d_denoised_np = np.zeros_like(noisy_np)
        results['psnr_bm3d'] = 0
        results['ssim_bm3d'] = 0
        results['time_bm3d'] = 0

    # 4) Zero-Shot N2N
    print("  Training Zero-Shot N2N...")
    start_time = time()
    zsn2n_model = zero_shot_n2n_training(noisy_img, max_epochs=2000, patience=10, verbose=False)
    time_zsn2n = time() - start_time

    final_denoised = denoise(zsn2n_model, noisy_img)
    final_denoised_np = final_denoised.cpu().squeeze(0).permute(1,2,0).numpy()

    psnr_zsn2n = peak_signal_noise_ratio(image_np, final_denoised_np)
    ssim_zsn2n = ssim_numpy(image_np, final_denoised_np)
    results['psnr_zsn2n'] = psnr_zsn2n
    results['ssim_zsn2n'] = ssim_zsn2n
    results['time_zsn2n'] = time_zsn2n

    # 5) ZSN2N + L0 (Post)
    print("  Running ZSN2N + L0 Post-processing...")
    start_time = time()
    smoother_post = L0Smoothing(param_lambda=0.002, max_iter=10)
    l0_on_zeroshot_np = smoother_post.run(final_denoised_np)
    l0_on_zeroshot_np = np.clip(l0_on_zeroshot_np, 0, 1)
    time_l0_post = time() - start_time

    psnr_l0_post = peak_signal_noise_ratio(image_np, l0_on_zeroshot_np)
    ssim_l0_post = ssim_numpy(image_np, l0_on_zeroshot_np)
    results['psnr_zsn2n_l0_post'] = psnr_l0_post
    results['ssim_zsn2n_l0_post'] = ssim_l0_post
    results['time_zsn2n_l0_post'] = time_l0_post

    # 6) ZSN2N + L0 (Joint)
    print("  Training ZSN2N + L0 Joint...")
    start_time = time()
    joint_model, best_is_post, best_post_img = zero_shot_n2n_l0_joint(
        noisy_img,
        clean_tensor=clean_img,  
        iterations=10,
        beta_init=2.0,
        beta_scale=2.0,
        lambda_l0=0.002,
        lambda_con=1,
        train_epochs=500,
        verbose=False
    )
    time_joint_training = time() - start_time

    if best_is_post and best_post_img is not None:
        joint_denoised_np = best_post_img
        time_joint_inference = 0
    else:
        start_time = time()
        joint_denoised = denoise(joint_model, noisy_img)
        joint_denoised_np = joint_denoised.cpu().squeeze(0).permute(1,2,0).numpy()
        time_joint_inference = time() - start_time

    psnr_joint = peak_signal_noise_ratio(image_np, joint_denoised_np)
    ssim_joint = ssim_numpy(image_np, joint_denoised_np)
    results['psnr_zsn2n_l0_joint'] = psnr_joint
    results['ssim_zsn2n_l0_joint'] = ssim_joint
    results['time_zsn2n_l0_joint'] = time_joint_training + time_joint_inference

    # ---------- 保存输出 ----------
    img_output_dir = os.path.join(output_dir, image_name.split('.')[0], noise_tag)
    os.makedirs(img_output_dir, exist_ok=True)

    to_pil = ToPILImage()
    images_to_save = [
        (image_np, "01_original.png"),
        (noisy_np, "02_noisy.png"),
        (smoothed_np, "03_l0_smoothing.png"),
        (median_filtered_np, "04_median_filter.png"),
        (bm3d_denoised_np, "05_bm3d.png"),
        (final_denoised_np, "06_zero_shot_n2n.png"),
        (l0_on_zeroshot_np, "07_zsn2n_l0_post.png"),
        (joint_denoised_np, "08_zsn2n_l0_joint.png")
    ]
    for img_data, filename in images_to_save:
        if img_data is not None and img_data.size > 0:
            img_pil = to_pil(torch.from_numpy(img_data).permute(2, 0, 1))
            img_pil.save(os.path.join(img_output_dir, filename))

    save_comparison_with_zoom(
        image_np, noisy_np, smoothed_np, final_denoised_np,
        median_filtered_np, l0_on_zeroshot_np, joint_denoised_np, bm3d_denoised_np,
        psnr_noisy, ssim_noisy,
        psnr_smoothed, ssim_smoothed,
        psnr_zsn2n, ssim_zsn2n,
        psnr_median, ssim_median,
        psnr_l0_post, ssim_l0_post,
        psnr_joint, ssim_joint,
        results['psnr_bm3d'], results['ssim_bm3d'],
        img_output_dir, "all_methods",
        zoom_y_start=70, zoom_y_end=170,
        zoom_x_start=170, zoom_x_end=270
    )

    print(f"  Results saved to: {img_output_dir}")
    return results


def save_comparison_with_zoom(
        clean_np, noisy_np, smoothed_np, denoised_np,
        median_filtered_np, l0_on_zeroshot_np, joint_denoised_np, bm3d_denoised_np,
        noisy_psnr, noisy_ssim, psnr_smoothed, smoothed_ssim,
        final_psnr, denoised_ssim, psnr_median, median_filtered_ssim,
        psnr_l0_on_zeroshot, ssim_l0_on_zeroshot, psnr_joint, ssim_joint,
        psnr_bm3d, ssim_bm3d,
        save_dir, base_filename,
        zoom_y_start = 100, zoom_y_end = 200, zoom_x_start = 150, zoom_x_end = 250
):
    os.makedirs(save_dir, exist_ok = True)

    clean_zoom = clean_np[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
    noisy_zoom = noisy_np[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
    smoothed_zoom = smoothed_np[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
    denoised_zoom = denoised_np[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
    median_zoom = median_filtered_np[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
    l0_zoom = l0_on_zeroshot_np[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
    joint_zoom = joint_denoised_np[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
    bm3d_zoom = bm3d_denoised_np[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]

    to_pil = ToPILImage()
    zoom_list = [
        (clean_zoom, "zoom_01_original.png"),
        (noisy_zoom, "zoom_02_noisy.png"),
        (smoothed_zoom, "zoom_03_l0.png"),
        (denoised_zoom, "zoom_04_zsn2n.png"),
        (median_zoom, "zoom_05_median.png"),
        (l0_zoom, "zoom_06_zsn2n_l0_post.png"),
        (joint_zoom, "zoom_07_zsn2n_l0_joint.png"),
        (bm3d_zoom, "zoom_08_bm3d.png"),
    ]
    for img_np, fname in zoom_list:
        to_pil(torch.from_numpy(img_np.transpose(2, 0, 1))).save(os.path.join(save_dir, fname))

    fig_combined, axes_combined = plt.subplots(2, 8, figsize=(32, 10))

    img_list = [
        (clean_np, "Original", None, None),
        (noisy_np, "Noisy", noisy_psnr, noisy_ssim),
        (smoothed_np, "L0", psnr_smoothed, smoothed_ssim),
        (denoised_np, "ZSN2N", final_psnr, denoised_ssim),
        (median_filtered_np, "Median", psnr_median, median_filtered_ssim),
        (l0_on_zeroshot_np, "ZSN2N+L0 Post", psnr_l0_on_zeroshot, ssim_l0_on_zeroshot),
        (joint_denoised_np, "ZSN2N+L0 Joint", psnr_joint, ssim_joint),
        (bm3d_denoised_np, "BM3D", psnr_bm3d, ssim_bm3d),
    ]
    for i, (img_np, title, psnr, ssim) in enumerate(img_list):
        axes_combined[0, i].imshow(img_np)
        axes_combined[0, i].set_title(title, fontsize=14)
        axes_combined[0, i].axis("off")
        if psnr is not None:
            axes_combined[0, i].text(
                0.5, -0.05, f"PSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}",
                transform=axes_combined[0, i].transAxes,
                fontsize=10, va="top", ha="center"
            )
    axes_combined[0, 0].add_patch(plt.Rectangle(
        (zoom_x_start, zoom_y_start),
        zoom_x_end - zoom_x_start,
        zoom_y_end - zoom_y_start,
        fill=False, edgecolor="red", linewidth=2
    ))

    zoom_img_list = [
        (clean_zoom, "Original"),
        (noisy_zoom, "Noisy"),
        (smoothed_zoom, "L0"),
        (denoised_zoom, "ZSN2N"),
        (median_zoom, "Median"),
        (l0_zoom, "ZSN2N+L0 Post"),
        (joint_zoom, "ZSN2N+L0 Joint"),
        (bm3d_zoom, "BM3D"),
    ]
    for i, (img_np, title) in enumerate(zoom_img_list):
        axes_combined[1, i].imshow(img_np)
        axes_combined[1, i].set_title(title, fontsize=12)
        axes_combined[1, i].axis("off")

    plt.tight_layout()
    fig_combined.savefig(os.path.join(save_dir, f"{base_filename}_comparison.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig_combined)

def main():
    print(f"Searching for images in: {INPUT_FOLDER}")
    
    image_files = get_image_files(INPUT_FOLDER)
    
    if not image_files:
        print(f"No image files found in {INPUT_FOLDER}")
        print(f"Supported extensions: {SUPPORTED_EXTENSIONS}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    all_results = []

    noise_types_to_test = [
        'gaussian',
        'poisson',
        'sp',
        [('gaussian', 0.1), ('sp', 0.1)],
        [('gaussian', 0.1), ('poisson', 50)],
        [('poisson', 50), ('sp', 0.1)],
        [('gaussian', 0.1), ('poisson', 50), ('sp', 0.1)],
    ]

    for i, img_path in enumerate(image_files):
        image_name = os.path.basename(img_path)
        print(f"\n[{i+1}/{len(image_files)}] Processing: {image_name}")
        
        for noise_type_current in noise_types_to_test:
            print(f"Noise type: {noise_type_current}")
            try:
                results = process_single_image(img_path, OUTPUT_FOLDER, image_name, noise_type_current)
                if results:
                    all_results.append(results)
            except Exception as e:
                print(f"Error processing {image_name} with {noise_type_current} noise: {e}")
                continue
    
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(OUTPUT_FOLDER, "batch_results.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total images processed: {len(all_results)}")
        print(f"Results saved to: {OUTPUT_FOLDER}")
        print(f"Detailed CSV report: {csv_path}")
        
        for noise_type in noise_types_to_test:
            subset = df[df['noise_type'] == noise_type]
            if not subset.empty:
                print(f"\n--- Results for {noise_type.upper()} noise ---")
                print(f"Images processed: {len(subset)}")
                
                print("Average PSNR Results:")
                methods = [col for col in subset.columns if col.startswith('psnr_') and col != 'psnr_noisy']
                for method in methods:
                    avg_psnr = subset[method].mean()
                    std_psnr = subset[method].std()
                    print(f"  {method.replace('psnr_', '').upper()}: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
                
                print("Average Processing Time:")
                time_methods = [col for col in subset.columns if col.startswith('time_')]
                for method in time_methods:
                    avg_time = subset[method].mean()
                    std_time = subset[method].std()
                    print(f"  {method.replace('time_', '').upper()}: {avg_time:.2f} ± {std_time:.2f}s")
        
        print(f"\n--- Overall Best Methods ---")
        methods = [col for col in df.columns if col.startswith('psnr_') and col != 'psnr_noisy']
        
        best_methods = {}
        for noise_type in noise_types_to_test:
            subset = df[df['noise_type'] == noise_type]
            if not subset.empty:
                best_method = subset[methods].mean().idxmax()
                best_psnr = subset[methods].mean().max()
                best_methods[noise_type] = (best_method, best_psnr)
        
        for noise_type, (method, psnr) in best_methods.items():
            method_name = method.replace('psnr_', '').upper()
            print(f"  {noise_type.upper()}: {method_name} ({psnr:.2f} dB)")

if __name__ == "__main__":
    main()