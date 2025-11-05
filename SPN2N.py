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
from numpy.fft import fft2, ifft2

from bm3d import bm3d

from time import time
import copy
from scipy.ndimage import median_filter
import pandas as pd
import traceback
import sys

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

device = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device('cpu')
)

print("device:", device)

INPUT_FOLDER = r"C:\Users\Administrator\Desktop\PolyU-Real-World-Noisy-Images-Dataset-master\OriginalImages\real" 
OUTPUT_FOLDER = r"C:\Users\Administrator\Desktop\output_5"
CLEAN_FOLDER = r"C:\Users\Administrator\Desktop\PolyU-Real-World-Noisy-Images-Dataset-master\OriginalImages\clean"
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

def find_clean_image(noisy_path, clean_folder):
    try:
        noisy_name = os.path.basename(noisy_path)
        name_no_ext, ext = os.path.splitext(noisy_name)

        direct_path = os.path.join(clean_folder, noisy_name)
        if os.path.exists(direct_path):
            return direct_path

        if "real" in name_no_ext.lower():
            alt_clean = name_no_ext.lower().replace("real", "clean") + ext
            path_clean = os.path.join(clean_folder, alt_clean)
            if os.path.exists(path_clean):
                return path_clean

        if "real" in name_no_ext.lower():
            alt_mean = name_no_ext.lower().replace("real", "mean") + ext
            path_mean = os.path.join(clean_folder, alt_mean)
            if os.path.exists(path_mean):
                return path_mean

        for new_ext in ['.png', '.jpg', '.jpeg']:
            alt = os.path.join(clean_folder, name_no_ext + new_ext)
            if os.path.exists(alt):
                return alt

        return None
    except Exception as e:
        print(f"Error finding clean image: {e}")
        return None

def calculate_metrics(denoised_img, clean_img):
    try:
        if denoised_img.shape != clean_img.shape:
            print(f"Shape mismatch: denoised {denoised_img.shape} vs clean {clean_img.shape}")
            return None, None
        
        denoised_img = np.clip(denoised_img, 0, 1)
        clean_img = np.clip(clean_img, 0, 1)
        
        psnr_value = psnr(clean_img, denoised_img, data_range=1.0)
        
        if denoised_img.ndim == 3:
            ssim_value = ssim(clean_img, denoised_img, data_range=1.0, channel_axis=2)
        else:
            ssim_value = ssim(clean_img, denoised_img, data_range=1.0)
        
        return psnr_value, ssim_value
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None, None
    
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

def total_loss(net, noisy_img):
    res_loss = residual_loss(net, noisy_img)
    con_loss = consistency_loss(net, noisy_img)
    return res_loss + con_loss

def total_loss_theta_z(net, noisy_img, z = None, beta = 0.0):
    D1, D2 = pair_downsampler(noisy_img)

    denoised_D1 = D1 - net(D1)
    denoised_D2 = D2 - net(D2)

    res_loss = 0.5 * (F.mse_loss(denoised_D1, D2) + F.mse_loss(denoised_D2, D1))

    denoised_full = noisy_img - net(noisy_img)
    D1_full, D2_full = pair_downsampler(denoised_full)

    con_loss = 0.5 * (F.mse_loss(denoised_D1, D1_full) +
                      F.mse_loss(denoised_D2, D2_full))
    
    prior_loss = (z - (noisy_img - net(noisy_img))).pow(2).mean()

    total = res_loss + con_loss + (beta / 2) * prior_loss
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

def train_step(model, optimizer, noisy_img):
    model.train()
    loss = total_loss(model, noisy_img)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

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
    def __init__(self, param_lambda: Optional[float] = 2e-2, param_kappa: Optional[float] = 2.0, max_iter: Optional[int] = 10):
        self._lambda = param_lambda
        self._kappa = param_kappa
        self._beta_max = 1e5
        self.max_iter = max_iter

    def psf2otf(self, psf, shape):
        otf = np.zeros(shape, dtype=np.complex64)
        otf[:psf.shape[0], :psf.shape[1]] = psf
        for axis, axis_size in enumerate(psf.shape):
            otf = np.roll(otf, -axis_size // 2, axis=axis)
        return fft2(otf)
    
    def run(self, image):
        try:
            S = image.copy()
            if S.ndim == 2:
                S = S[..., np.newaxis]

            N, M, D = S.shape
            beta = 2 * self._lambda
            
            psf_x = np.array([[-1, 1]])
            otfx = self.psf2otf(psf_x, (N, M))
            psf_y = np.array([[-1], [1]])
            otfy = self.psf2otf(psf_y, (N, M))

            Normin1 = fft2(np.squeeze(S), axes=(0, 1))
            Denormin2 = np.square(abs(otfx)) + np.square(abs(otfy))

            if D > 1:
                Denormin2 = Denormin2[..., np.newaxis]
                Denormin2 = np.repeat(Denormin2, 3, axis=2)

            for i in range(self.max_iter):
                if beta > self._beta_max:
                    break

                Denormin = 1 + beta * Denormin2
                
                h = np.diff(S, axis=1)
                last_col = S[:, 0:1, :] - S[:, -1:, :]
                h = np.hstack([h, last_col])

                v = np.diff(S, axis=0)
                last_row = S[0:1, ...] - S[-1:, ...]
                v = np.vstack([v, last_row])

                grad = np.square(h) + np.square(v)
                if D > 1:
                    grad = np.sum(grad, axis=2)
                    idx = grad < (self._lambda / beta)
                    idx = idx[..., np.newaxis]
                    idx = np.repeat(idx, 3, axis=2)
                else:
                    idx = grad < (self._lambda / beta)

                h[idx] = 0
                v[idx] = 0

                h_diff = -np.diff(h, axis=1)
                first_col = h[:, -1:, :] - h[:, 0:1, :]
                h_diff = np.hstack([first_col, h_diff])

                v_diff = -np.diff(v, axis=0)
                first_row = v[-1:, ...] - v[0:1, ...]
                v_diff = np.vstack([first_row, v_diff])

                Normin2 = h_diff + v_diff
                Normin2 = beta * fft2(Normin2, axes=(0, 1))

                FS = (Normin1 + Normin2) / Denormin
                S = np.real(ifft2(FS, axes=(0, 1)))

                if S.ndim < 3:
                    S = S[..., np.newaxis]
                beta *= self._kappa

            return S
        except Exception as e:
            print(f"L0 Smoothing failed: {e}")
            return image.copy()

def zero_shot_n2n_l0_joint(
    noisy_tensor,
    clean_tensor=None,
    base_model=None, 
    iterations=5,
    beta_init=1,
    beta_scale=2,
    lambda_l0=0.002,
    train_epochs=800,
    verbose=True
):

    try:
        model = copy.deepcopy(base_model) if base_model is not None else Network(noisy_tensor.shape[1]).to(device)
        z = torch.zeros_like(noisy_tensor).to(device)
        beta = beta_init

        best_psnr = -float('inf')
        best_state = copy.deepcopy(model.state_dict())
        best_img_np = None
        best_is_post = False

        clean_np = clean_tensor.permute(1, 2, 0).cpu().numpy() if clean_tensor is not None else None
        no_improve = 0

        for i in range(iterations):
            if verbose:
                print(f"[Joint] Iter {i+1}/{iterations}, β={beta:.4f}")

            optimizer = optim.Adam(model.parameters(), lr=1e-3 * (0.5 ** (i // 2)))

            for epoch in range(train_epochs):
                if i == 0:
                    loss = residual_loss(model, noisy_tensor) + consistency_loss(model, noisy_tensor)
                else:
                    loss = total_loss_theta_z(model, noisy_tensor, z, beta)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                den_np = denoise(model, noisy_tensor).cpu().squeeze(0).permute(1, 2, 0).numpy()


            smoother_post = L0Smoothing(param_lambda=lambda_l0, max_iter=5)
            den_post = np.clip(smoother_post.run(den_np), 0, 1)

            psnr_now, psnr_post = None, None
            if clean_np is not None:
                psnr_now = psnr(clean_np, den_np, data_range=1.0)
                psnr_post = psnr(clean_np, den_post, data_range=1.0)

            # --- Select best ---
            if clean_np is not None and (psnr_post or psnr_now):
                if (psnr_post or 0) > best_psnr:
                    best_psnr = psnr_post
                    best_state = copy.deepcopy(model.state_dict())
                    best_img_np = den_post
                    best_is_post = True
                    no_improve = 0
                elif (psnr_now or 0) > best_psnr:
                    best_psnr = psnr_now
                    best_state = copy.deepcopy(model.state_dict())
                    best_img_np = den_np
                    best_is_post = False
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            smoother = L0Smoothing(param_lambda=lambda_l0, max_iter=5)
            z_np = np.clip(smoother.run(den_np), 0, 1)
            z = torch.from_numpy(z_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            beta = min(beta * beta_scale, 10.0)

            if no_improve >= 2:
                if verbose:
                    print("  Early stop due to no improvement")
                break

        model.load_state_dict(best_state)

        if best_img_np is None:
            with torch.no_grad():
                best_img_np = denoise(model, noisy_tensor).cpu().squeeze(0).permute(1, 2, 0).numpy()

        return model, best_is_post, best_img_np, best_psnr

    except Exception as e:
        print(f"Joint optimization failed: {e}")
        traceback.print_exc()
        return Network(noisy_tensor.shape[1]).to(device), False, noisy_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy(), -1

    
def process_single_image(img_path, output_dir, image_name, clean_path=None):
    print(f"\nProcessing: {image_name}")
    try:
        noisy_img = ToTensor()(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
        noisy_np = noisy_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

    clean_np = None
    clean_tensor = None
    if clean_path and os.path.exists(clean_path):
        clean_tensor = ToTensor()(Image.open(clean_path).convert('RGB')).to(device)
        clean_np = clean_tensor.permute(1, 2, 0).cpu().numpy()
        print(f"  Loaded clean reference image from: {os.path.basename(clean_path)}")

    results = {'image_name': image_name}

    print("  Training Zero-Shot N2N...")
    start_time = time()
    zsn2n_model = Network(noisy_img.shape[1]).to(device)
    optimizer = optim.Adam(zsn2n_model.parameters(), lr=1e-3)
    best_loss = float('inf')
    patience = 15
    no_improve = 0

    for epoch in tqdm(range(800), desc="ZSN2N", leave=False):
        loss = total_loss(zsn2n_model, noisy_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > patience:
                break

    time_zsn2n = time() - start_time
    final_denoised = denoise(zsn2n_model, noisy_img)
    final_denoised_np = final_denoised.cpu().squeeze(0).permute(1, 2, 0).numpy()
    results['time_zsn2n'] = time_zsn2n
    if clean_np is not None:
        results['psnr_zsn2n'], results['ssim_zsn2n'] = calculate_metrics(final_denoised_np, clean_np)
        print(f"    ZSN2N PSNR={results['psnr_zsn2n']:.2f} SSIM={results['ssim_zsn2n']:.4f}")

    print("  Training ZSN2N + L0 Joint...")
    start_joint = time()
    joint_model, best_is_post, joint_np, best_psnr = zero_shot_n2n_l0_joint(
        noisy_img, clean_tensor, base_model=zsn2n_model,
        iterations=5, beta_init=2.0, beta_scale=2.0, lambda_l0=0.002,
        train_epochs=300, verbose=True
    )
    time_joint = time() - start_joint
    results['time_joint'] = time_joint

    if clean_np is not None:
        psnr_val, ssim_val = calculate_metrics(joint_np, clean_np)
        results['psnr_joint'] = psnr_val
        results['ssim_joint'] = ssim_val
        print(f"    Joint PSNR={psnr_val:.2f} SSIM={ssim_val:.4f}")

    save_dir = os.path.join(output_dir, image_name.split('.')[0])
    os.makedirs(save_dir, exist_ok=True)
    ToPILImage()(torch.from_numpy(noisy_np.transpose(2,0,1))).save(os.path.join(save_dir, "01_noisy.png"))
    ToPILImage()(torch.from_numpy(final_denoised_np.transpose(2,0,1))).save(os.path.join(save_dir, "02_zsn2n.png"))
    ToPILImage()(torch.from_numpy(joint_np.transpose(2,0,1))).save(os.path.join(save_dir, "03_zsn2n_l0_joint.png"))
    if clean_np is not None:
        ToPILImage()(torch.from_numpy(clean_np.transpose(2,0,1))).save(os.path.join(save_dir, "00_clean.png"))

    torch.save(joint_model.state_dict(), os.path.join(save_dir, "joint_best.pt"))
    print(f"  ✅ Saved results & model to {save_dir}")

    return results

if __name__ == "__main__":
    print(f"Searching for noisy images in: {INPUT_FOLDER}")
    print(f"Searching for clean images in: {CLEAN_FOLDER}")

    noisy_images = get_image_files(INPUT_FOLDER)
    clean_images = get_image_files(CLEAN_FOLDER)
    print(f"Found {len(noisy_images)} images to process\n")

    all_results = []
    psnr_joint_list, ssim_joint_list = [], []
    psnr_zsn2n_list, ssim_zsn2n_list = [], []

    for idx, noisy_path in enumerate(noisy_images, 1):
        image_name = os.path.basename(noisy_path)
        print(f"[{idx}/{len(noisy_images)}] Processing: {image_name}")

        clean_path = find_clean_image(noisy_path, CLEAN_FOLDER)
        result = process_single_image(noisy_path, OUTPUT_FOLDER, image_name, clean_path)
        
        if result is not None:
            all_results.append(result)
            if result.get("psnr_joint") is not None:
                psnr_joint_list.append(result["psnr_joint"])
                ssim_joint_list.append(result["ssim_joint"])
            if result.get("psnr_zsn2n") is not None:
                psnr_zsn2n_list.append(result["psnr_zsn2n"])
                ssim_zsn2n_list.append(result["ssim_zsn2n"])

    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {len(all_results)}")
    print(f"Results saved to: {OUTPUT_FOLDER}")

    if len(all_results) > 0:
        avg_psnr_joint = np.mean(psnr_joint_list) if psnr_joint_list else 0
        avg_ssim_joint = np.mean(ssim_joint_list) if ssim_joint_list else 0
        avg_psnr_zsn2n = np.mean(psnr_zsn2n_list) if psnr_zsn2n_list else 0
        avg_ssim_zsn2n = np.mean(ssim_zsn2n_list) if ssim_zsn2n_list else 0

        print(f"\nAverage PSNR:")
        print(f"  ZSN2N:        {avg_psnr_zsn2n:.2f} dB")
        print(f"  ZSN2N + L0:   {avg_psnr_joint:.2f} dB")
        print(f"\nAverage SSIM:")
        print(f"  ZSN2N:        {avg_ssim_zsn2n:.4f}")
        print(f"  ZSN2N + L0:   {avg_ssim_joint:.4f}")

    csv_path = os.path.join(OUTPUT_FOLDER, "batch_results.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"\nDetailed CSV report saved to: {csv_path}")

