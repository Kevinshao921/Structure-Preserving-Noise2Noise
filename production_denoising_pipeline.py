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

try:
    from bm3d import bm3d
    BM3D_AVAILABLE = True
    print("BM3D library loaded successfully")
except ImportError as e:
    print(f"BM3D not available: {e}")
    BM3D_AVAILABLE = False

from time import time
import copy
from scipy.ndimage import median_filter
import pandas as pd
import traceback
import sys

device = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device('cpu')
)

print("device:", device)

INPUT_FOLDER = r"INPUT_PATH" 
OUTPUT_FOLDER = r"OUTPUT_PATH"
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
        try:
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
        except Exception as e:
            print(f"L0 Smoothing failed: {e}")
            return image.copy()
    
def apply_median_filter(image_np, kernel_size = 5):
    try:
        if image_np.ndim == 3:
            filtered_np = np.zeros_like(image_np)
            for c in range(image_np.shape[2]):
                filtered_np[:, :, c] = median_filter(image_np[:, :, c], size = kernel_size)
        else:
            filtered_np = median_filter(image_np, size = kernel_size)
        return np.clip(filtered_np, 0, 1)
    except Exception as e:
        print(f"Median filter failed: {e}")
        return np.clip(image_np, 0, 1)

def safe_bm3d(noisy, sigma_psd):
    if not BM3D_AVAILABLE:
        print("BM3D not available, returning original image")
        return noisy.copy()
        
    try:
        if noisy is None or noisy.size == 0:
            print("BM3D: Invalid input data")
            return noisy.copy()

        if np.isnan(noisy).any() or np.isinf(noisy).any():
            print("BM3D: Input contains NaN/Inf values")
            return noisy.copy()
            
        input_data = np.clip(noisy, 0, 1).astype(np.float64)
        if sigma_psd <= 0 or sigma_psd > 1:
            sigma_psd = 0.1
            print(f"BM3D: Adjusted sigma to {sigma_psd}")
        
        print(f"    Calling BM3D with sigma={sigma_psd}")
        
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("BM3D timeout")
        
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)
        
        try:
            denoised = bm3d(input_data, sigma_psd=sigma_psd)
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0) 

        if denoised is None:
            print("BM3D returned None")
            return noisy.copy()

        if np.isnan(denoised).any() or np.isinf(denoised).any():
            print("BM3D output contains NaN/Inf")
            return noisy.copy()
        
        if denoised.max() <= 1e-8:
            print("BM3D output is nearly black")
            return noisy.copy()
        
        denoised = np.clip(denoised, 0, 1)
        
        print("    BM3D completed successfully")
        return denoised

    except TimeoutError:
        print("BM3D timed out after 30 seconds")
        return noisy.copy()
    except MemoryError:
        print("BM3D: Out of memory")
        return noisy.copy()
    except Exception as e:
        print(f"BM3D failed with error: {type(e).__name__}: {e}")
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
    train_epochs=500, 
    verbose=False
):
    try:
        model = Network(noisy_tensor.shape[1]).to(device)
        z = torch.zeros_like(noisy_tensor).to(device)
        beta = beta_init

        best_score = float('inf') 
        best_state = copy.deepcopy(model.state_dict())
        best_is_post = False
        best_post_img = None  

        def eval_score(t_img):
            with torch.no_grad():
                return total_loss(model, t_img).item()

        patience = 3
        no_improve_count = 0
        
        for i in range(iterations):
            if verbose:
                print(f"[Joint] Iter {i+1}/{iterations}, beta={beta:.4f}")

            lr = 1e-3 if i < iterations//2 else 5e-4
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            epoch_patience = 50
            best_loss = float('inf')
            no_improve = 0
            
            for epoch in range(train_epochs):
                loss = (residual_loss(model, noisy_tensor) + lambda_con * consistency_loss(model, noisy_tensor)) if i == 0 \
                       else total_loss_theta_z(model, noisy_tensor, z, beta, lambda_con)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if loss.item() < best_loss - 1e-6:
                    best_loss = loss.item()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= epoch_patience:
                        break

            score_now = eval_score(noisy_tensor)
            
            if i >= iterations // 2:  
                with torch.no_grad():
                    den0 = denoise(model, noisy_tensor).cpu().squeeze(0).permute(1,2,0).numpy()
                
                smoother_post = L0Smoothing(param_lambda=lambda_l0, max_iter=5) 
                den0_post = np.clip(smoother_post.run(den0), 0, 1)
                
                den0_post_t = torch.from_numpy(den0_post).permute(2,0,1).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    score_post = total_loss(model, den0_post_t).item()

                if score_post < best_score:
                    best_score = score_post
                    best_state = copy.deepcopy(model.state_dict())
                    best_is_post = True
                    best_post_img = den0_post
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                den0 = denoise(model, noisy_tensor).cpu().squeeze(0).permute(1,2,0).numpy()
            
            if score_now < best_score:
                best_score = score_now
                best_state = copy.deepcopy(model.state_dict())
                best_is_post = False
                best_post_img = None
                no_improve_count = 0

            if i < iterations - 1: 
                smoother = L0Smoothing(param_lambda=lambda_l0, max_iter=5)  
                z_np = np.clip(smoother.run(den0), 0, 1)
                z = torch.from_numpy(z_np).permute(2,0,1).unsqueeze(0).float().to(device)

            beta = min(beta * beta_scale, 10.0)
            
            if no_improve_count >= patience:
                if verbose:
                    print(f"Early stopping at iteration {i+1}")
                break

        model.load_state_dict(best_state)
        return model, best_is_post, best_post_img
        
    except Exception as e:
        print(f"Joint optimization failed: {e}")
        model = Network(noisy_tensor.shape[1]).to(device)
        return model, False, None

def zero_shot_n2n_training(noisy_img, max_epochs=2000, patience=10, verbose=False):
    try:
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
    except Exception as e:
        print(f"Zero-shot N2N training failed: {e}")
        return Network(noisy_img.shape[1]).to(device)

def process_single_image(img_path, output_dir, image_name):
    print(f"\nProcessing: {image_name}")

    try:
        noisy_img = ToTensor()(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
        noisy_np = noisy_img.squeeze(0).permute(1,2,0).cpu().numpy()
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None
    
    bm3d_sigma = 0.1

    results = {}
    results['image_name'] = image_name

    print("  Running L0 Smoothing...")
    start_time = time()
    try:
        smoother = L0Smoothing(param_lambda=0.03, max_iter=10)
        smoothed_np = smoother.run(noisy_np)
        smoothed_np = np.clip(smoothed_np, 0, 1)
        time_l0 = time() - start_time
        
        results['time_l0'] = time_l0
        print(f"    L0 Smoothing completed in {time_l0:.2f}s")
    except Exception as e:
        print(f"    L0 Smoothing failed: {e}")
        smoothed_np = noisy_np.copy()
        results['time_l0'] = 0

    print("  Running Median Filter...")
    start_time = time()
    try:
        median_filtered_np = apply_median_filter(noisy_np, kernel_size=5)
        time_median = time() - start_time

        results['time_median'] = time_median
        print(f"    Median Filter completed in {time_median:.2f}s")
    except Exception as e:
        print(f"    Median Filter failed: {e}")
        median_filtered_np = noisy_np.copy()
        results['time_median'] = 0

    print("  Running BM3D...")
    bm3d_start = time()
    try:
        print("    Attempting BM3D denoising...")
        
        noisy_img_bm3d = noisy_np.copy().astype(np.float32)
        if noisy_img_bm3d.max() > 1.0:
            noisy_img_bm3d /= 255.0

        print("    BM3D input dtype:", noisy_img_bm3d.dtype,
                "min:", noisy_img_bm3d.min(),
                "max:", noisy_img_bm3d.max())

        h, w = noisy_img_bm3d.shape[:2]
        if h * w > 2000000: 
            print(f"    Image too large ({h}x{w}), skipping BM3D to avoid hanging")
            bm3d_denoised_np = noisy_np.copy()
            results['time_bm3d'] = 0
        else:
            bm3d_denoised_np = safe_bm3d(noisy_img_bm3d, sigma_psd=0.1)  
            bm3d_time = time() - bm3d_start
            print(f"    BM3D completed in {bm3d_time:.2f}s")
            results['time_bm3d'] = bm3d_time
            
    except Exception as e:
        print(f"    BM3D failed: {e}")
        bm3d_denoised_np = noisy_np.copy()  
        results['time_bm3d'] = 0

    print("  Training Zero-Shot N2N...")
    start_time = time()
    try:
        zsn2n_model = zero_shot_n2n_training(noisy_img, max_epochs=2000, patience=10, verbose=False)
        time_zsn2n = time() - start_time
        
        final_denoised = denoise(zsn2n_model, noisy_img)
        final_denoised_np = final_denoised.cpu().squeeze(0).permute(1,2,0).numpy()

        results['time_zsn2n'] = time_zsn2n
        print(f"    Zero-Shot N2N completed in {time_zsn2n:.2f}s")
    except Exception as e:
        print(f"    Zero-Shot N2N failed: {e}")
        final_denoised_np = noisy_np.copy()
        results['time_zsn2n'] = 0

    print("  Running ZSN2N + L0 Post-processing...")
    start_time = time()
    try:
        smoother_post = L0Smoothing(param_lambda=0.002, max_iter=10)
        l0_on_zeroshot_np = smoother_post.run(final_denoised_np)
        l0_on_zeroshot_np = np.clip(l0_on_zeroshot_np, 0, 1)
        time_l0_post = time() - start_time
        
        results['time_zsn2n_l0_post'] = time_l0_post
        print(f"    ZSN2N + L0 Post completed in {time_l0_post:.2f}s")
    except Exception as e:
        print(f"    ZSN2N + L0 Post failed: {e}")
        l0_on_zeroshot_np = final_denoised_np.copy()
        results['time_zsn2n_l0_post'] = 0

    print("  Training ZSN2N + L0 Joint...")
    start_time = time()
    try:
        joint_model, best_is_post, best_post_img = zero_shot_n2n_l0_joint(
            noisy_img,
            iterations=5, 
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

        results['time_zsn2n_l0_joint'] = time_joint_training + time_joint_inference
        print(f"    ZSN2N + L0 Joint completed in {time_joint_training + time_joint_inference:.2f}s")
    except Exception as e:
        print(f"    ZSN2N + L0 Joint failed: {e}")
        joint_denoised_np = final_denoised_np.copy()
        results['time_zsn2n_l0_joint'] = 0

    img_output_dir = os.path.join(output_dir, image_name.split('.')[0])
    os.makedirs(img_output_dir, exist_ok=True)
    
    to_pil = ToPILImage()
    images_to_save = [
        (noisy_np, "01_input_noisy.png"),
        (smoothed_np, "02_l0_smoothing.png"),
        (median_filtered_np, "03_median_filter.png"),
        (bm3d_denoised_np, "04_bm3d.png"),
        (final_denoised_np, "05_zero_shot_n2n.png"),
        (l0_on_zeroshot_np, "06_zsn2n_l0_post.png"),
        (joint_denoised_np, "07_zsn2n_l0_joint.png")
    ]
    
    for img_data, filename in images_to_save:
        try:
            if img_data is not None and img_data.size > 0:
                img_pil = to_pil(torch.from_numpy(img_data).permute(2,0,1))
                img_pil.save(os.path.join(img_output_dir, filename))
        except Exception as e:
            print(f"    Failed to save {filename}: {e}")

    try:
        save_comparison_with_zoom(
            noisy_np, smoothed_np, final_denoised_np,
            median_filtered_np, l0_on_zeroshot_np, joint_denoised_np, bm3d_denoised_np,
            img_output_dir, "all_methods_comparison"
        )
    except Exception as e:
        print(f"    Failed to save comparison plot: {e}")
    
    print(f"  Results saved to: {img_output_dir}")
    
    return results

def save_comparison_with_zoom(
        noisy_np, smoothed_np, denoised_np,
        median_filtered_np, l0_on_zeroshot_np, joint_denoised_np, bm3d_denoised_np,
        save_dir, base_filename,
        zoom_y_start = 100, zoom_y_end = 200, zoom_x_start = 150, zoom_x_end = 250
):
    try:
        os.makedirs(save_dir, exist_ok = True)
        h, w = noisy_np.shape[:2]
        zoom_y_start = max(0, min(zoom_y_start, h-100))
        zoom_y_end = min(h, zoom_y_start + 100)
        zoom_x_start = max(0, min(zoom_x_start, w-100))
        zoom_x_end = min(w, zoom_x_start + 100)

        noisy_zoom = noisy_np[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
        smoothed_zoom = smoothed_np[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
        denoised_zoom = denoised_np[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
        median_zoom = median_filtered_np[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
        l0_zoom = l0_on_zeroshot_np[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
        joint_zoom = joint_denoised_np[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
        bm3d_zoom = bm3d_denoised_np[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]

        to_pil = ToPILImage()
        zoom_list = [
            (noisy_zoom, "zoom_01_noisy.png"),
            (smoothed_zoom, "zoom_02_l0.png"),
            (denoised_zoom, "zoom_03_zsn2n.png"),
            (median_zoom, "zoom_04_median.png"),
            (l0_zoom, "zoom_05_zsn2n_l0_post.png"),
            (joint_zoom, "zoom_06_zsn2n_l0_joint.png"),
            (bm3d_zoom, "zoom_07_bm3d.png"),
        ]
        for img_np, fname in zoom_list:
            try:
                to_pil(torch.from_numpy(img_np.transpose(2, 0, 1))).save(os.path.join(save_dir, fname))
            except Exception as e:
                print(f"Failed to save {fname}: {e}")

        fig_combined, axes_combined = plt.subplots(2, 7, figsize=(28, 10))

        img_list = [
            (noisy_np, "Noisy"),
            (smoothed_np, "L0"),
            (denoised_np, "ZSN2N"),
            (median_filtered_np, "Median"),
            (l0_on_zeroshot_np, "ZSN2N+L0 Post"),
            (joint_denoised_np, "ZSN2N+L0 Joint"),
            (bm3d_denoised_np, "BM3D"),
        ]
        for i, (img_np, title) in enumerate(img_list):
            axes_combined[0, i].imshow(img_np)
            axes_combined[0, i].set_title(title, fontsize=14)
            axes_combined[0, i].axis("off")
            
        axes_combined[0, 0].add_patch(plt.Rectangle(
            (zoom_x_start, zoom_y_start),
            zoom_x_end - zoom_x_start,
            zoom_y_end - zoom_y_start,
            fill=False, edgecolor="red", linewidth=2
        ))

        zoom_img_list = [
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
    except Exception as e:
        print(f"Failed to create comparison with zoom: {e}")

def main():
    print(f"Searching for images in: {INPUT_FOLDER}")
    
    image_files = get_image_files(INPUT_FOLDER)
    
    if not image_files:
        print(f"No image files found in {INPUT_FOLDER}")
        print(f"Supported extensions: {SUPPORTED_EXTENSIONS}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    all_results = []

    for i, img_path in enumerate(image_files):
        image_name = os.path.basename(img_path)
        print(f"\n[{i+1}/{len(image_files)}] Processing: {image_name}")
        
        try:
            results = process_single_image(img_path, OUTPUT_FOLDER, image_name)
            if results:
                all_results.append(results)
            print(f"Successfully processed {image_name}")
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            traceback.print_exc()
            continue
    
    if all_results:
        try:
            df = pd.DataFrame(all_results)
            csv_path = os.path.join(OUTPUT_FOLDER, "batch_results.csv")
            df.to_csv(csv_path, index=False)
            
            print(f"\n{'='*60}")
            print("BATCH PROCESSING SUMMARY")
            print(f"{'='*60}")
            print(f"Total images processed: {len(all_results)}")
            print(f"Results saved to: {OUTPUT_FOLDER}")
            print(f"Detailed CSV report: {csv_path}")
            
            print("\nAverage Processing Times:")
            time_columns = [col for col in df.columns if col.startswith('time_')]
            for col in time_columns:
                if col in df.columns:
                    avg_time = df[col].mean()
                    method_name = col.replace('time_', '').replace('_', ' ').upper()
                    print(f"  {method_name}: {avg_time:.2f}s")
        except Exception as e:
            print(f"Error creating summary report: {e}")
    else:
        print("No images were successfully processed.")

if __name__ == "__main__":
    main()