# Structure-Preserving Noise2Noise (SPN2N)

Implementation of **"Structure-Preserving Noise2Noise for Image Denoising with Zero-Shot Learning and L0 Gradient Minimization"**

**Accepted to the 2026 IEEE 6th International Conference on Power, Electronics and Computer Applications (ICPECA 2026).**
Changyang Sun\*, Yu Shao\* (\*equal contribution)

## Overview

SPN2N is a zero-shot, single-image denoising method that needs no clean images or paired training data. It combines:

- **Zero-Shot Noise2Noise**: a lightweight 3-layer CNN trained on pairs of sub-images downsampled from the noisy input, using residual and consistency losses.
- **L0 gradient minimization**: an edge-preserving prior that keeps flat regions smooth and edges sharp.
- **Half-Quadratic Splitting (HQS)**: alternates between updating the network parameters θ and the structure-preserving image z, with a penalty β that increases each round.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Edit the paths at the top of `SPN2N.py`:
   - `INPUT_FOLDER`: noisy images
   - `CLEAN_FOLDER`: clean references (optional, only used for PSNR/SSIM)
   - `OUTPUT_FOLDER`: where results are saved
2. Run:
   ```bash
   python SPN2N.py
   ```
3. For each image it saves `01_noisy.png`, `02_zsn2n.png`, `03_zsn2n_l0_joint.png` and `joint_best.pt`, plus a `batch_results.csv` summary.

## Results

Evaluated on CBSD68 with Gaussian, Poisson, salt-and-pepper and mixed noise, and on the PolyU real-world noisy image dataset. Across mixed-noise settings, SPN2N averages **24.98 dB / 0.744 (PSNR / SSIM)**, compared with ZSN2N + L0 post-processing (24.27 / 0.656), median filter (24.02 / 0.562), ZSN2N (23.95 / 0.608), BM3D (21.50 / 0.498) and L0 smoothing alone (15.63 / 0.240). It gets the best PSNR in 15 of 24 settings and the best SSIM in 23 of 24.

![Comparison of denoising methods](all_methods_comparison.png)

## Citation

BibTeX will be added once the paper is available on IEEE Xplore.

## References

- K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian. *Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering*. IEEE TIP, 2007. [[paper]](https://doi.org/10.1109/TIP.2007.901238)
- L. Xu, C. Lu, Y. Xu, and J. Jia. *Image Smoothing via L0 Gradient Minimization*. ACM TOG (SIGGRAPH Asia), 2011. [[paper]](https://doi.org/10.1145/2070781.2024208)
- J. Lehtinen et al. *Noise2Noise: Learning Image Restoration without Clean Data*. ICML, 2018. [[paper]](https://arxiv.org/abs/1803.04189)
- A. Krull, T.-O. Buchholz, and F. Jug. *Noise2Void: Learning Denoising from Single Noisy Images*. CVPR, 2019. [[paper]](https://doi.org/10.1109/CVPR.2019.00223)
- J. Batson and L. Royer. *Noise2Self: Blind Denoising by Self-Supervision*. ICML, 2019. [[paper]](https://arxiv.org/abs/1901.11365)
- Y. Mansour and R. Heckel. *Zero-Shot Noise2Noise: Efficient Image Denoising without any Data*. CVPR, 2023. [[paper]](https://arxiv.org/abs/2303.11253)
- D. Geman and C. Yang. *Nonlinear Image Recovery with Half-Quadratic Regularization*. IEEE TIP, 1995.
- K. Zhang, W. Zuo, and L. Zhang. *FFDNet: Toward a Fast and Flexible Solution for CNN-based Image Denoising*. IEEE TIP, 2018. [[paper]](https://doi.org/10.1109/TIP.2018.2839891)
- S. W. Zamir et al. *Restormer: Efficient Transformer for High-Resolution Image Restoration*. CVPR, 2022. [[paper]](https://arxiv.org/abs/2111.09881)
- J. Liang et al. *SwinIR: Image Restoration Using Swin Transformer*. ICCV Workshops, 2021. [[paper]](https://arxiv.org/abs/2108.10257)

## License

MIT
