# Wiener_Filter.py
"""
Traditional restoration pipeline (Wiener + RL + TV + optional pre/post denoise)

Dependencies:
    numpy, scipy, scikit-image (skimage), opencv-python
Install (if needed):
    pip install numpy scipy scikit-image opencv-python
"""

import os
import numpy as np
import cv2
from scipy.signal import wiener as adaptive_wiener
from numpy.fft import fft2, ifft2, ifftshift
from scipy.fft import fft2, ifft2, fftfreq, fftshift
from skimage import img_as_float, color
from skimage.transform import resize
from skimage.restoration import (
    unsupervised_wiener,
    richardson_lucy,
    denoise_tv_chambolle,
    estimate_sigma as skimage_estimate_sigma,
)
from scipy.ndimage import gaussian_filter
from src.models import iterative_backprojection_tv



def unsupervised_wiener_custom(image, psf_init=None, iterations=10, clip=True, balance=0.1):
    """
    Tái hiện hàm restoration.unsupervised_wiener của scikit-image.
    Input:
        image: Ảnh grayscale (2D array, float [0,1] hoặc [0,255])
        psf_init: Kernel khởi tạo (2D array, nếu None thì dùng Gaussian 5x5)
        iterations: Số lần lặp để tinh chỉnh kernel
        clip: Có clip ảnh khôi phục về [0,1] không
        balance: Tham số regularization (tương tự K trong Wiener filter)
    Output:
        deconvolved: Ảnh khôi phục
        psf: Kernel ước lượng
    """
    # Chuẩn hóa ảnh
    img = np.asarray(image, dtype=float)
    if img.max() > 1.0:
        img = img / 255.0  # Giả sử ảnh [0,255] nếu không chuẩn hóa

    # Khởi tạo kernel nếu không cung cấp
    if psf_init is None:
        psf = np.ones((5, 5)) / 25  # Kernel Gaussian nhỏ
        psf = gaussian_filter(psf, sigma=1)
        psf /= psf.sum()
    else:
        psf = np.asarray(psf_init, dtype=float)
        psf /= psf.sum()

    # Wiener filter cơ bản
    def wiener_step(img, kernel, K):
        kernel = kernel / np.sum(kernel)
        dummy = np.copy(img)
        dummy = fft2(dummy)
        kernel = fft2(kernel, s=img.shape)
        kernel = np.conj(kernel) / (np.abs(kernel)**2 + K)
        dummy = dummy * kernel
        dummy = np.abs(ifft2(dummy))
        return dummy

    # Lặp để tinh chỉnh kernel
    deconvolved = np.copy(img)
    for _ in range(iterations):
        # Bước 1: Khôi phục ảnh với kernel hiện tại
        deconvolved = wiener_step(img, psf, balance)
        if clip:
            deconvolved = np.clip(deconvolved, 0, 1)

        # Bước 2: Ước lượng kernel mới
        # Dùng cross-correlation giữa ảnh gốc và ảnh khôi phục
        psf_new = np.real(ifft2(fft2(img) * np.conj(fft2(deconvolved))))
        psf_new = np.clip(psf_new, 0, None)  # Giữ positive
        psf_new = psf_new[0:psf.shape[0], 0:psf.shape[1]]  # Cắt về kích thước ban đầu
        psf_new /= psf_new.sum() + 1e-6  # Normalize

        # Cập nhật kernel (tránh dao động lớn)
        psf = 0.5 * psf + 0.5 * psf_new

    # Wiener filter cuối cùng với kernel tối ưu
    deconvolved = wiener_step(img, psf, balance)
    if clip:
        deconvolved = np.clip(deconvolved, 0, 1)

    return deconvolved, psf


def unsupervised_wiener_improved(image, psf_init=5.0, reg=None, user_params=None, clip=True, rng=None):
    """
    Improved unsupervised Wiener filter for grayscale or RGB images using Gibbs sampler.
    Fixes quadrant swap and 180-degree flip issues by centering PSF and correct meshgrid indexing.
    Input:
        image: Grayscale (2D) or RGB (3D) float array [0,1]
        psf_init: Initial sigma for Gaussian PSF (float)
        reg: Regularization transfer function (ndarray, optional; default Laplacian)
        user_params: Dict with 'threshold' (1e-4), 'burnin' (15), 'min_num_iter' (30), 'max_num_iter' (200)
        clip: Clip output to [0,1]
        rng: np.random.Generator (optional)
    Output:
        deconvolved: Restored image (same shape as input)
        psf: Estimated PSF (2D array)
    """
    # Defaults
    if user_params is None:
        user_params = {'threshold': 1e-4, 'burnin': 15, 'min_num_iter': 30, 'max_num_iter': 200}
    if rng is None:
        rng = np.random.default_rng()

    # Check if image is RGB (3D) or grayscale (2D)
    image = np.asarray(image, dtype=float)
    is_rgb = len(image.shape) == 3 and image.shape[-1] == 3
    if is_rgb:
        restored_img = np.zeros_like(image)
        psf_final = None
        for c in range(3):
            restored_img[..., c], psf = unsupervised_wiener_improved(
                image[..., c], psf_init, reg, user_params, clip, rng
            )
            if c == 0:  # Store PSF from first channel (assume same for all)
                psf_final = psf
        return restored_img, psf_final

    # Normalize image to [0,1]
    if image.max() > 1.0:
        image /= 255.0

    # Shapes
    shape = image.shape
    N = shape[0] * shape[1]

    # Fourier grid for Laplacian regularization
    if reg is None:
        fx = fftfreq(shape[0])
        fy = fftfreq(shape[1])
        FX, FY = np.meshgrid(fx, fy)  # Correct order: x, y
        reg = 2 * (2 - np.cos(2 * np.pi * FX) - np.cos(2 * np.pi * FY))

    # Generate Gaussian PSF with correct centering
    def get_psf_sigma(sigma, shape):
        x = np.arange(-shape[0]//2, shape[0]//2)
        y = np.arange(-shape[1]//2, shape[1]//2)
        X, Y = np.meshgrid(x, y)  # Correct order: x, y
        psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        psf /= psf.sum()
        return fftshift(psf)  # Center the PSF

    psf = get_psf_sigma(psf_init, shape)
    Lambda_H = fft2(psf)  # PSF transfer function

    # Initial values
    gamma_eps = N / np.linalg.norm(fft2(image))**2
    gamma_1 = 1.0
    sigma_psf = psf_init
    ft_img = fft2(image)
    ft_x = np.copy(ft_img)

    # Gibbs sampling
    x_samples = []
    prev_mean = np.zeros(shape, dtype=complex)
    k = 0
    converged = False
    while not converged and k < user_params['max_num_iter']:
        # Step 1: Sample image circ x^(k+1)
        abs_Lambda_H_sq = np.abs(Lambda_H)**2
        Sigma_inv = gamma_eps * abs_Lambda_H_sq + gamma_1 * reg
        Sigma = 1 / (Sigma_inv + 1e-10)
        mu = gamma_eps * Sigma * np.conj(Lambda_H) * ft_img
        eta_real = rng.normal(0, 1, shape)
        eta_imag = rng.normal(0, 1, shape)
        eta = eta_real + 1j * eta_imag
        ft_x = mu + np.sqrt(Sigma) * eta / np.sqrt(2)

        # Step 2: Sample gamma_eps
        residual = ft_img - Lambda_H * ft_x
        beta_eps = np.linalg.norm(residual)**2 / 2
        alpha_eps = N / 2
        gamma_eps = rng.gamma(alpha_eps, 1 / beta_eps) if beta_eps > 0 else 1e-6

        # Step 3: Sample gamma_1
        dx = reg * np.abs(ft_x)**2
        beta_1 = np.sum(dx) / 2
        alpha_1 = (N - 1) / 2
        gamma_1 = rng.gamma(alpha_1, 1 / beta_1) if beta_1 > 0 else 1e-6

        # Step 4: Sample PSF sigma via Metropolis-Hastings
        sigma_p = 0.1 + rng.random() * 9.9
        psf_p = get_psf_sigma(sigma_p, shape)
        Lambda_H_p = fft2(psf_p)
        resid_old = ft_img - Lambda_H * ft_x
        resid_p = ft_img - Lambda_H_p * ft_x
        J = (gamma_eps / 2) * (np.linalg.norm(resid_old)**2 - np.linalg.norm(resid_p)**2)
        if np.log(rng.random()) < min(J, 0):
            sigma_psf = sigma_p
            Lambda_H = Lambda_H_p

        # Collect samples
        k += 1
        if k > user_params['burnin']:
            x_samples.append(ft_x)
            if len(x_samples) >= user_params['min_num_iter']:
                current_mean = np.mean(x_samples, axis=0)
                rel_change = np.linalg.norm(current_mean - prev_mean) / np.linalg.norm(current_mean)
                if rel_change < user_params['threshold']:
                    converged = True
                prev_mean = current_mean

    # Final deconvolved
    if not x_samples:
        x_samples = [ft_x]
    ft_mean = np.mean(x_samples, axis=0)
    deconvolved = np.real(ifft2(ft_mean))
    if clip:
        deconvolved = np.clip(deconvolved, 0, 1)

    # Estimated PSF
    estimated_psf = get_psf_sigma(sigma_psf, shape)

    return deconvolved, estimated_psf


# Assuming the following functions are defined elsewhere:
# - iterative_backprojection_tv(lr, scale, kernel, iterations=20, alpha=1.0, tv_weight=0.05)
# - unsupervised_wiener_improved(image, psf_init=5.0, reg=None, user_params=None, clip=True, rng=None)

def joint_iterative_backprojection_wiener(lr_img, scale=2, num_iters=50, psf_update_freq=10, 
                                          beta=0.7, wiener_iters=20, clip=True, rng=None):
    """
    Joint Iterative Backprojection with Unsupervised Wiener for blind super-resolution.
    Calls iterative_backprojection_tv and unsupervised_wiener_improved.
    
    Input:
        lr_img: Low-res degraded image (2D grayscale or 3D RGB, float [0,1])
        scale: Downsampling factor (int, e.g., 2 for 2x upsampling)
        num_iters: Max iterations for IBP loop
        psf_update_freq: Run Wiener every N iterations to update PSF (int)
        beta: Backprojection gain factor (float, 0.5-1.0 for stability)
        wiener_iters: Iterations for unsupervised_wiener_improved (int, reduce for speed)
        clip: Clip output to [0,1]
        rng: np.random.Generator for Wiener (optional)
    
    Output:
        hr_img: Restored high-res image (same type as input)
        estimated_psf: Final estimated PSF (2D array)
    """
    # Handle RGB
    is_rgb = len(np.shape(lr_img)) == 3 and lr_img.shape[-1] == 3
    if is_rgb:
        hr_img = np.zeros((lr_img.shape[0]*scale, lr_img.shape[1]*scale, 3))
        psf_final = None
        for c in range(3):
            hr_img[..., c], psf = joint_iterative_backprojection_wiener(
                lr_img[..., c], scale, num_iters, psf_update_freq, beta, wiener_iters, clip, rng
            )
            if c == 0:
                psf_final = psf
        if clip:
            hr_img = np.clip(hr_img, 0, 1)
        return hr_img, psf_final
    
    # Normalize
    img = np.asarray(lr_img, dtype=float)
    if img.max() > 1.0:
        img /= 255.0
    
    # Target HR shape
    hr_shape = (img.shape[0] * scale, img.shape[1] * scale)
    
    # Initialize HR: Bicubic upsample LR
    hr_current = resize(img, hr_shape, order=3, anti_aliasing=True)  # Bicubic
    
    # Initial PSF estimate: Run Wiener on upsampled LR (or directly on LR)
    small_img = resize(img, (min(128, hr_shape[0]), min(128, hr_shape[1])), order=3)
    _, initial_psf = unsupervised_wiener_improved(small_img)
    
    # Resize PSF to match
    psf_size = min(15, min(hr_shape))  # Limit PSF size
    initial_psf = resize(initial_psf, (psf_size, psf_size), order=0)  # Nearest for kernel
    initial_psf /= initial_psf.sum() + 1e-10  # Normalize
    psf = initial_psf.copy()
    
    # IBP loop
    prev_mse = float('inf')
    for iter in range(num_iters):
        # Call iterative_backprojection_tv with current PSF as kernel
        hr_current = iterative_backprojection_tv(img, scale, psf)
        
        # Update PSF periodically
        if (iter + 1) % psf_update_freq == 0:
            # Downsample HR to pseudo-LR
            pseudo_lr = resize(hr_current, img.shape, order=3, anti_aliasing=True)
            
            # Run Wiener to get new PSF
            _, new_psf = unsupervised_wiener_improved(
                pseudo_lr, psf_init=np.mean(np.diag(psf)) if psf.size > 1 else 1.0,
                clip=clip, rng=rng
            )
            
            # Resize and normalize new PSF
            new_psf = resize(new_psf, (psf_size, psf_size), order=0)
            new_psf /= new_psf.sum() + 1e-10
            
            # Smooth update to avoid oscillation
            psf = 0.7 * psf + 0.3 * new_psf
        
        # Simulate LR for MSE check
        blurred = np.real(ifft2(fft2(hr_current) * fft2(psf, s=hr_shape)))
        simulated_lr = resize(blurred, img.shape, order=3, anti_aliasing=True)
        error = img - simulated_lr
        mse = np.mean(error**2)
        
        if abs(prev_mse - mse) < 1e-6 or mse < 1e-6:
            break
        prev_mse = mse
    
    return hr_current, psf
