# import numpy as np
# from scipy import signal, ndimage

# def gaussian_kernel(k=9, sigma=1.6):
#     ax = np.arange(-k//2+1., k//2+1.)
#     xx, yy = np.meshgrid(ax, ax)
#     kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma **2))
#     return kernel / np.sum(kernel)
    
# def degrade_image(hr, scale=4, kernel=None, noise_std=0.01):
#     if kernel is None:
#         kernel = gaussian_kernel(9,1.6)
    
#     # Bước 1: Blur
#     blurred = signal.convolve2d(hr, kernel, mode='same', boundary='symm')
#     lr = blurred[::scale, ::scale]
#     noisy = lr + np.random.normal(0, noise_std, lr.shape)
#     noisy = np.clip(noisy, 0, 1)
#     return noisy
    
import cv2
import numpy as np
from scipy import signal

def gaussian_kernel(size=9, sigma=1.6):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)

def degrade_image(hr, scale=4, kernel=None, noise_type="gaussian", noise_std=0.01):
    if kernel is None:
        kernel = gaussian_kernel(9, 1.6)

    # Handle both grayscale and color images
    if hr.ndim == 2:  # Grayscale image
        # Blur
        hr_blur = signal.convolve2d(hr, kernel, mode="same", boundary="symm")
        # Downsample
        lr = hr_blur[::scale, ::scale]
    else:  # Color image (3D)
        hr_blur = np.zeros_like(hr)
        # Apply blur to each channel separately
        for c in range(hr.shape[2]):
            hr_blur[:, :, c] = signal.convolve2d(hr[:, :, c], kernel, mode="same", boundary="symm")
        # Downsample
        lr = hr_blur[::scale, ::scale, :]

    # Add noise
    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_std, lr.shape)
    elif noise_type == "rayleigh":
        noise = np.random.rayleigh(scale=noise_std, size=lr.shape)
    elif noise_type == "gamma":
        noise = np.random.gamma(shape=2.0, scale=noise_std, size=lr.shape)
    elif noise_type == "exponential":
        noise = np.random.exponential(scale=noise_std, size=lr.shape)
    elif noise_type == "uniform":
        noise = np.random.uniform(-noise_std, noise_std, size=lr.shape)
    elif noise_type == "saltpepper":
        prob = noise_std
        noise = np.zeros_like(lr)
        mask = np.random.rand(*lr.shape)
        lr[mask < prob/2] = 0
        lr[mask > 1 - prob/2] = 1
        return lr
    else:
        noise = 0

    lr_noisy = np.clip(lr + noise, 0, 1)
    return lr_noisy
