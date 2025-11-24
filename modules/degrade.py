import cv2
import numpy as np

def gaussian_kernel(size=9, sigma=1.6):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)

def blur(image, kernel):
    """Hỗ trợ ảnh màu và xám"""
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)

def degrade_image(hr, scale, kernel=None, noise_std=0.0):
    if kernel is None:
        kernel = gaussian_kernel(9, 1.6)
    
    # 1. Blur
    hr_blur = blur(hr, kernel)
    
    # 2. Downsample
    lr = hr_blur[::scale, ::scale]
    
    # 3. Add Noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, lr.shape)
        lr = lr + noise
    
    return np.clip(lr, 0.0, 1.0).astype(np.float32)

def simulate_lr_from_hr(hr_est, scale, kernel):
    blurred = blur(hr_est, kernel)
    return blurred[::scale, ::scale]


