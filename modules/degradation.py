import cv2
import numpy as np

def gaussian_kernel(size=9, sigma=1.6):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)

def blur(image, kernel):
    """
    Hàm làm mờ hỗ trợ cả ảnh xám và ảnh màu.
    Dùng cv2.filter2D để thay thế vòng lặp for (nhanh hơn rất nhiều).
    """
    # cv2.filter2D tự động xử lý từng kênh màu nếu là ảnh màu
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)

def simulate_lr_from_hr(hr_est, scale, kernel):
    """Mô phỏng LR từ HR: Blur + Downsample (làm thủ công)."""
    # Bước 1: Blur
    blurred = blur(hr_est, kernel)
    # Bước 2: Downsample
    lr = blurred[::scale, ::scale]

    return lr

def degrade_image(hr, scale=4, kernel=None, noise_type="gaussian", noise_std=0.01):
    if kernel is None:
        kernel = gaussian_kernel(9, 1.6)

    # Blur
    hr_blur = blur(hr, kernel)

    # Downsample
    lr = hr_blur[::scale, ::scale]

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
