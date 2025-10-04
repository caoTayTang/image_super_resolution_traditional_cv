import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal

# def blur(image, kernel_size=3):
#     # kernel mean blur
#     kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

#     pad = kernel_size // 2
#     padded_img = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant')

#     h, w, c = image.shape
#     blurred = np.zeros_like(image)

#     for y in range(h):
#         for x in range(w):
#             for ch in range(c):
#                 region = padded_img[y:y+kernel_size, x:x+kernel_size, ch]
#                 blurred[y, x, ch] = np.sum(region * kernel)
#     return blurred

def blur(image, kernel):
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    if image.ndim == 2:  # ảnh grayscale
        image = image[:, :, None]

    h, w, c = image.shape
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
    out = np.zeros_like(image)

    for y in range(h):
        for x in range(w):
            region = padded[y:y+k_h, x:x+k_w]
            for ch in range(c):
                out[y, x, ch] = np.sum(region[:, :, ch] * kernel)

    return out if c > 1 else out[:, :, 0]

def simulate_lr_from_hr(hr_est, scale, kernel):
    """Mô phỏng LR từ HR: Blur + Downsample (làm thủ công)."""
    # Bước 1: Blur
    blurred = blur(hr_est, kernel)
    # Bước 2: Downsample
    lr = blurred[::scale, ::scale]

    return lr