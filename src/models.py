import numpy as np
from scipy import signal
from modules.downsample import blur, simulate_lr_from_hr
from modules.degradation import gaussian_kernel
from modules.upsample import upsample_nearest

def iterative_backprojection(lr, upsample=upsample_nearest, scale=4, iterations=20, alpha=1.0, size=9, sigma=1.6):
    """
    Iterative Back-Projection + Total Variation Denoising
    - lr: ảnh LR
    - scale: tỉ lệ phóng đại
    - kernel: kernel blur
    - iterations: số vòng lặp
    - alpha: hệ số bước
    - tv_weight: trọng số TV denoise
    """
    target_h = lr.shape[0] * scale
    target_w = lr.shape[1] * scale
    
    kernel = gaussian_kernel(size, sigma)

    x = upsample(lr, scale)

    for it in range(iterations):
        # Sinh LR giả lập
        sim_lr = simulate_lr_from_hr(x, scale, kernel)
        # Sai số ở LR
        err_lr = lr - sim_lr
        # Upsample sai số về HR
        err_up = upsample(err_lr, scale)
        # Back-projection
        flipped_kernel = np.flipud(np.fliplr(kernel))
        backproj = blur(err_up, flipped_kernel)
        # backproj = signal.convolve2d(err_up, flipped_kernel, mode='same', boundary='symm')
        # Cập nhật ảnh
        x = x + alpha * backproj
        x = np.clip(x, 0, 1)

    return x