import numpy as np
from modules.degradation import gaussian_kernel
from modules.degradation import blur, simulate_lr_from_hr
from modules.upsample import upsample_bicubic

def iterative_backprojection(
    lr,
    upsample=upsample_bicubic,
    scale=4,
    iterations=20,
    alpha=1.0,
    size=9,
    sigma=1.6,
    kernel=None,
):
    """
    Iterative Back-Projection (IBP) with optional enhancements.
    Parameters:
    -----------
    lr : np.ndarray
        Low-resolution input image (normalized 0-1).
    upsample : callable
        Upsampling function (default: nearest).
    scale : int
        Scale factor between LR and HR.
    iterations : int
        Number of IBP iterations.
    alpha : float
        Step size (learning rate).
    kernel : np.ndarray or None
        Blur kernel. If None, Gaussian kernel is used.
    size, sigma : int, float
        Parameters for Gaussian kernel.

    Returns:
    --------
    x : np.ndarray
        The reconstructed HR image
    """

    # Khởi tạo kernel ---
    if kernel is None:
        kernel = gaussian_kernel(size, sigma)

    # Khởi tạo ảnh HR nội suy ---
    x = upsample(lr, scale)

    # Vòng lặp IBP ---
    for it in range(iterations):
        print(f"[IBP] Iteration {it+1}/{iterations}")
        
        sim_lr = simulate_lr_from_hr(x, scale, kernel)
        
        err_lr = lr - sim_lr
        err_up = upsample(err_lr, scale)

        flipped_kernel = np.flipud(np.fliplr(kernel))
        backproj = blur(err_up, flipped_kernel)

        x = x + alpha * backproj
        x = np.clip(x, 0, 1)
    return x