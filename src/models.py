import numpy as np
from skimage.restoration import denoise_tv_chambolle
from modules.degradation.degrade import gaussian_kernel
from modules.degradation.downsample import blur, simulate_lr_from_hr
from modules.upsample import upsample_nearest

def iterative_backprojection(
    lr,
    upsample=upsample_nearest,
    scale=4,
    iterations=20,
    alpha=1.0,
    size=9,
    sigma=1.6,
    kernel=None,
    denoise=False,
    adaptive_alpha=False,
    dynamic_blur=False,
    early_stop=False,
    return_mse=False,
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
    denoise : bool
        Apply Total Variation denoising after each update.
    adaptive_alpha : bool
        Enable adaptive alpha based on LR error.
    dynamic_blur : bool
        Reduce blur sigma over iterations (coarse-to-fine).
    early_stop : bool
        Stop early if error converges.
    return_mse : bool
        Return per-iteration MSE (for visualization).

    Returns:
    --------
    x : np.ndarray
        The reconstructed HR image.
    mse_list (optional) : list
        List of MSE values per iteration (if return_mse=True).
    """

    # --- 1️⃣ Khởi tạo kernel ---
    if kernel is None:
        kernel = gaussian_kernel(size, sigma)

    # --- 2️⃣ Khởi tạo ảnh HR nội suy ---
    x = upsample(lr, scale)

    mse_list = []
    prev_err = np.inf

    # --- 3️⃣ Vòng lặp IBP ---
    for it in range(iterations):
        if np.isnan(x).any():
            print(f"[DEBUG] NaN detected at iteration {it}")
            break
        
        # (a) Mô phỏng ảnh LR từ HR
        sim_lr = simulate_lr_from_hr(x, scale, kernel)

        # (b) Tính sai số giữa LR thật và LR mô phỏng
        err_lr = lr - sim_lr
        mse = np.mean(err_lr**2)
        mse_list.append(mse)
        print(f"Iter {it}: MSE={mse:.6f}, alpha={alpha:.3f}")

        # (c) Upsample sai số trở lại HR
        err_up = upsample(err_lr, scale)

        # (d) Back-projection: lật kernel để phản hồi sai số
        flipped_kernel = np.flipud(np.fliplr(kernel))
        backproj = blur(err_up, flipped_kernel)

        # (e) Alpha động (nếu bật)
        if adaptive_alpha:
            alpha = 0.5 + 0.5 * np.exp(-mse * 10)

        # (f) Cập nhật ảnh HR
        x = x + alpha * backproj

        # (g) Denoise nhẹ nếu bật
        if denoise:
            x = denoise_tv_chambolle(x, weight=0.05)

        # (h) Clip giá trị hợp lệ
        x = np.clip(x, 0, 1)

        # (i) Blur động (sigma giảm dần)
        if dynamic_blur:
            sigma = sigma * 0.95
            kernel = gaussian_kernel(size, sigma)

        # (j) Dừng sớm nếu hội tụ
        if early_stop and abs(prev_err - mse) < 1e-6:
            break
        prev_err = mse

    # --- 4️⃣ Trả kết quả ---
    if return_mse:
        return x, mse_list
    else:
        return x