def iterative_backprojection_tv(lr, scale, kernel, iterations=20, alpha=1.0, tv_weight=0.05):
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

    # Khởi tạo bằng Bicubic upsampling
    x = upsample_bicubic(lr, scale, output_shape=(target_h, target_w))

    for it in range(iterations):
        # Sinh LR giả lập
        sim_lr = simulate_lr_from_hr(x, scale, kernel)
        # Sai số ở LR
        err_lr = lr - sim_lr
        # Upsample sai số về HR
        err_up = upsample_bicubic(err_lr, scale, output_shape=(target_h, target_w))
        # Back-projection
        flipped_kernel = np.flipud(np.fliplr(kernel))
        backproj = signal.convolve2d(err_up, flipped_kernel, mode='same', boundary='symm')
        # Cập nhật ảnh
        x = x + alpha * backproj
        x = np.clip(x, 0, 1)
        # TV denoise
        x = denoise_tv_chambolle(x, weight=tv_weight, channel_axis=None)

    return x