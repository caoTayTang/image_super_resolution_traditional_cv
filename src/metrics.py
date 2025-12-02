import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

def ensure_float_0_1(img):
    """
    Chuyển đổi ảnh sang float32 và chuẩn hóa về [0, 1]
    """
    img = np.array(img).astype(np.float32)
    
    # Nếu ảnh đang là 0-255 (hoặc lớn hơn 1), chia cho 255
    if img.max() > 1.0:
        img = img / 255.0
        
    # Clip chặt lại để đảm bảo không có số nào < 0 hoặc > 1 do lỗi làm tròn
    img = np.clip(img, 0.0, 1.0)
    return img

def compute_basic_metrics(img1, img2, multichannel=True):
    """
    Tính các chỉ số đánh giá chất lượng ảnh (PSNR, SSIM, MSE)
    Args:
        img1, img2: Ảnh đầu vào (sẽ tự động được convert về float [0,1])
        multichannel: True nếu là ảnh màu
    """
    # 1. Chuẩn hóa cả 2 ảnh về cùng kiểu float [0,1]
    img1 = ensure_float_0_1(img1)
    img2 = ensure_float_0_1(img2)
    
    # Ensure images have same shape
    if img1.shape != img2.shape:
        # Thử resize img1 về img2 nếu cần thiết (tùy chọn), hoặc báo lỗi
        raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")
    
    metrics = {}
    metrics['MSE'] = mse(img1, img2)
    
    # PSNR (Giờ đây data_range luôn là 1.0 vì đã chuẩn hóa)
    metrics['PSNR'] = psnr(img1, img2, data_range=1.0)
    
    # SSIM logic (giữ nguyên logic cửa sổ tốt của bạn)
    min_dim = min(img1.shape[0], img1.shape[1])
    
    if min_dim >= 11: win_size = 11
    elif min_dim >= 7: win_size = 7
    elif min_dim >= 5: win_size = 5
    else: win_size = 3
    
    if win_size % 2 == 0: win_size -= 1
    
    try:
        # Kiểm tra xem ảnh có kênh màu hay không (ndim=3)
        channel_axis = 2 if (img1.ndim == 3 and multichannel) else None
        
        metrics['SSIM'] = ssim(
            img1, img2, 
            win_size=win_size,
            channel_axis=channel_axis,
            data_range=1.0
        )
    except Exception as e:
        print(f"Warning: SSIM calculation failed: {e}")
        metrics['SSIM'] = 0.0
    
    return metrics