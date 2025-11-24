import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

def ensure_float_0_1(img):
    """
    Chuyển đổi ảnh sang float32 và chuẩn hóa về [0, 1] bất kể đầu vào là gì
    """
    img = np.array(img).astype(np.float32)
    
    # Nếu giá trị lớn nhất > 1.0, ta đoán nó là ảnh 0-255 -> Chia 255
    if img.max() > 1.0001: 
        img = img / 255.0
        
    # Clip chặt lại [0, 1]
    img = np.clip(img, 0.0, 1.0)
    return img

def compute_basic_metrics(img1, img2, multichannel=True):
    # --- DEBUG: In ra để kiểm tra xem có bị lệch thang đo không ---
    # print(f"Max img1: {img1.max()}, Max img2: {img2.max()}") 
    
    # 1. Chuẩn hóa cả 2 ảnh về cùng kiểu float [0,1]
    img1 = ensure_float_0_1(img1)
    img2 = ensure_float_0_1(img2)
    
    metrics = {}
    metrics['MSE'] = mse(img1, img2)
    
    # data_range=1.0 là cực kỳ quan trọng vì ta đã chuẩn hóa ở trên
    metrics['PSNR'] = psnr(img1, img2, data_range=1.0)
    
    # Tính SSIM
    min_dim = min(img1.shape[0], img1.shape[1])
    win_size = min(7, min_dim if min_dim % 2 != 0 else min_dim - 1)
    
    try:
        channel_axis = 2 if (img1.ndim == 3) else None
        metrics['SSIM'] = ssim(img1, img2, win_size=win_size, channel_axis=channel_axis, data_range=1.0)
    except:
        metrics['SSIM'] = 0.0
    
    return metrics