import os
import cv2
import shutil
from glob import glob
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim

def load_hr_image(path, scale=4):
    # 1. Đọc ảnh màu (Mặc định OpenCV là BGR)
    hr = cv2.imread(path)
    
    # 2. Chuyển từ BGR sang RGB để hiển thị đúng màu
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
    
    # 3. Chuẩn hóa về float [0, 1]
    hr = hr.astype("float32") / 255.0
    
    # 4. Cắt ảnh cho chẵn với scale (Cropping)
    H, W = hr.shape[:2] # Lấy H, W (bỏ qua kênh màu nếu có)
    Hc, Wc = H - (H % scale), W - (W % scale)
    
    # Trả về ảnh đã crop
    # Slice [:Hc, :Wc] sẽ tự động lấy hết các kênh màu ở chiều thứ 3
    return hr[:Hc, :Wc]

def img_classify(base_dir, output_dir):
    for dir in output_dir:
        os.makedirs(dir, exist_ok=True)
    srfs = [2,3,4]

    for srf in srfs:
        srf_folder = os.path.join(base_dir, f"image_SRF_{srf}")
        lr_out = os.path.join(output_dir[0], f"LR_x{srf}")
        hr_out = os.path.join(output_dir[1], f'HR_x{srf}')
        os.makedirs(lr_out, exist_ok=True)
        os.makedirs(hr_out, exist_ok=True)

        images = glob(os.path.join(srf_folder, '*.png'))
        for img_path in images:
            file_name = os.path.basename(img_path)
            if '_LR' in file_name:
                shutil.copy(img_path, os.path.join(lr_out, file_name))
            elif '_HR' in file_name:
                shutil.copy(img_path, os.path.join(hr_out, file_name))
            else:
                print(f'Warning: unknown type {file_name}')
        print(f'SRF_{srf} done: {len(images)} images processed.')
        
def calculate_psnr(img1, img2, crop_border=0):
    """Tính PSNR trên kênh Y, có cắt viền."""
    # Đảm bảo kiểu dữ liệu float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Nếu ảnh đang ở range 0-1 thì đưa về 0-255
    if img1.max() <= 1.0: img1 = img1 * 255.0
    if img2.max() <= 1.0: img2 = img2 * 255.0

    # Cắt viền (Shave)
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1, img2, crop_border=0):
    """Tính SSIM trên kênh Y, có cắt viền."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if img1.max() <= 1.0: img1 = img1 * 255.0
    if img2.max() <= 1.0: img2 = img2 * 255.0

    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]

    return ssim(img1, img2, data_range=255.0)