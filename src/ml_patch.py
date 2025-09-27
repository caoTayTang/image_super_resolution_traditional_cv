from src import utils
import cv2
import numpy as np
import os
from glob import glob

def extract_patches(lr_folder, hr_folder, patch_size=5, scale=2, overlap=0.5):
    """
    Chia ảnh LR và HR thành patch
    Args:
        lr_folder: folder chứa LR images
        hr_folder: folder chứa HR images
        patch_size: kích thước patch LR (n x n)
        scale: scale factor giữa HR và LR
        overlap: tỉ lệ chồng lấn (0~1)
    Returns:
        LR_patches: list các patch LR
        HR_patches: list các patch HR
    """
    LR_patches = []
    HR_patches = []

    step = int(patch_size * (1 - overlap))  # step size sliding window

    lr_images = sorted(glob(os.path.join(lr_folder, '*.png')))
    hr_images = sorted(glob(os.path.join(hr_folder, '*.png')))

    assert len(lr_images) == len(hr_images), "Số ảnh LR và HR phải bằng nhau"

    for lr_path, hr_path in zip(lr_images, hr_images):
        lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
        hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)

        h_lr, w_lr = lr_img.shape[:2]
        h_hr, w_hr = hr_img.shape[:2]

        # Kiểm tra HR size có đúng scale
        assert h_hr == h_lr * scale and w_hr == w_lr * scale, \
            f"HR image size must be scale * LR image size: {hr_path}"

        for i in range(0, h_lr - patch_size + 1, step):
            for j in range(0, w_lr - patch_size + 1, step):
                lr_patch = lr_img[i:i+patch_size, j:j+patch_size, :]
                # HR patch tương ứng
                hr_i, hr_j = i*scale, j*scale
                hr_patch = hr_img[hr_i:hr_i+patch_size*scale, hr_j:hr_j+patch_size*scale, :]

                LR_patches.append(lr_patch)
                HR_patches.append(hr_patch)

    print(f"Extracted {len(LR_patches)} LR patches and {len(HR_patches)} HR patches")
    return LR_patches, HR_patches

def process_all_patches(base_dir, output_dir, patch_size=5, overlap=0.5):
    """
    Duyệt tất cả scale factor và lưu patch
    base_dir: list: base_dir[0] là dir tới degraded_lr, base_dir[1] là dir tới input_hr
    """
    os.makedirs(output_dir, exist_ok=True)
    scales = [2, 3, 4]

    for s in scales:
        lr_folder = os.path.join(base_dir[0], f'LR_x{s}')
        hr_folder = os.path.join(base_dir[1], f'HR_x{s}')
        LR_patches, HR_patches = extract_patches(lr_folder, hr_folder,
                                                  patch_size=patch_size, scale=s,
                                                  overlap=overlap)
        # Lưu file numpy
        np.save(os.path.join(output_dir, f'LR_patches_x{s}.npy'), LR_patches)
        np.save(os.path.join(output_dir, f'HR_patches_x{s}.npy'), HR_patches)
        print(f"Saved LR/HR patches for scale x{s} to {output_dir}")