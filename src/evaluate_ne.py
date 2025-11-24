# evaluate_ne.py

import sys
import os
import cv2
import numpy as np
import time
import argparse
from glob import glob

# Thêm thư mục cha vào path để import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import utils
from src import ml_patch

def run_ne_inference(lr_img_path, knn_model, feat_vectors, hr_patches_db, scale, patch_size, overlap, k_neighbors, epsilon):
    """
    Hàm tái tạo ảnh NE cho một ảnh LR đầu vào.
    Trả về: Ảnh kênh Y (float 0-255)
    """
    # 1. Trích xuất patches và features từ ảnh LR
    _, target_LR_feats, target_lr_y_means = ml_patch.extract_patches_per_image(
        lr_img_path, scale=scale, patch_size=patch_size, overlap=overlap
    )
    
    predicted_patches = []
    
    # 2. Duyệt qua từng feature patch để tìm láng giềng
    # (Lưu ý: Có thể tối ưu bằng batch query nhưng loop dễ hiểu hơn)
    for i, query_feat in enumerate(target_LR_feats):
        # KNN Search
        distances, indices = ml_patch.query_knn(knn_model, feat_vectors, query_feat, k=k_neighbors)
        
        neighbor_vecs = feat_vectors[indices]
        neighbor_hr = hr_patches_db[indices]
        
        # Tính trọng số LLE
        w = ml_patch.compute_weights(query_feat, neighbor_vecs, epsilon=epsilon)
        
        # Tái tạo patch HR
        pred_patch = ml_patch.map_hr_patches(neighbor_hr, w, target_lr_y_means[i])
        predicted_patches.append(pred_patch)
        
    # 3. Ghép các patch lại thành ảnh (Aggregation)
    lr_img = cv2.imread(lr_img_path)
    h_lr, w_lr = lr_img.shape[:2]
    h_hr, w_hr = h_lr * scale, w_lr * scale
    
    predicted_y = np.zeros((h_hr, w_hr), dtype=np.float32)
    weight_mask = np.zeros((h_hr, w_hr), dtype=np.float32)
    
    step = int(patch_size * (1 - overlap))
    patch_idx = 0
    
    for i in range(0, h_lr - patch_size + 1, step):
        for j in range(0, w_lr - patch_size + 1, step):
            if patch_idx >= len(predicted_patches): break
            
            hr_i, hr_j = i * scale, j * scale
            p_h, p_w = predicted_patches[patch_idx].shape
            
            predicted_y[hr_i:hr_i+p_h, hr_j:hr_j+p_w] += predicted_patches[patch_idx]
            weight_mask[hr_i:hr_i+p_h, hr_j:hr_j+p_w] += 1.0
            patch_idx += 1
            
    # Normalize (chia trung bình vùng chồng lấn)
    weight_mask[weight_mask == 0] = 1.0
    predicted_y /= weight_mask
    
    return predicted_y * 255.0  # Trả về range 0-255

def main():
    # --- CẤU HÌNH ---
    parser = argparse.ArgumentParser(description="Evaluate Neighbor Embedding SR")
    parser.add_argument('--scale', type=int, default=4, help='Upscale factor (2, 3, 4)')
    parser.add_argument('--patch_size', type=int, default=5, help='Patch size')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors for KNN')
    
    args = parser.parse_args()
    
    SCALE = args.scale
    PATCH_SIZE = args.patch_size
    OVERLAP = 0.5
    K_NEIGHBORS = args.k
    EPSILON = 1e-6

    # Đường dẫn dữ liệu
    DATA_ROOT = '../data' # Sửa lại nếu cần
    TRAIN_FEATS_PATH = os.path.join(DATA_ROOT, f'YIQ_patches/LR_features_patches_x{SCALE}.npy')
    TRAIN_LABEL_PATH = os.path.join(DATA_ROOT, f'YIQ_patches/HR_centered_patches_x{SCALE}.npy')
    TEST_LR_FOLDER = os.path.join(DATA_ROOT, f'target_lr/LR_x{SCALE}')
    TEST_HR_FOLDER = os.path.join(DATA_ROOT, f'target_hr/HR_x{SCALE}')

    print(f"=== BẮT ĐẦU ĐÁNH GIÁ NE (Scale x{SCALE}) ===")
    
    # 1. Load dữ liệu Training
    if not os.path.exists(TRAIN_FEATS_PATH):
        print(f"Error: Không tìm thấy file {TRAIN_FEATS_PATH}")
        return

    print("-> Loading training patches...")
    LR_features_patches = np.load(TRAIN_FEATS_PATH)
    HR_centered_patches = np.load(TRAIN_LABEL_PATH)
    
    # 2. Build KNN
    print(f"-> Building KNN Index ({LR_features_patches.shape[0]} patches)...")
    t0 = time.time()
    knn_model, feat_vectors = ml_patch.build_knn_idx(LR_features_patches, k=K_NEIGHBORS)
    print(f"-> KNN Built in {time.time() - t0:.2f}s")

    # 3. Chuẩn bị Test Set
    test_images = sorted(glob(os.path.join(TEST_LR_FOLDER, '*.png')))
    gt_images = sorted(glob(os.path.join(TEST_HR_FOLDER, '*.png')))
    
    if not test_images:
        print("Error: Không tìm thấy ảnh test trong folder Set5.")
        return

    # 4. Vòng lặp đánh giá
    print("\n" + "="*85)
    print(f"{'Image Name':<20} | {'Bicubic (PSNR/SSIM)':<25} | {'NE (PSNR/SSIM)':<25} | {'Gain':<8}")
    print("-" * 85)

    results = [] # [psnr_bic, ssim_bic, psnr_ne, ssim_ne]

    for lr_path, hr_path in zip(test_images, gt_images):
        img_name = os.path.basename(lr_path)
        
        # -- A. Ground Truth (HR) --
        hr_img = cv2.imread(hr_path)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        hr_yiq = ml_patch.rgb2yiq(hr_img)
        hr_y = hr_yiq[:, :, 0] * 255.0
        
        # -- B. Bicubic Baseline --
        lr_img = cv2.imread(lr_path)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = lr_img.shape[:2]
        bic_hr = cv2.resize(lr_img, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)
        bic_yiq = ml_patch.rgb2yiq(bic_hr)
        bic_y = bic_yiq[:, :, 0] * 255.0
        
        # -- C. NE Reconstruction --
        ne_y = run_ne_inference(lr_path, knn_model, feat_vectors, HR_centered_patches, 
                                SCALE, PATCH_SIZE, OVERLAP, K_NEIGHBORS, EPSILON)
        
        # Clip giá trị hợp lệ (0-255)
        ne_y = np.clip(ne_y, 0, 255)
        bic_y = np.clip(bic_y, 0, 255)
        
        # -- D. Tính Metrics (PSNR/SSIM trên kênh Y, có cắt viền) --
        border_cut = SCALE*PATCH_SIZE // 2
        p_bic = utils.calculate_psnr(bic_y, hr_y, crop_border=border_cut)
        s_bic = utils.calculate_ssim(bic_y, hr_y, crop_border=border_cut)
        
        p_ne = utils.calculate_psnr(ne_y, hr_y, crop_border=border_cut)
        s_ne = utils.calculate_ssim(ne_y, hr_y, crop_border=border_cut)
        
        results.append([p_bic, s_bic, p_ne, s_ne])
        
        print(f"{img_name:<20} | {p_bic:.2f} dB / {s_bic:.4f}        | {p_ne:.2f} dB / {s_ne:.4f}        | {p_ne - p_bic:+.2f}")

    # 5. Tính trung bình
    avg = np.mean(results, axis=0)
    print("-" * 85)
    print(f"{'AVERAGE':<20} | {avg[0]:.2f} dB / {avg[1]:.4f}        | {avg[2]:.2f} dB / {avg[3]:.4f}        | {avg[2] - avg[0]:+.2f}")
    print("=" * 85)

if __name__ == "__main__":
    main()