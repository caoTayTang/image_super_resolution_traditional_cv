import cv2
import numpy as np
import os
from src.metrics import compute_basic_metrics
from src.Wiener_Filter import wiener_unsupervised_mcmc
from modules.upsample import upsample_bicubic

def load_image(image_path):
    """
    Load image từ đường dẫn, convert sang RGB float32 [0,1]
    """
    # Load ảnh màu
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image from {image_path}")
    
    # Nếu ảnh có alpha channel, bỏ qua
    if img.shape[-1] == 4:
        img = img[..., :3]
    
    # Convert BGR to RGB
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to float [0,1]
    img = img.astype(np.float32) / 255.0
    
    return img

def save_image(img, save_path):
    """
    Save image từ numpy array [0,1] float sang file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Clip và convert sang uint8
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV saving
    if img_uint8.ndim == 3:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(save_path, img_uint8)
    print(f"   Saved: {save_path}")

# thêm hàm lưu tấm ảnh so sánh ngang (bicubic | wiener | hr)
def save_horizontal_comparison(img_list, save_path):
    """
    img_list: list of images in float [0,1], each grayscale or RGB.
    Lưu ảnh ghép ngang, resize các ảnh thứ 2..n về kích thước ảnh đầu tiên.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def to_bgr_uint8(im, target_shape=None):
        im = np.clip(im, 0, 1)
        if im.ndim == 2:  # grayscale -> BGR
            im_uint = (im * 255).astype(np.uint8)
            im_bgr = cv2.cvtColor(im_uint, cv2.COLOR_GRAY2BGR)
        else:  # RGB -> BGR
            im_uint = (im * 255).astype(np.uint8)
            im_bgr = cv2.cvtColor(im_uint, cv2.COLOR_RGB2BGR)
        if target_shape is not None:
            im_bgr = cv2.resize(im_bgr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
        return im_bgr

    target_shape = img_list[0].shape[:2]
    imgs_bgr = [to_bgr_uint8(im, target_shape) for im in img_list]
    out = np.concatenate(imgs_bgr, axis=1)
    cv2.imwrite(save_path, out)
    print(f"   Saved comparison: {save_path}")

# NEW: save 2x2 grid (row-major: [0,1] first row, [2,3] second row)
def save_2x2_grid(imgs, save_path):
    """
    imgs: list/tuple of 4 images (float in [0,1]), each grayscale or RGB.
    Arranges them in 2 rows x 2 cols and saves as BGR uint8.
    Order: [top-left, top-right, bottom-left, bottom-right]
    """
    assert len(imgs) == 4, "Expected 4 images for 2x2 grid"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def to_bgr_uint8(im, target_shape=None):
        im = np.clip(im, 0, 1)
        if im.ndim == 2:
            im_uint = (im * 255).astype(np.uint8)
            im_bgr = cv2.cvtColor(im_uint, cv2.COLOR_GRAY2BGR)
        else:
            im_uint = (im * 255).astype(np.uint8)
            im_bgr = cv2.cvtColor(im_uint, cv2.COLOR_RGB2BGR)
        if target_shape is not None:
            im_bgr = cv2.resize(im_bgr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
        return im_bgr

    # Use first image shape as target
    target_shape = imgs[0].shape[:2]
    bgr_imgs = [to_bgr_uint8(im, target_shape) for im in imgs]

    top = np.concatenate([bgr_imgs[0], bgr_imgs[1]], axis=1)
    bottom = np.concatenate([bgr_imgs[2], bgr_imgs[3]], axis=1)
    out = np.concatenate([top, bottom], axis=0)

    cv2.imwrite(save_path, out)
    print(f"   Saved 2x2 grid: {save_path}")

def test_image_restoration(degraded_path, hr_path):
    """
    Test pipeline: 
    1. Load RGB -> Convert YCrCb
    2. Upsample all channels
    3. Compare:
       - Case A: Bicubic Upsample Only
       - Case B: Bicubic + Wiener Filter on Y channel
    """
    print(f"\nTesting pipeline for: {os.path.basename(degraded_path)}")
    image_name = os.path.splitext(os.path.basename(degraded_path))[0]
    
    # 1. Load Images
    print("1. Loading images...")
    lr_rgb = load_image(degraded_path)
    hr_rgb = load_image(hr_path)
    
    print(f"   LR shape: {lr_rgb.shape}")
    print(f"   HR shape: {hr_rgb.shape}")

    # 2. Convert LR to YCrCb
    print("2. Converting to YCrCb and Upsampling...")
    if lr_rgb.ndim == 3:
        lr_ycrcb = cv2.cvtColor(lr_rgb, cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(lr_ycrcb)
    else:
        Y = lr_rgb
        Cr = None
        Cb = None

    # 3. Upsample all channels (Scale = 2)
    scale = 2
    if lr_rgb.ndim == 3:
        Y_up = upsample_bicubic(Y, scale=scale).astype(np.float32)
        Cr_up = upsample_bicubic(Cr, scale=scale).astype(np.float32)
        Cb_up = upsample_bicubic(Cb, scale=scale).astype(np.float32)
    else:
        Y_up = upsample_bicubic(lr_rgb, scale=scale)

    # ==========================================
    # CASE A: UPSAMPLE ONLY (BICUBIC BASELINE)
    # ==========================================
    print("3. Processing Case A: Bicubic Only...")
    if lr_rgb.ndim == 3:
        # Merge channels
        bicubic_ycrcb = cv2.merge([Y_up, Cr_up, Cb_up])
        # Convert back to RGB
        img_bicubic = cv2.cvtColor(bicubic_ycrcb, cv2.COLOR_YCrCb2RGB)
        img_bicubic = np.clip(img_bicubic, 0, 1)
    else:
        img_bicubic = np.clip(Y_up, 0, 1)
    
    # Save Bicubic result
    save_image(img_bicubic, f"data/predicted_sr/{image_name}_bicubic_only.png")

    # ==========================================
    # CASE B: UPSAMPLE + WIENER (PROPOSED)
    # ==========================================
    print("4. Processing Case B: Wiener Filter on Y channel...")
    
    # Apply Wiener Filter ONLY on Y channel
    wiener_result = wiener_unsupervised_mcmc(Y_up)
    
    if isinstance(wiener_result, tuple):
        Y_wiener = wiener_result[0] # Lấy ảnh kết quả
    else:
        Y_wiener = wiener_result

    # Ensure float32
    if lr_rgb.ndim == 3:
        Y_wiener = Y_wiener.astype(np.float32)
    
    # Merge Wiener Y with original upsampled Cr, Cb
    if lr_rgb.ndim == 3:
        wiener_ycrcb = cv2.merge([Y_wiener, Cr_up, Cb_up])
        # Convert back to RGB
        img_wiener = cv2.cvtColor(wiener_ycrcb, cv2.COLOR_YCrCb2RGB)
        img_wiener = np.clip(img_wiener, 0, 1)
    else:
        img_wiener = np.clip(Y_wiener, 0, 1)

    # Save Wiener result
    save_image(img_wiener, f"data/predicted_sr/{image_name}_wiener_restored.png")

    # Lưu tấm so sánh ngang: Bicubic | Wiener | HR
    os.makedirs("results", exist_ok=True)
    comp_path = f"results/{image_name}_comparison_row.png"
    save_horizontal_comparison([img_bicubic, img_wiener, hr_rgb], comp_path)

    # NEW: Save 2x2 grid: LR (upsampled for display) | Bicubic | Wiener | HR
    # Upsample LR for visualization to match predicted size
    if lr_rgb.ndim == 3:
        lr_up_display = upsample_bicubic(lr_rgb, scale=scale)
    else:
        lr_up_display = upsample_bicubic(lr_rgb, scale=scale)
    grid_path = f"results/{image_name}_comparison_2x2.png"
    save_2x2_grid([lr_up_display, img_bicubic, img_wiener, hr_rgb], grid_path)

    # ==========================================
    # METRICS COMPARISON
    # ==========================================
    print("\n=== COMPARING METRICS ===")
    
    # Metrics for Bicubic
    print("--- Bicubic Only ---")
    metrics_bicubic = compute_basic_metrics(img_bicubic, hr_rgb, multichannel=True)
    for k, v in metrics_bicubic.items():
        print(f"{k}: {v:.4f}")

    # Metrics for Wiener
    print("--- Wiener Restoration ---")
    metrics_wiener = compute_basic_metrics(img_wiener, hr_rgb, multichannel=True)
    for k, v in metrics_wiener.items():
        print(f"{k}: {v:.4f}")

    return {
        'bicubic': metrics_bicubic,
        'wiener': metrics_wiener
    }

if __name__ == "__main__":
    # Create directories
    os.makedirs("data/predicted_sr", exist_ok=True)
    
    # Paths (Adjust as needed)
    degraded_path = "data/degraded_lr/lena512_degraded_lr_2_noisy.png"
    hr_path = "data/input_hr/lena512.jpg"
    
    try:
        if os.path.exists(degraded_path) and os.path.exists(hr_path):
            test_image_restoration(degraded_path, hr_path)
            print("\nTest completed successfully!")
        else:
            print(f"Error: Input files not found.")
            print(f"Check: {degraded_path}")
            print(f"Check: {hr_path}")
            
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()