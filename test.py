import cv2
import numpy as np
import os
from src.metrics import compute_basic_metrics
from skimage.restoration import unsupervised_wiener as sk_unsupervised_wiener
from scipy.ndimage import gaussian_filter
from src.Wiener_Filter import gaussian_psf, wiener_base_real, unsupervised_wiener_improved1
from src.models import iterative_backprojection
from modules.upsample import upsample_bicubic

def load_image(image_path):
    """
    Load image từ đường dẫn
    Returns: numpy array in [0,1] float
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image from {image_path}")
    if img.shape[-1] == 4:
        # Nếu ảnh có alpha channel, bỏ qua alpha
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
    
    # Convert from [0,1] float to [0,255] uint8
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    if img_uint8.ndim == 3:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(save_path, img_uint8)
    print(f"   Saved: {save_path}")

def test_image_restoration(degraded_path, hr_path):
    """
    Test pipeline: load degraded image and corresponding HR image -> restore -> compute metrics
    """
    print(f"Testing image restoration pipeline for degraded image: {degraded_path}")
    
    # Get image name for saving
    image_name = os.path.splitext(os.path.basename(degraded_path))[0]
    
    # 1. Load degraded and HR images
    print("1. Loading degraded and HR images...")
    degraded_img = load_image(degraded_path)
    original_img = load_image(hr_path)
    print(f"   Degraded image shape: {degraded_img.shape}")
    print(f"   HR image shape: {original_img.shape}")
    
    # 2. Restore image using Wiener Filter
    print("2. Restoring image...")
    restored_img = degraded_img.copy()
    restored_img = upsample_bicubic(restored_img, scale=4)
    restored_img = wiener_base_real(restored_img)

    print(f"   Restored image shape: {restored_img.shape}")
    
    # Save predicted super-resolution image
    print("   Saving predicted SR image...")
    save_image(restored_img, f"data/predicted_sr/{image_name}_predicted_sr.png")
    
    # 3. Compute metrics
    print("3. Computing metrics...")
    metrics = compute_basic_metrics(restored_img, original_img, multichannel=(original_img.ndim == 3))
    
    print("\n=== METRICS RESULTS ===")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    return {
        'original': original_img,
        'degraded': degraded_img,
        'restored': restored_img,
        'metrics': metrics
    }

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data/predicted_sr", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Đường dẫn đến ảnh degraded và HR
    degraded_path = "data/degraded_lr/img_001_SRF_4_LR.png"
    hr_path = "data/input_hr/img_001_SRF_4_HR.png"
    
    try:
        results = test_image_restoration(degraded_path, hr_path)
        print("\nTest completed successfully!")
        
        # Save comparison results
        print("\n4. Saving comparison results...")
        image_name = os.path.splitext(os.path.basename(degraded_path))[0]
        
        # Save original HR image for comparison
        save_image(results['original'], f"results/{image_name}_original_hr.png")
        
        # Save degraded LR image (also to results for easy comparison)
        save_image(results['degraded'], f"results/{image_name}_degraded_lr.png")
        
        # Save restored SR image (also to results for easy comparison)
        save_image(results['restored'], f"results/{image_name}_restored_sr.png")
        
        print("\nFiles saved:")
        print(f"├── data/predicted_sr/{image_name}_predicted_sr.png")
        print(f"├── results/{image_name}_original_hr.png")
        print(f"├── results/{image_name}_degraded_lr.png")
        print(f"└── results/{image_name}_restored_sr.png")
            
    except Exception as e:
        print(f"Error during testing: {e}")