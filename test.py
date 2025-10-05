import cv2
import numpy as np
import os
from src.metrics import compute_basic_metrics
from modules.degradation import degrade_image
from skimage import restoration
from src.models import iterative_backprojection_tv
from src.Wiener_Filter import unsupervised_wiener_custom, unsupervised_wiener_improved, joint_iterative_backprojection_wiener

def load_image(image_path):
    """
    Load image từ đường dẫn
    Returns: numpy array in [0,1] float
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image from {image_path}")
    if img.shape[-1] == 4:
        # Nếu ảnh có alpha channel, bỏ qua alpha
        img = img[..., :3]
    
    # Convert BGR to RGB
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

def test_image_restoration(image_path):
    """
    Test pipeline: load -> degrade -> restore -> compute metrics
    """
    print(f"Testing image restoration pipeline for: {image_path}")
    
    # Get image name for saving
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. Load original image
    print("1. Loading original image...")
    original_img = load_image(image_path)
    print(f"   Image shape: {original_img.shape}")
    
    # 2. Degrade image (blur + downscale)
    print("2. Degrading image...")
    degraded_img = degrade_image(original_img, scale=4)
    print(f"   Degraded image shape: {degraded_img.shape}")
    
    # Save degraded low-resolution image
    print("   Saving degraded LR image...")
    save_image(degraded_img, f"data/degraded_lr/{image_name}_degraded_lr.png")
    
    # 3. Restore image using Wiener Filter
    print("3. Restoring image...")
    # restored_img = np.zeros_like(original_img)
    # for c in range(3):
    #     # restored_img[..., c], _ = restoration.unsupervised_wiener(degraded_img[..., c])
    #     restored_img[..., c], _ = unsupervised_wiener_custom(degraded_img[..., c], psf_init=None, iterations=200, balance=0.1)
    # restored_img, _ = unsupervised_wiener_improved(degraded_img)
    # restored_img = iterative_backprojection_tv(degraded_img, scale=1)
    restored_img, _ = joint_iterative_backprojection_wiener(degraded_img, scale=4, num_iters=20, psf_update_freq=2)
    print(f"   Restored image shape: {restored_img.shape}")
    
    # Save predicted super-resolution image
    print("   Saving predicted SR image...")
    save_image(restored_img, f"data/predicted_sr/{image_name}_predicted_sr.png")
    
    # 4. Compute metrics
    print("4. Computing metrics...")
    metrics = compute_basic_metrics(restored_img, original_img, multichannel=(original_img.ndim==3))
    
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
    os.makedirs("data/degraded_lr", exist_ok=True)
    os.makedirs("data/predicted_sr", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Test với một ảnh mẫu
    # Thay đổi đường dẫn này theo ảnh của bạn
    init_psf = np.ones((5, 5)) / 25
    image_path = "data/input_hr/cameraman.png"
    
    try:
        results = test_image_restoration(image_path)
        print("\nTest completed successfully!")
        
        # Save comparison results
        print("\n5. Saving comparison results...")
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save original HR image for comparison
        save_image(results['original'], f"results/{image_name}_original_hr.png")
        
        # Save degraded LR image (also to results for easy comparison)
        save_image(results['degraded'], f"results/{image_name}_degraded_lr.png")
        
        # Save restored SR image (also to results for easy comparison)
        save_image(results['restored'], f"results/{image_name}_restored_sr.png")
        
        print("\nFiles saved:")
        print(f"├── data/degraded_lr/{image_name}_degraded_lr.png")
        print(f"├── data/predicted_sr/{image_name}_predicted_sr.png")
        print(f"├── results/{image_name}_original_hr.png")
        print(f"├── results/{image_name}_degraded_lr.png")
        print(f"└── results/{image_name}_restored_sr.png")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure:")
        print("1. Image path is correct")
        print("2. modules/degradation.py has degrade_image function")
        print("3. src/Wiener_Filter.py has restore_image function")