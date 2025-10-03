import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

def compute_basic_metrics(img1, img2, multichannel=True):
    """
    Compute basic image quality metrics
    Args:
        img1, img2: Images in [0,1] float format
        multichannel: True if images have color channels
    """
    # Ensure images have same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")
    
    # Ensure images are in valid range
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    metrics = {}
    
    # MSE
    metrics['MSE'] = mse(img1, img2)
    
    # PSNR
    metrics['PSNR'] = psnr(img1, img2, data_range=1.0)
    
    # SSIM with proper window size
    min_dim = min(img1.shape[0], img1.shape[1])
    
    # Choose appropriate window size
    if min_dim >= 11:
        win_size = 11
    elif min_dim >= 7:
        win_size = 7
    elif min_dim >= 5:
        win_size = 5
    else:
        win_size = 3  # Minimum possible
    
    # Make sure win_size is odd
    if win_size % 2 == 0:
        win_size -= 1
    
    try:
        if multichannel and img1.ndim == 3:
            metrics['SSIM'] = ssim(
                img1, img2, 
                win_size=win_size,
                channel_axis=2,  # Color channel is the last axis
                data_range=1.0
            )
        else:
            metrics['SSIM'] = ssim(
                img1, img2,
                win_size=win_size,
                data_range=1.0
            )
    except Exception as e:
        print(f"Warning: SSIM calculation failed: {e}")
        metrics['SSIM'] = 0.0
    
    return metrics

def compute_lpips(img1, img2):
    """
    Compute LPIPS metric (requires lpips library)
    """
    try:
        import lpips
        import torch
        
        # Initialize LPIPS model
        loss_fn = lpips.LPIPS(net='alex')
        
        # Convert to tensor and proper format
        # LPIPS expects tensors in [-1, 1] range with shape (N, C, H, W)
        def to_tensor(img):
            if img.ndim == 3:
                img = img.transpose(2, 0, 1)  # HWC to CHW
            img = torch.from_numpy(img).float()
            img = img.unsqueeze(0)  # Add batch dimension
            img = img * 2.0 - 1.0  # [0,1] to [-1,1]
            return img
        
        tensor1 = to_tensor(img1)
        tensor2 = to_tensor(img2)
        
        # Compute LPIPS
        with torch.no_grad():
            distance = loss_fn(tensor1, tensor2)
        
        return distance.item()
        
    except ImportError:
        raise ImportError("LPIPS requires: pip install lpips")
    except Exception as e:
        raise RuntimeError(f"LPIPS computation failed: {e}")