import numpy as np
from PIL import Image

def pil_to_np(img: Image.Image, gray=True):
    """PIL → np.float32 [0,1]."""
    if gray:
        return np.array(img.convert("L"), dtype=np.float32) / 255.0
    else:
        return np.array(img, dtype=np.float32) / 255.0

def np_to_pil(arr: np.ndarray):
    """Convert np.float32 [0,1] or uint8 → PIL Image safely."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

    # Handle grayscale or single-channel
    if arr.ndim == 2:
        return Image.fromarray(arr)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr.squeeze(axis=2)  # (H, W)
            return Image.fromarray(arr, mode="L")
        elif arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")
        else:
            raise ValueError(f"Unsupported number of channels: {arr.shape[2]}")
    else:
        raise ValueError(f"Invalid array shape: {arr.shape}")
