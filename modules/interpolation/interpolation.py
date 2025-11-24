import numpy as np
from PIL import Image

from .resize import resize
from .constant import *

def sr_interpolation(img: Image.Image, method='nearest', scale=2) -> Image.Image:
    if method == 'nearest':
        inter = INTER_NEAREST
    elif method == 'bilinear':
        inter = INTER_LINEAR
    elif method == 'bicubic':
        inter = INTER_CUBIC
    else:
        raise ValueError(f"Invalid method for interpolation {method}")
    
    img = np.array(img).astype(np.float32)
    out = resize(img, scale, interpolation=inter)
    if out.ndim == 3 and out.shape[2] == 1:
        out = np.squeeze(out, axis=2)
        mode = "L"
    elif out.ndim == 2:
        mode = "L"
    elif out.ndim == 3 and out.shape[2] == 3:
        mode = "RGB"
    else:
        raise ValueError(f"[sr_interpolation] Unexpected shape after resize: {out.shape}")

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode=mode)