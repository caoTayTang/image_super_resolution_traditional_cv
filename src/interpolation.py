import cv2 
import numpy as np
from PIL import Image

def sr_interpolation(img: Image.Image, method='nearest', scale=2) -> Image.Image:
    """
    TODO: Implement Nearest/Bilinear/Bicubic/Edge-directed interpolation.
    """
    img = np.array(img)
    if method == 'nearest':
        inter = cv2.INTER_NEAREST
    elif method == 'bilinear':
        inter = cv2.INTER_LINEAR
    elif method == 'bicubic':
        inter = cv2.INTER_CUBIC
    elif method == 'lanczos':
        inter = cv2.INTER_LANCZOS4
    else:
        raise ValueError(f"Invalid method for interpolation {method}")
        # return Image._fromarray(img)
    
    h, w = img.shape[:2]
    out = cv2.resize(img, (w * scale, h * scale), interpolation=inter)
    return Image.fromarray(out)