import numpy as np
from PIL import Image
from .bilinear import inter_linear
from .nearest import inter_nearest
from .bicubic import inter_cubic
from .constant import *

def _prepare_grid(img, scale):
    if isinstance(img, Image.Image):
        img = np.array(img).astype(np.float32)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    h, w, c = img.shape
    new_h, new_w = int(h * scale), int(w * scale)
    y, x = np.meshgrid(np.arange(new_h) / scale, np.arange(new_w) / scale, indexing='ij')
    return img, h, w, c, new_h, new_w, x, y

def resize(img, scale=2, interpolation=INTER_LINEAR):
    img, h, w, c, new_h, new_w, x, y = _prepare_grid(img, scale)
    if interpolation == INTER_LINEAR:
        out = inter_linear(img, x, y)
    elif interpolation == INTER_NEAREST:
        out = inter_nearest(img, x, y)
    elif interpolation == INTER_CUBIC:
        out = inter_cubic(img, x, y)
    else:
        raise ValueError(f"Unknown interpolation type: {interpolation}")
    return np.uint8(out)