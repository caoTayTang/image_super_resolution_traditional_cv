import cv2
import numpy as np

def upsample_nearest(img, scale):
    """Nearest Neighbor Upsampling (cv2)."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def upsample_bilinear(img, scale):
    """Bilinear Upsampling (cv2)."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)


def upsample_bicubic(img, scale):
    """Bicubic Upsampling (cv2)."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
