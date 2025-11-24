import cv2

def upsample_nearest(img, scale):
    """Nearest Neighbor Upsampling (Dùng OpenCV)."""
    # cv2.resize tự động xử lý cả ảnh xám và ảnh màu
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

def upsample_bilinear(img, scale):
    """Bilinear Upsampling (Dùng OpenCV)."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)

def upsample_bicubic(img, scale):
    """Bicubic Upsampling (Dùng OpenCV)."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)