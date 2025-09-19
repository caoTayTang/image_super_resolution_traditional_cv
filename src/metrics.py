import numpy as np

def psnr(img1, img2):
    """
    Tính Peak Signal-to-Noise Ratio giữa 2 ảnh.
    img1, img2: numpy array, giá trị trong [0,1]
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # vì ảnh đã chuẩn hóa [0,1]
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value