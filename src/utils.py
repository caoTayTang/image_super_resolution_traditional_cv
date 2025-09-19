import cv2

def load_hr_image(path, scale=4):
    hr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    hr = hr.astype("float32") / 255.0
    H, W = hr.shape
    Hc, Wc = H - (H % scale), W - (W % scale)
    return hr[:Hc, :Wc]
