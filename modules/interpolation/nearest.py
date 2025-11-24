import numpy as np
from PIL import Image

# ---------- Nearest Neighbor ----------
def inter_nearest(img, x, y):
    h, w, c = img.shape
    x0 = np.clip(np.round(x).astype(int), 0, w - 1)
    y0 = np.clip(np.round(y).astype(int), 0, h - 1)
    return img[y0, x0]