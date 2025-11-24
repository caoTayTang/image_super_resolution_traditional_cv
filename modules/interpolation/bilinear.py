import numpy as np
from PIL import Image

# ----------  Bilinear ----------
def inter_linear(img, x, y):
    h, w, c = img.shape
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)

    dx = x - x0
    dy = y - y0

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (1 - dx) * (1 - dy)
    wb = (1 - dx) * dy
    wc = dx * (1 - dy)
    wd = dx * dy

    return (Ia * wa[..., None] + Ib * wb[..., None] +
            Ic * wc[..., None] + Id * wd[..., None])