import numpy as np
from PIL import Image

# ---------- Bicubic ----------
def cubic_weight(t):
    a = -0.5  # Catmull-Rom parameter
    abs_t = np.abs(t)
    abs_t2 = abs_t**2
    abs_t3 = abs_t**3
    w = np.zeros_like(t)

    mask1 = abs_t <= 1
    mask2 = (abs_t > 1) & (abs_t < 2)

    w[mask1] = (a + 2)*abs_t3[mask1] - (a + 3)*abs_t2[mask1] + 1
    w[mask2] = a*abs_t3[mask2] - 5*a*abs_t2[mask2] + 8*a*abs_t[mask2] - 4*a
    return w

def inter_cubic(img, x, y):
    h, w, c = img.shape
    out = np.zeros((x.shape[0], x.shape[1], c), dtype=np.float32)

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)

    for j in range(-1, 3):
        for i in range(-1, 3):
            xi = np.clip(x0 + i, 0, w - 1)
            yj = np.clip(y0 + j, 0, h - 1)

            wx = cubic_weight(x - xi)
            wy = cubic_weight(y - yj)
            wxy = wx * wy
            out += img[yj, xi] * wxy[..., None]

    return out