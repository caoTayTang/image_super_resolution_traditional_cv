import numpy as np
from scipy import ndimage, signal

# def upsample_bilinear(lr, scale):
#     h_lr, w_lr = lr.shape
#     h_hr, w_hr = h_lr * scale, w_lr*scale
    
#     # Tạo lưới cho ảnh HR
#     x_hr = np.arrange(w_hr) / scale
#     y_hr = np.arrange(h_hr) / scale
#     x0 = np.floor(x_hr).astype(int)
#     y0 = np.floor(y_hr).astype(int)
#     x1 = np.minimum(x0 + 1, w_lr - 1)
#     y1 = np.minimum(y0 + 1, h_lr - 1)
    
#     # Tính trọng số
#     wx = x_hr - x0
#     wy = y_hr - y0
    
#     #Tạo ảnh HR
#     hr = np.zeros((h_hr, w_hr))
    
#     for i in range(h_hr):
#         for j in range(w_hr):
#             top = (1 - wx[j]) * lr[y0[i], x0[j]] + wx[j] * lr[y0[i], x1[j]]
#             bottom = (1 - wx[j]) * lr[y1[i], x0[j]] + wx[j] * lr[y1[i], x1[j]]
#             hr[i,j] = (1 - wy[i]) * top + wy[i] * bottom

#     return hr

# def upsample_bicubic(lr, scale, output_shape=None):
#     """Upsample ảnh bằng bicubic interpolation."""
#     up = ndimage.zoom(lr, zoom=scale, order=3)
#     if output_shape is not None and up.shape != output_shape:
#         h, w = up.shape
#         th, tw = output_shape
#         # crop hoặc pad cho khớp kích thước
#         start_h = max(0, (h - th)//2)
#         start_w = max(0, (w - tw)//2)
#         up = up[start_h:start_h+th, start_w:start_w+tw]
#     return up

# def upsample_zero_insert(err_lr, scale):
#     h_lr, w_lr = err_lr.shape
#     h_hr, w_hr = h_lr*scale, w_lr*scale
#     err_up = np.zeros((h_hr, w_hr), dtype=err_lr.dtype)
#     err_up[::scale, ::scale] = err_lr  # zero-inserted
#     return err_up

# def cubic_kernel(x):
#     a = -0.5
#     absx = np.abs(x)
#     absx2 = absx**2
#     absx3 = absx**3
#     y = np.zeros_like(x)
#     mask1 = absx <= 1
#     y[mask1] = (a+2)*absx3[mask1] - (a+3)*absx2[mask1] + 1
#     mask2 = (absx > 1) & (absx < 2)
#     y[mask2] = a*absx3[mask2] - 5*a*absx2[mask2] + 8*a*absx[mask2] - 4*a
#     return y
    
    
def upsample_nearest(img, scale):
    """Nearest Neighbor Upsampling."""
    H, W = img.shape
    new_H, new_W = H * scale, W * scale
    out = np.zeros((new_H, new_W), dtype=img.dtype)

    for i in range(new_H):
        for j in range(new_W):
            # Tìm tọa độ pixel gần nhất trong ảnh gốc
            src_i = int(round(i / scale))
            src_j = int(round(j / scale))
            src_i = min(src_i, H - 1)
            src_j = min(src_j, W - 1)
            out[i, j] = img[src_i, src_j]

    return out

def upsample_bilinear(img, scale):
    """Bilinear Upsampling."""
    H, W = img.shape
    new_H, new_W = H * scale, W * scale
    out = np.zeros((new_H, new_W), dtype=float)

    for i in range(new_H):
        for j in range(new_W):
            x = i / scale
            y = j / scale

            x0 = int(np.floor(x))
            x1 = min(x0 + 1, H - 1)
            y0 = int(np.floor(y))
            y1 = min(y0 + 1, W - 1)

            dx = x - x0
            dy = y - y0

            # Nội suy bilinear
            val = (1 - dx) * (1 - dy) * img[x0, y0] + \
                  (dx)     * (1 - dy) * img[x1, y0] + \
                  (1 - dx) * (dy)     * img[x0, y1] + \
                  (dx)     * (dy)     * img[x1, y1]

            out[i, j] = val

    return out


def cubic_weight(t, a=-0.5):
    """Hàm cubic kernel (Catmull-Rom / bicubic)."""
    t = abs(t)
    if t <= 1:
        return (a + 2) * (t**3) - (a + 3) * (t**2) + 1
    elif t < 2:
        return a * (t**3) - 5*a*(t**2) + 8*a*t - 4*a
    return 0

def upsample_bicubic(img, scale):
    """Bicubic Upsampling."""
    H, W = img.shape
    new_H, new_W = H * scale, W * scale
    out = np.zeros((new_H, new_W), dtype=float)

    for i in range(new_H):
        for j in range(new_W):
            x = i / scale
            y = j / scale
            x_int = int(np.floor(x))
            y_int = int(np.floor(y))

            val = 0.0
            weight_sum = 0.0

            for m in range(-1, 3):
                for n in range(-1, 3):
                    xm = min(max(x_int + m, 0), H - 1)
                    yn = min(max(y_int + n, 0), W - 1)

                    wx = cubic_weight(x - (x_int + m))
                    wy = cubic_weight(y - (y_int + n))
                    w = wx * wy

                    val += img[xm, yn] * w
                    weight_sum += w

            out[i, j] = val / weight_sum if weight_sum != 0 else val

    return out