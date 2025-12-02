import numpy as np
from modules.degradation import *
from modules.upsample import upsample_bicubic

def iterative_backprojection(
    lr,
    upsample=upsample_bicubic,
    scale=4,
    iterations=20,
    alpha=1.0,
    size=9,
    sigma=1.6,
    kernel=None
):
    if kernel is None:
        kernel = gaussian_kernel(size, sigma)

    x = upsample(lr, scale)
    flipped_kernel = np.flipud(np.fliplr(kernel))

    for it in range(iterations):
        print(f"[IBP] Iteration {it+1}/{iterations}")

        sim_lr = simulate_lr_from_hr(x, scale, kernel)

        err_lr = lr - sim_lr
        err_up = upsample(err_lr, scale)

        flipped_kernel = np.flipud(np.fliplr(kernel))
        backproj = blur(err_up, flipped_kernel)

        x = x + alpha * backproj
        x = np.clip(x, 0, 1)
        
    return x