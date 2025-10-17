import numpy as np
from scipy.fft import fft2, ifft2, fftfreq, fftshift, rfftn, irfftn
from scipy.signal import convolve2d, fftconvolve
from scipy.fft import rfftn, irfftn
from scipy.ndimage import gaussian_filter



def _pad_to_shape(ir, shape):
    """Zero-pad IR đến cùng kích thước ảnh và circular shift tâm về (0,...)."""
    out = np.zeros(shape, dtype=ir.dtype)
    insert_slices = tuple(slice(0, s) for s in ir.shape)
    out[insert_slices] = ir
    for ax, k in enumerate(ir.shape):
        out = np.roll(out, - (k // 2), axis=ax)
    return out


def _laplacian_ir(ndim=2):
    """Sinh Laplacian kernel kích thước 3^N."""
    shape = (3,) * ndim
    ir = np.zeros(shape, dtype=float)
    center = (shape[0] // 2,) * ndim
    ir[center] = -2 * ndim
    for ax in range(ndim):
        idx_plus = list(center)
        idx_minus = list(center)
        idx_plus[ax] = 2
        idx_minus[ax] = 0
        ir[tuple(idx_plus)] = 1.0
        ir[tuple(idx_minus)] = 1.0
    return ir


def _gaussian_psf(size=5, sigma=1.5, ndim=2):
    """Sinh Gaussian PSF chuẩn hóa (sum=1)."""
    # tạo grid
    coords = [np.arange(size) - size // 2 for _ in range(ndim)]
    grids = np.meshgrid(*coords, indexing='ij')
    dist2 = sum((g ** 2 for g in grids))
    psf = np.exp(-dist2 / (2 * sigma ** 2))
    psf /= psf.sum()
    return psf


def wiener_base_real(image, psf=None, balance=0.01, reg=None,
                     clip=True, eps=1e-12, psf_size=5, psf_sigma=1.5):
    """
    Wiener deconvolution cho ảnh thực

    Parameters
    ----------
    image : ndarray
        Ảnh đầu vào (real-valued).
    psf : ndarray or None
        Point Spread Function (impulse response). Nếu None → Gaussian.
    balance : float
        Hệ số điều chỉnh λ giữa fidelity và regularization.
    reg : ndarray or None
        Toán tử regularization (mặc định Laplacian).
    clip : bool
        Nếu True, giới hạn output về [-1,1].
    eps : float
        Giá trị nhỏ tránh chia cho 0.
    psf_size : int
        Kích thước kernel Gaussian mặc định.
    psf_sigma : float
        Sigma mặc định cho PSF Gaussian.

    Returns
    -------
    ndarray : ảnh khôi phục (cùng shape với input)
    """
    image = np.asarray(image, dtype=np.float64)
    ndim = image.ndim
    shape = image.shape

    # PSF mặc định = Gaussian kernel
    if psf is None:
        psf = _gaussian_psf(psf_size, psf_sigma, ndim)

    # Regularization mặc định = Laplacian
    if reg is None:
        reg = _laplacian_ir(ndim)

    # Fourier transform
    H = rfftn(_pad_to_shape(psf, shape))
    D = rfftn(_pad_to_shape(reg, shape))
    Y = rfftn(image)

    # Wiener filter: conj(H) / (|H|^2 + λ|D|^2)
    denom = np.abs(H)**2 + balance * np.abs(D)**2
    denom = np.where(denom < eps, eps, denom)
    W = np.conj(H) / denom

    # Áp dụng và inverse FFT
    X = irfftn(W * Y, s=shape)

    if clip:
        X = np.clip(X, -1.0, 1.0)

    return X


# Generate Gaussian PSF with correct centering
def get_psf_sigma(sigma, shape):
    x = np.arange(-shape[0]//2, shape[0]//2)
    y = np.arange(-shape[1]//2, shape[1]//2)
    X, Y = np.meshgrid(x, y)  # Correct order: x, y
    psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    psf /= psf.sum()
    return fftshift(psf)  # Center the PSF

# Wiener filter cơ bản
def wiener_base(img, kernel, K=0.01):
    """
    img: Ảnh bị mờ (2D array, float [0,1])
    kernel: Kernel mờ (2D array, float) - Dự đoán
    K: Hằng số ổn định (float, thường nhỏ, ví dụ 0.01)
    """
    kernel = kernel / np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s=img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel)**2 + K)
    dummy = dummy * kernel
    dummy = np.real(ifft2(dummy))
    return dummy


def unsupervised_wiener_improved(image, psf_init=1, reg=None, user_params=None, clip=True, rng=None):
    """
    Improved unsupervised Wiener filter for grayscale or RGB images using Gibbs sampler.
    Fixes quadrant swap and 180-degree flip issues by centering PSF and correct meshgrid indexing.
    Input:
        image: Grayscale (2D) or RGB (3D) float array [0,1]
        psf_init: Initial sigma for Gaussian PSF (float)
        reg: Regularization transfer function (ndarray, optional; default Laplacian)
        user_params: Dict with 'threshold' (1e-4), 'burnin' (15), 'min_num_iter' (30), 'max_num_iter' (200)
        clip: Clip output to [0,1]
        rng: np.random.Generator (optional)
    Output:
        deconvolved: Restored image (same shape as input)
        psf: Estimated PSF (2D array)
    """
    # Defaults
    if user_params is None:
        user_params = {'threshold': 1e-4, 'burnin': 15, 'min_num_iter': 30, 'max_num_iter': 200}
    if rng is None:
        rng = np.random.default_rng()

    # Check if image is RGB (3D) or grayscale (2D)
    image = np.asarray(image, dtype=float)
    is_rgb = len(image.shape) == 3 and image.shape[-1] == 3
    if is_rgb:
        restored_img = np.zeros_like(image)
        psf_final = None
        for c in range(3):
            restored_img[..., c], psf = unsupervised_wiener_improved(
                image[..., c], psf_init, reg, user_params, clip, rng
            )
            if c == 0:  # Store PSF from first channel (assume same for all)
                psf_final = psf
        return restored_img, psf_final

    # Normalize image to [0,1]
    if image.max() > 1.0:
        image /= 255.0

    # Shapes
    shape = image.shape
    N = shape[0] * shape[1]

    # Fourier grid for Laplacian regularization
    if reg is None:
        fx = fftfreq(shape[0])
        fy = fftfreq(shape[1])
        FX, FY = np.meshgrid(fx, fy)  # Correct order: x, y
        reg = 2 * (2 - np.cos(2 * np.pi * FX) - np.cos(2 * np.pi * FY))

    psf = get_psf_sigma(psf_init, shape)
    Lambda_H = fft2(psf)  # PSF transfer function

    # Initial values
    gamma_eps = N / np.linalg.norm(fft2(image))**2
    gamma_1 = 1.0
    sigma_psf = psf_init
    ft_img = fft2(image)
    ft_x = np.copy(ft_img)

    # Gibbs sampling
    x_samples = []
    prev_mean = np.zeros(shape, dtype=complex)
    k = 0
    converged = False
    while not converged and k < user_params['max_num_iter']:
        # Step 1: Sample image circ x^(k+1)
        abs_Lambda_H_sq = np.abs(Lambda_H)**2
        Sigma_inv = gamma_eps * abs_Lambda_H_sq + gamma_1 * reg
        Sigma = 1 / (Sigma_inv + 1e-10)
        mu = gamma_eps * Sigma * np.conj(Lambda_H) * ft_img
        eta_real = rng.normal(0, 1, shape)
        eta_imag = rng.normal(0, 1, shape)
        eta = eta_real + 1j * eta_imag
        ft_x = mu + np.sqrt(Sigma) * eta / np.sqrt(2)

        # Step 2: Sample gamma_eps
        residual = ft_img - Lambda_H * ft_x
        beta_eps = np.linalg.norm(residual)**2 / 2
        alpha_eps = N / 2
        gamma_eps = rng.gamma(alpha_eps, 1 / beta_eps) if beta_eps > 0 else 1e-6

        # Step 3: Sample gamma_1
        dx = reg * np.abs(ft_x)**2
        beta_1 = np.sum(dx) / 2
        alpha_1 = (N - 1) / 2
        gamma_1 = rng.gamma(alpha_1, 1 / beta_1) if beta_1 > 0 else 1e-6

        # Step 4: Sample PSF sigma via Metropolis-Hastings
        sigma_p = 0.1 + rng.random() * 9.9
        psf_p = get_psf_sigma(sigma_p, shape)
        Lambda_H_p = fft2(psf_p)
        resid_old = ft_img - Lambda_H * ft_x
        resid_p = ft_img - Lambda_H_p * ft_x
        J = (gamma_eps / 2) * (np.linalg.norm(resid_old)**2 - np.linalg.norm(resid_p)**2)
        if np.log(rng.random()) < min(J, 0):
            sigma_psf = sigma_p
            Lambda_H = Lambda_H_p

        # Collect samples
        k += 1
        if k > user_params['burnin']:
            x_samples.append(ft_x)
            if len(x_samples) >= user_params['min_num_iter']:
                current_mean = np.mean(x_samples, axis=0)
                rel_change = np.linalg.norm(current_mean - prev_mean) / np.linalg.norm(current_mean)
                if rel_change < user_params['threshold']:
                    converged = True
                prev_mean = current_mean

    # Final deconvolved
    if not x_samples:
        x_samples = [ft_x]
    ft_mean = np.mean(x_samples, axis=0)
    deconvolved = np.real(ifft2(ft_mean))
    if clip:
        deconvolved = np.clip(deconvolved, 0, 1)

    # Estimated PSF
    estimated_psf = get_psf_sigma(sigma_psf, shape)

    return deconvolved, estimated_psf



# Helper function to pad kernel to a given shape and compute FFT
def fft_pad(kernel, shape):
    pad_y = shape[0] - kernel.shape[0]
    pad_x = shape[1] - kernel.shape[1]
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left
    padded = np.pad(kernel, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    return np.fft.fft2(padded)

# Compute |D|^2 in frequency domain (D is gradient operator)
def get_D_power(shape):
    # Horizontal gradient kernel [1, -1]
    dx = np.array([[1, -1]])
    # Vertical gradient kernel [1; -1]
    dy = np.array([[1], [-1]])
    Dx = fft_pad(dx, shape)
    Dy = fft_pad(dy, shape)
    return np.abs(Dx)**2 + np.abs(Dy)**2

# Learn restoration filter w1 using FD closed-form and crop to spatial domain
def learn_restoration_filter(h, gamma, s, fft_shape=(256, 256)):
    H = fft_pad(h, fft_shape)
    D2 = get_D_power(fft_shape)
    # Wiener-like filter in FD
    W = np.conj(H) / (np.abs(H)**2 + (1 / gamma) * D2 + 1e-10)  # Small epsilon to avoid division issues
    w_fd = np.fft.ifft2(W).real
    # Crop center to s x s
    half = s // 2
    center_y, center_x = fft_shape[0] // 2, fft_shape[1] // 2
    w_crop = w_fd[center_y - half:center_y + half + 1, center_x - half:center_x + half + 1]
    # Adjust to preserve sum (DC component)
    current_sum = np.sum(w_crop)
    desired_sum = np.real(W[0, 0])
    add = (desired_sum - current_sum) / (s * s)
    w1 = w_crop + add
    return w1

# Learn update filters w2x and w2y using FD closed-form and crop to spatial domain
def learn_update_filters(h, gamma, beta, s, fft_shape=(256, 256)):
    H = fft_pad(h, fft_shape)
    D2 = get_D_power(fft_shape)
    denom = D2 + (gamma / beta) * np.abs(H)**2 + 1e-10  # Small epsilon
    # Horizontal
    dx = np.array([[1, -1]])
    Dx = fft_pad(dx, fft_shape)
    Wx = np.conj(Dx) / denom
    wx_fd = np.fft.ifft2(Wx).real
    half = s // 2
    center_y, center_x = fft_shape[0] // 2, fft_shape[1] // 2
    wx_crop = wx_fd[center_y - half:center_y + half + 1, center_x - half:center_x + half + 1]
    current_sum_x = np.sum(wx_crop)
    desired_sum_x = np.real(Wx[0, 0])
    add_x = (desired_sum_x - current_sum_x) / (s * s)
    wx = wx_crop + add_x
    # Vertical
    dy = np.array([[1], [-1]])
    Dy = fft_pad(dy, fft_shape)
    Wy = np.conj(Dy) / denom
    wy_fd = np.fft.ifft2(Wy).real
    wy_crop = wy_fd[center_y - half:center_y + half + 1, center_x - half:center_x + half + 1]
    current_sum_y = np.sum(wy_crop)
    desired_sum_y = np.real(Wy[0, 0])
    add_y = (desired_sum_y - current_sum_y) / (s * s)
    wy = wy_crop + add_y
    return wx, wy

# Compute image gradients (Du)
def compute_gradient(u):
    dx = np.array([[1, -1]])
    dy = np.array([[1], [-1]])
    ux = convolve2d(u, dx, mode='same')
    uy = convolve2d(u, dy, mode='same')
    return ux, uy

# Iterative Wiener Filtering and Thresholding (IWFT) - Algorithm 2
def iwft(g, gamma, h=None, beta=None, s=15, N=15, tol=1e-4, fft_shape=(256, 256)):

    if h is None:
        h = np.ones((5, 5)) / 25.0  
    # Learn filters
    w1 = learn_restoration_filter(h, gamma, s, fft_shape)
    wx, wy = learn_update_filters(h, gamma, beta if beta else 10 * np.max(g), s, fft_shape)
    
    # Initial estimation
    u1 = convolve2d(g, w1, mode='same')
    u = u1.copy()
    
    # Default beta if not provided
    if beta is None:
        beta = 10 * np.max(g)
    
    # Initialize a (Lagrange multiplier, 2 channels)
    ax = np.zeros_like(g)
    ay = np.zeros_like(g)
    
    k = N
    while k > 0:
        u_old = u.copy()
        
        # Compute Du
        ux, uy = compute_gradient(u)
        
        # Compute norm ||Du - a||_2
        diff_x = ux - ax
        diff_y = uy - ay
        norm2 = np.sqrt(diff_x**2 + diff_y**2)
        
        # Soft thresholding
        thresh = np.maximum(norm2 - 1 / beta, 0)
        denom = norm2 + 1e-10  # Avoid division by zero
        vx = diff_x * (thresh / denom)
        vy = diff_y * (thresh / denom)
        
        # Update a
        ax = ax - ux + vx
        ay = ay - uy + vy
        
        # Update u
        update_x = convolve2d(vx + ax, wx, mode='same')
        update_y = convolve2d(vy + ay, wy, mode='same')
        u = u1 + update_x + update_y
        
        # Check relative tolerance
        rel_change = np.linalg.norm(u - u_old) / (np.linalg.norm(u_old) + 1e-10)
        if rel_change < tol:
            break
        
        k -= 1
    
    return u