import numpy as np
from scipy.fft import fft2, ifft2, fftfreq, fftshift, rfftn, irfftn
from scipy.fft import rfftn, irfftn
from numpy.random import gamma as sample_gamma



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


def gaussian_psf(size=5, sigma=1.5, ndim=2):
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
        psf = gaussian_psf(psf_size, psf_sigma, ndim)

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

def wiener_unsupervised_mcmc(
    image, 
    psf=None, 
    reg=None,
    n_iter=100,
    burn_in=20,
    clip=True, 
    eps=1e-12, 
    psf_size=15, 
    psf_sigma=5
):
    """
    Wiener deconvolution không giám sát sử dụng MCMC (Gibbs Sampler)
    để ước tính các siêu tham số.

    Hàm này thay thế việc truyền vào 'balance' cố định bằng cách
    tự động lấy mẫu các tham số precision (gamma_epsilon, gamma_1)
    dựa trên Mục 5.B của bài báo.

    Parameters
    ----------
    image : ndarray
        Ảnh đầu vào (real-valued).
    psf : ndarray or None
        Point Spread Function. Nếu None -> Gaussian.
    reg : ndarray or None
        Toán tử regularization (mặc định Laplacian).
    n_iter : int
        Tổng số lần lặp MCMC.
    burn_in : int
        Số lần lặp ban đầu để loại bỏ (giai đoạn 'cháy').
    clip : bool
        Nếu True, giới hạn output về [-1, 1].
    eps : float
        Giá trị nhỏ tránh chia cho 0.
    
    Returns
    -------
    ndarray : ảnh khôi phục (cùng shape với input)
    """
    
    # 1. CHUẨN HÓA ẢNH (QUAN TRỌNG NHẤT)
    # Đưa ảnh về khoảng [0, 1] để tránh residual quá lớn làm sập gamma_e
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min > eps:
        image_norm = (image - img_min) / (img_max - img_min)
    else:
        image_norm = image.copy()

    image_norm = np.asarray(image_norm, dtype=np.float64)
    ndim = image_norm.ndim
    shape = image_norm.shape
    N = image_norm.size

    # --- Khởi tạo ---
    # PSF mặc định = Gaussian kernel
    if psf is None:
        psf = gaussian_psf(psf_size, psf_sigma, ndim)

    # Regularization mặc định = Laplacian [cite: 348]
    if reg is None:
        reg = _laplacian_ir(ndim)

    # Chuyển đổi sang miền Fourier
    H = rfftn(_pad_to_shape(psf, shape))
    D = rfftn(_pad_to_shape(reg, shape))
    Y = rfftn(image_norm)

    # Các thành phần tiền tính toán
    H_conj = np.conj(H)
    H_abs_sq = np.abs(H)**2
    D_abs_sq = np.abs(D)**2

    # Khởi tạo các siêu tham số (precision)
    gamma_epsilon = 1.0 / (np.var(image_norm) * 0.01 + eps) # Giả định nhiễu thấp ban đầu
    gamma_1 = 1.0 / np.var(image_norm)

    alpha0_eps, beta0_eps = 1e-2, 1e-2
    alpha0_1, beta0_1 = 1e-2, 1e-2
    GAMMA_MIN, GAMMA_MAX = 0.031, 1e2

    # Khởi tạo ảnh phục hồi (X_fft)
    # Bắt đầu bằng chính ảnh mờ
    X_fft = Y.copy()

    # Biến để lưu trữ tổng các mẫu (để lấy trung bình)
    X_samples_sum = np.zeros_like(Y, dtype=complex)
    samples_count = 0
    
    print(f"Bắt đầu MCMC với {n_iter} lần lặp (burn-in: {burn_in})...")

    # --- Vòng lặp MCMC (Gibbs Sampler) ---
    for k in range(n_iter):
        
        # 1. LẤY MẪU ẢNH (Mục 5.A)
        # ---------------------------
        # Tính toán bộ lọc Wiener-Hunt dựa trên các tham số gamma hiện tại
        # Đây chính là phần lõi của hàm wiener_base_real cũ
        
        # balance = gamma_1 / gamma_epsilon
        # W = H_conj / (H_abs_sq + balance * D_abs_sq)
        
        # Thay vì tính 'balance', ta dùng công thức gốc từ Mục 5.A:
        # mu = gamma_e * (gamma_e * |H|^2 + gamma_1 * |D|^2)^-1 * H_conj * Y
        # X_fft = mu 
        # (Đây là ước tính MAP/Mean, không phải mẫu đầy đủ, nhưng phổ biến)

        inv_sigma_fft = gamma_epsilon * H_abs_sq + gamma_1 * D_abs_sq
        sigma_fft = 1.0 / np.where(inv_sigma_fft < eps, eps, inv_sigma_fft)

        mu_fft = gamma_epsilon * sigma_fft * H_conj * Y

        # Thay vì X_fft = mu, ta cần X ~ N(mu, sigma)
        # Trong miền tần số: X = mu + sqrt(sigma) * noise
        # Noise phức chuẩn tắc
        noise_real = np.random.standard_normal(X_fft.shape)
        noise_imag = np.random.standard_normal(X_fft.shape)
        complex_noise = (noise_real + 1j * noise_imag)
        
        # Tính trung bình hậu nghiệm mu
        # X_fft = gamma_epsilon * sigma_fft * H_conj * Y
        X_fft = mu_fft + np.sqrt(sigma_fft) * complex_noise * 0.5 
        
        
        # 2. LẤY MẪU CÁC THAM SỐ PRECISION (Mục 5.B)
        # ---------------------------------------------
        # Sử dụng Jeffreys' prior (alpha=0, beta_inv=0)
        
        # A. Lấy mẫu gamma_epsilon (Noise precision)
        # alpha_eps = N / 2.0
        # residual_sq_sum = np.sum(np.abs(Y - H * X_fft)**2)
        # beta_eps = 2.0 / np.where(residual_sq_sum < eps, eps, residual_sq_sum)
        # gamma_epsilon = sample_gamma(alpha_eps, beta_eps)
        alpha_eps = alpha0_eps + N / 2.0
        residual_sq_sum = np.sum(np.abs(Y - H * X_fft)**2)
        beta_eps = beta0_eps + residual_sq_sum / 2.0
        gamma_epsilon = np.clip(sample_gamma(alpha_eps, 1.0/beta_eps), GAMMA_MIN, GAMMA_MAX)
        
        # B. Lấy mẫu gamma_1 (Image/Regularization precision)
        # alpha_1 = (N - 1) / 2.0 # (Bỏ qua 1 bậc tự do của gamma_0)
        # reg_sq_sum = np.sum(np.abs(D * X_fft)**2)
        # beta_1 = 2.0 / np.where(reg_sq_sum < eps, eps, reg_sq_sum)
        # gamma_1 = sample_gamma(alpha_1, beta_1)
        alpha_1 = alpha0_1 + (N - 1) / 2.0
        reg_sq_sum = np.sum(np.abs(D * X_fft)**2)
        beta_1 = beta0_1 + reg_sq_sum / 2.0
        gamma_1 = np.clip(sample_gamma(alpha_1, 1.0/beta_1), GAMMA_MIN, GAMMA_MAX)

        # 3. LƯU TRỮ MẪU (SAU KHI BURN-IN)
        # ---------------------------------
        if k >= burn_in:
            X_samples_sum += mu_fft
            samples_count += 1
            
        if (k + 1) % 10 == 0:
            balance = gamma_1 / (gamma_epsilon + eps)
            print(f" Iter {k+1}: ge={gamma_epsilon:.3f}, g1={gamma_1:.3f}, bal={balance:.4f}, Res={residual_sq_sum:.4f}")

    # --- HOÀN TẤT ---
    # Tính toán ước tính cuối cùng bằng trung bình các mẫu
    # Đây là ước tính trung bình hậu nghiệm (posterior mean)
    if samples_count == 0:
        print("Cảnh báo: Không có mẫu nào được thu thập (burn_in >= n_iter)")
        X_mean_fft = X_fft # Trả về mẫu cuối cùng
    else:
        X_mean_fft = X_samples_sum / samples_count
    
    X_final = irfftn(X_mean_fft, s=shape)
    
    # 4. KHÔI PHỤC SCALE GỐC (DENORMALIZE)
    if img_max - img_min > eps:
        X_final = X_final * (img_max - img_min) + img_min
        
    if clip:
        # Clip theo giá trị gốc của ảnh đầu vào (ví dụ 0-255) hoặc -1,1 tuỳ input
        # Ở đây clip theo min/max ban đầu của ảnh
        X_final = np.clip(X_final, image.min(), image.max())

    return X_final