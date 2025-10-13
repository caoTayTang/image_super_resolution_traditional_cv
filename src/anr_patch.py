
import os, cv2, pickle, numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
try:
    # When imported as package: src.anr_patch
    from .interpolation import sr_interpolation
except Exception:
    # Fallback for script-style execution
    from src.interpolation import sr_interpolation

def rgb2yiq(img_rgb: np.ndarray) -> np.ndarray:
    # NTSC color scheme
    M = np.array([[0.299,  0.587,  0.114],
                  [0.596, -0.274, -0.322],
                  [0.211, -0.523,  0.312]], dtype=np.float32)
    h, w, _ = img_rgb.shape
    return (img_rgb.reshape(-1,3) @ M.T).reshape(h,w,3)

def yiq2rgb(img_yiq: np.ndarray) -> np.ndarray:
    # Inverse of NTSC color scheme
    M_inv = np.array([[1.0,  0.956,  0.621],
                      [1.0, -0.272, -0.647],
                      [1.0, -1.106,  1.703]], dtype=np.float32)
    h, w, _ = img_yiq.shape
    return (img_yiq.reshape(-1,3) @ M_inv.T).reshape(h,w,3)

def _conv2d_same(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)

def compute_y_derivatives(y: np.ndarray):
    kdx = np.array([[0,0,0],[-1,0,1],[0,0,0]], np.float32)
    kdy = kdx.T
    kdxx = np.array([[0,0,0],[1,-2,1],[0,0,0]], np.float32)
    kdyy = kdxx.T
    dx  = _conv2d_same(y, kdx)
    dy  = _conv2d_same(y, kdy)
    dxx = _conv2d_same(y, kdxx)
    dyy = _conv2d_same(y, kdyy)
    return dx, dy, dxx, dyy

def feature_map_from_y(y_up: np.ndarray) -> np.ndarray:
    dx, dy, dxx, dyy = compute_y_derivatives(y_up)
    return np.stack([dx, dy, dxx, dyy], axis=-1).astype(np.float32)

def extract_patches_from_feature_map(F: np.ndarray, patch_size: int, step: int):
    H, W, C = F.shape
    ps = patch_size
    feats, coords = [], []
    for i in range(0, H-ps+1, step):
        for j in range(0, W-ps+1, step):
            feats.append(F[i:i+ps, j:j+ps, :].reshape(-1))
            coords.append((i,j))
    if not feats:
        return np.zeros((0, ps*ps*C), np.float32), []
    return np.stack(feats,0).astype(np.float32), coords

def extract_y_patches(y: np.ndarray, coords: List[Tuple[int,int]], patch_size: int):
    ps = patch_size
    patches = [ y[i:i+ps, j:j+ps].reshape(-1) for (i,j) in coords ]
    if not patches:
        return np.zeros((0, ps*ps), np.float32)
    return np.stack(patches,0).astype(np.float32)

def reconstruct_from_patches(coords, patches, H, W, patch_size):
    ps = patch_size
    acc = np.zeros((H,W), np.float32)
    wgt = np.zeros((H,W), np.float32)
    for (i,j), pvec in zip(coords, patches):
        p = pvec.reshape(ps,ps)
        acc[i:i+ps, j:j+ps] += p
        wgt[i:i+ps, j:j+ps] += 1.0
    wgt[wgt==0] = 1.0
    return acc / wgt

@dataclass
class APlusConfig:
    scale: int = 2
    patch_size: int = 7
    step: int = 3
    n_anchors: int = 1024
    pca_dim: int = 30
    ridge_lambda: float = 1e-2
    rng_seed: int = 42

@dataclass
class APlusModel:
    cfg: APlusConfig
    pca: PCA
    kmeans: KMeans
    projections: List[np.ndarray]
    feat_dim: int
    patch_dim: int

def _read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

# sr_interpolation(img, method) replace with sr_interpolation(img, "bicubic")
def _bicubic_resize(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    # Deprecated in favor of sr_interpolation(..., method='bicubic', scale=s)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def build_training_matrices(lr_dir: str, hr_dir: str, cfg: APlusConfig):
    # Giai đoạn chuẩn bị dữ liệu huấn luyện cho A+:
    # - Tạo ma trận đặc trưng X từ patch đặc trưng Y (đạo hàm) của ảnh LR đã nội suy lên HR (bicubic)
    # - Tạo ma trận mục tiêu Y là phần dư (residual) giữa patch Y_HR và patch Y_bicubic (HR - bicubic)
    lr_list = sorted([os.path.join(lr_dir,f) for f in os.listdir(lr_dir) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))])
    hr_list = sorted([os.path.join(hr_dir,f) for f in os.listdir(hr_dir) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))])
    assert len(lr_list)==len(hr_list) and len(lr_list)>0, "LR/HR folders must be paired & non-empty"

    all_X, all_Y = [], []
    ps = cfg.patch_size
    for lr_path, hr_path in zip(lr_list, hr_list):
        # 1) Đọc ảnh và chuyển sang không gian YIQ, lấy kênh độ sáng Y
        lr_rgb = _read_rgb(lr_path)
        hr_rgb = _read_rgb(hr_path)
        lr_yiq = rgb2yiq(lr_rgb)
        hr_yiq = rgb2yiq(hr_rgb)
        lr_Y = lr_yiq[:,:,0]; hr_Y = hr_yiq[:,:,0]
        # 2) Nội suy bicubic LR_Y lên kích thước HR theo scale
        H_hr, W_hr = hr_Y.shape
        lr_Y_up = sr_interpolation(lr_Y, method='bicubic', scale=cfg.scale)

        # 3) Tính đặc trưng từ Y_up (đạo hàm bậc 1/2) và trích các patch đặc trưng
        F = feature_map_from_y(lr_Y_up)
        X_raw, coords = extract_patches_from_feature_map(F, ps, cfg.step)

        # 4) Tạo patch mục tiêu là phần dư Y_residual = Y_HR - Y_bicubic
        Y_hr = extract_y_patches(hr_Y, coords, ps)
        Y_lr = extract_y_patches(lr_Y_up, coords, ps)
        Y_residuals = (Y_hr - Y_lr).astype(np.float32)

        if X_raw.shape[0] == Y_residuals.shape[0]:
            all_X.append(X_raw); all_Y.append(Y_residuals)

    X = np.concatenate(all_X, 0) if all_X else np.zeros((0, ps*ps*4), np.float32)
    Y = np.concatenate(all_Y, 0) if all_Y else np.zeros((0, ps*ps), np.float32)
    return X, Y

def train_aplus(lr_dir: str, hr_dir: str, cfg: Optional[APlusConfig]=None) -> APlusModel:
    if cfg is None:
        cfg = APlusConfig()
    np.random.seed(cfg.rng_seed)

    # Bước 1: Trích xuất dữ liệu huấn luyện (X_raw đặc trưng patch, Y residual)
    X_raw, Y = build_training_matrices(lr_dir, hr_dir, cfg)
    assert X_raw.shape[0] > 0, "No training samples extracted."

    # Bước 2: Giảm chiều + whitening với PCA để làm ổn định hồi quy
    pca = PCA(n_components=min(cfg.pca_dim, X_raw.shape[1]), random_state=cfg.rng_seed, whiten=True)
    X = pca.fit_transform(X_raw)

    # Bước 3: Phân cụm KMeans trên không gian PCA để tạo các anchor (từ điển địa phương)
    kmeans = KMeans(n_clusters=cfg.n_anchors, random_state=cfg.rng_seed, n_init='auto')
    labels = kmeans.fit_predict(X)

    d = X.shape[1]; m = Y.shape[1]
    projections: List[np.ndarray] = []
    # Bước 4: Với mỗi anchor, học ma trận chiếu tuyến tính (ridge regression)
    for a in range(cfg.n_anchors):
        idx = np.where(labels==a)[0]
        Xa = X if len(idx) < d+1 else X[idx]
        Ya = Y if len(idx) < d+1 else Y[idx]
        XtX = Xa.T @ Xa
        XtY = Xa.T @ Ya
        P = np.linalg.solve(XtX + cfg.ridge_lambda*np.eye(d, dtype=np.float32), XtY)
        projections.append(P.astype(np.float32))

    return APlusModel(cfg=cfg, pca=pca, kmeans=kmeans, projections=projections, feat_dim=d, patch_dim=m)

def save_model(model: APlusModel, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f: pickle.dump(model, f)

def load_model(path: str) -> APlusModel:
    with open(path, 'rb') as f: return pickle.load(f)

def predict_image_aplus_array(lr_rgb: np.ndarray, model: APlusModel) -> np.ndarray:
    """
    Predict SR result from an in-memory RGB image (H,W,3).
    Accepts uint8 [0-255] or float [0..1]/[0..255]. Returns RGB uint8.
    """
    cfg = model.cfg

    # Bước 0: Chuẩn hóa dữ liệu ảnh về [0,1]
    lr_rgb = lr_rgb.astype(np.float32)
    if lr_rgb.max() > 1.5:
        lr_rgb = lr_rgb / 255.0

    H_lr, W_lr, _ = lr_rgb.shape
    s = cfg.scale

    # Bước 1: Chuyển RGB -> YIQ và tách các kênh
    lr_yiq = rgb2yiq(lr_rgb)
    Y = lr_yiq[:, :, 0]; I = lr_yiq[:, :, 1]; Q = lr_yiq[:, :, 2]

    # Bước 2: Nội suy bicubic Y, I, Q lên kích thước HR theo scale
    H_hr, W_hr = H_lr * s, W_lr * s
    Y_up = sr_interpolation(Y, method='bicubic', scale=s)
    I_hr = sr_interpolation(I, method='bicubic', scale=s)
    Q_hr = sr_interpolation(Q, method='bicubic', scale=s)

    # Bước 3: Tính đặc trưng từ Y_up và trích patch đặc trưng
    F = feature_map_from_y(Y_up)
    X_raw, coords = extract_patches_from_feature_map(F, cfg.patch_size, cfg.step)
    if not coords:
        yiq = np.stack([Y_up, I_hr, Q_hr], -1)
        rgb = np.clip(yiq2rgb(yiq), 0, 1)
        return (rgb * 255.0 + 0.5).astype(np.uint8)

    # Bước 4: Ánh xạ đặc trưng qua PCA và gán anchor gần nhất
    X = model.pca.transform(X_raw)

    centers = model.kmeans.cluster_centers_
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1, keepdims=True).T
    dist2 = x2 + c2 - 2.0 * (X @ centers.T)
    nearest = np.argmin(dist2, axis=1)

    # Bước 5: Với mỗi anchor, áp dụng ma trận chiếu để dự đoán phần dư Y
    preds = np.zeros((X.shape[0], model.patch_dim), np.float32)
    for a in range(cfg.n_anchors):
        idx = np.where(nearest == a)[0]
        if len(idx) == 0:
            continue
        P = model.projections[a]
        preds[idx] = X[idx] @ P

    # Bước 6: Cộng residual vào Y_bicubic và ghép lại ảnh Y theo cơ chế overlap-add
    Y_base = extract_y_patches(Y_up, coords, cfg.patch_size)
    Y_patches = Y_base + preds
    Y_rec = reconstruct_from_patches(coords, Y_patches, H_hr, W_hr, cfg.patch_size)

    # Bước 7: Ghép kênh Y tái tạo với I/Q nội suy và chuyển YIQ -> RGB
    yiq = np.stack([Y_rec, I_hr, Q_hr], -1)
    rgb = np.clip(yiq2rgb(yiq), 0, 1)
    return (rgb * 255.0 + 0.5).astype(np.uint8)

def predict_image_aplus(lr_path: str, model: APlusModel) -> np.ndarray:

    lr_rgb = _read_rgb(lr_path)
    return predict_image_aplus_array(lr_rgb, model)

def sr_anr(
    img,
    ckpt_path: str,
    train_lr_dir: Optional[str] = None,
    train_hr_dir: Optional[str] = None,
    cfg_dict: Optional[dict] = None,
    input_bgr: bool = True,
) -> np.ndarray:
    # Hàm tiện ích cấp cao:
    # - Bước 1: Nếu có checkpoint thì load; nếu chưa có sẽ train A+ từ thư mục LR/HR
    # - Bước 2: Dự đoán cho ảnh (đường dẫn hoặc mảng numpy). Nếu là BGR thì chuyển sang RGB trước khi xử lý
    if os.path.exists(ckpt_path):
        model = load_model(ckpt_path)
    else:
        if not train_lr_dir or not train_hr_dir:
            raise ValueError("train_lr_dir/train_hr_dir are required to train when checkpoint is missing.")
        cfg = APlusConfig(**cfg_dict) if cfg_dict is not None else APlusConfig()
        model = train_aplus(train_lr_dir, train_hr_dir, cfg)
        save_model(model, ckpt_path)

    if isinstance(img, str):
        return predict_image_aplus(img, model)
    elif isinstance(img, np.ndarray):
        lr_rgb = img
        if input_bgr:
            lr_rgb = cv2.cvtColor(lr_rgb, cv2.COLOR_BGR2RGB)
        return predict_image_aplus_array(lr_rgb, model)
    else:
        raise TypeError("img must be a filepath (str) or a numpy array")

def train_and_save(lr_dir: str, hr_dir: str, ckpt_path: str, cfg_dict: Optional[dict]=None) -> str:
    cfg = APlusConfig(**cfg_dict) if cfg_dict is not None else APlusConfig()
    model = train_aplus(lr_dir, hr_dir, cfg)
    save_model(model, ckpt_path)
    return ckpt_path

def run_inference_dir(lr_dir: str, ckpt_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    model = load_model(ckpt_path)
    files = [f for f in sorted(os.listdir(lr_dir)) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
    for fname in files:
        lr_path = os.path.join(lr_dir, fname)
        hr_rgb = predict_image_aplus(lr_path, model)
        out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + '_ANR_Aplus.png')
        cv2.imwrite(out_path, cv2.cvtColor(hr_rgb, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="A+ SR on Y with bicubic I/Q")
    sub = p.add_subparsers(dest="cmd")

    t = sub.add_parser("train")
    t.add_argument("--lr_dir", required=True)
    t.add_argument("--hr_dir", required=True)
    t.add_argument("--ckpt", required=True)
    t.add_argument("--scale", type=int, default=2)
    t.add_argument("--patch_size", type=int, default=7)
    t.add_argument("--step", type=int, default=3)
    t.add_argument("--n_anchors", type=int, default=1024)
    t.add_argument("--pca_dim", type=int, default=30)
    t.add_argument("--ridge_lambda", type=float, default=1e-2)
    t.add_argument("--rng_seed", type=int, default=42)

    r = sub.add_parser("infer")
    r.add_argument("--lr_dir", required=True)
    r.add_argument("--ckpt", required=True)
    r.add_argument("--out_dir", required=True)

    args = p.parse_args()
    if args.cmd == "train":
        cfg = dict(scale=args.scale, patch_size=args.patch_size, step=args.step,
                   n_anchors=args.n_anchors, pca_dim=args.pca_dim,
                   ridge_lambda=args.ridge_lambda, rng_seed=args.rng_seed)
        train_and_save(args.lr_dir, args.hr_dir, args.ckpt, cfg)
    elif args.cmd == "infer":
        run_inference_dir(args.lr_dir, args.ckpt, args.out_dir)
    else:
        p.print_help()
