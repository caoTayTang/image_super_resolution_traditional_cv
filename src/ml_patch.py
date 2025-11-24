from src import utils
import cv2
import numpy as np
import os
from glob import glob
from sklearn.neighbors import NearestNeighbors

def extract_patches(lr_folder, hr_folder, patch_size=5, scale=2, overlap=0.5):
    """
    Chia ảnh LR và HR thành patch
    Args:
        lr_folder: folder chứa LR images
        hr_folder: folder chứa HR images
        patch_size: kích thước patch LR (n x n)
        scale: scale factor giữa HR và LR
        overlap: tỉ lệ chồng lấn (0~1)
    Returns:
        LR_patches: list các patch LR theo kênh màu YIQ
        HR_patches: list các patch HR theo kênh màu YIQ
    """
    LR_patches = []
    HR_patches = []
    LR_features_patches = []
    HR_centered_patches = []
    LR_y_means = []

    step = int(patch_size * (1 - overlap))  # step size sliding window

    lr_images = sorted(glob(os.path.join(lr_folder, '*.png')))
    hr_images = sorted(glob(os.path.join(hr_folder, '*.png')))

    assert len(lr_images) == len(hr_images), "Số ảnh LR và HR phải bằng nhau"

    for lr_path, hr_path in zip(lr_images, hr_images):
        lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
        hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)


        h_lr, w_lr = lr_img.shape[:2]
        h_hr, w_hr = hr_img.shape[:2]

        # Kiểm tra HR size có đúng scale
        assert h_hr == h_lr * scale and w_hr == w_lr * scale, \
            f"HR image size must be scale * LR image size: {hr_path}"

        lr_img = rgb2yiq(lr_img.astype(np.float32) / 255.0)
        hr_img = rgb2yiq(hr_img.astype(np.float32) / 255.0)

        lr_y_img = lr_img[:, :, 0]  # Lấy kênh Y của ảnh
        lr_y_img = img_padding(lr_y_img, padding=2, padding_mode='replicate')  # Padding kênh Y, logic đúng cho padding = 2 thôi đừng sửa
        lr_img_features = compute_grads(lr_y_img)  # Tính features gradient bậc 1 và 2 của ảnh

        for i in range(0, h_lr - patch_size + 1, step):
            for j in range(0, w_lr - patch_size + 1, step):
                lr_patch = lr_img[i:i+patch_size, j:j+patch_size, :]
                lr_feature_patch = lr_img_features[i:i+patch_size, j:j+patch_size, :]
                # HR patch tương ứng
                hr_i, hr_j = i*scale, j*scale
                hr_patch = hr_img[hr_i:hr_i+patch_size*scale, hr_j:hr_j+patch_size*scale, :]

                # Lưu mean của LR và centered HR
                lr_patch_y = lr_patch[:, :, 0]
                lr_y_mean = float(np.mean(lr_patch_y))

                hr_patch_y = hr_patch[:, :, 0]
                hr_y_centered = hr_patch_y - lr_y_mean

                LR_patches.append(lr_patch)
                LR_features_patches.append(lr_feature_patch)
                LR_y_means.append(lr_y_mean)

                HR_patches.append(hr_patch)
                HR_centered_patches.append(hr_y_centered)

                

    print(f"Extracted {len(LR_patches)} LR patches, {len(HR_patches)} HR patches, {len(LR_features_patches)} LR feature patches, {len(LR_y_means)} LR_Y_means and {len(HR_centered_patches)} HR_centered_patches from {len(lr_images)} image pairs.")
    return np.array(LR_patches), np.array(HR_patches), np.array(LR_features_patches), np.array(LR_y_means), np.array(HR_centered_patches)

def process_all_patches(base_dir, output_dir, patch_size=5, overlap=0.5):
    """
    Duyệt tất cả scale factor và lưu patch
    base_dir: list: base_dir[0] là dir tới degraded_lr, base_dir[1] là dir tới input_hr
    """
    os.makedirs(output_dir, exist_ok=True)
    scales = [2, 3, 4]

    for s in scales:
        lr_folder = os.path.join(base_dir[0], f'LR_x{s}')
        hr_folder = os.path.join(base_dir[1], f'HR_x{s}') 
        LR_patches, HR_patches, LR_features_patches, LR_y_mean_patches, HR_centered_patches = extract_patches(lr_folder, hr_folder,
                                                  patch_size=patch_size, scale=s,
                                                  overlap=overlap)
        # Lưu file numpy
        np.save(os.path.join(output_dir, f'LR_patches_x{s}.npy'), LR_patches)
        np.save(os.path.join(output_dir, f'HR_patches_x{s}.npy'), HR_patches)
        np.save(os.path.join(output_dir, f'LR_features_patches_x{s}.npy'), LR_features_patches)
        np.save(os.path.join(output_dir, f'LR_y_mean_patches_x{s}.npy'), LR_y_mean_patches)
        np.save(os.path.join(output_dir, f'HR_centered_patches_x{s}.npy'), HR_centered_patches)
        print(f"Saved normalized YIQ LR/HR patches for scale x{s} to {output_dir}")

def rgb2yiq(rgb):
    # Ma trận chuyển đổi từ RGB sang YIQ
    transform = np.array([[0.299,  0.587,  0.114],
                          [0.596, -0.274, -0.322],
                          [0.211, -0.523,  0.312]])
    

    shape = rgb.shape
    flat_rgb = rgb.reshape(-1, 3)
    flat_yiq = np.dot(flat_rgb, transform.T)
    yiq = flat_yiq.reshape(shape)
    return yiq

def yiq2rgb(yiq):
    """
    Chuyển mảng YIQ sang RGB
    yiq: numpy array (H, W, 3), giá trị float 0~1 (có thể <0 hoặc >1)
    Trả về mảng RGB float 0~1
    """
    # ma trận chuyển đổi
    T = np.array([[1.0, 0.956, 0.621],
                  [1.0, -0.272, -0.647],
                  [1.0, -1.106, 1.703]])

    # reshape để nhân ma trận: (H*W,3)
    shape = yiq.shape
    yiq_flat = yiq.reshape(-1, 3)

    rgb_flat = yiq_flat @ T.T  # nhân ma trận
    rgb = rgb_flat.reshape(shape)

    return rgb

def img_padding(img, padding=2, padding_mode='zero'):
    """
    Padding ảnh
    mặc định padding 2 pixel, mode 'zero' mặc định với paper
    Args:
        imgs: ảnh theo channel Y (đã chuyển sang YIQ)
        padding: số pixel cần padding
        padding_mode: 'zero' hoặc 'replicate' hoặc 'reflection'
    Returns:
        padded_img: ảnh đã được padding
    """
    if padding_mode == 'zero':
        mode='constant'
    elif padding_mode == 'replicate':
        mode='edge'
    elif padding_mode == 'reflection':
        mode='symmetric'
    else:
        raise ValueError(f'Unknown padding mode {padding_mode}')
    
    padded = np.pad(img, ((padding,padding), (padding,padding)), mode = mode)
    return padded

def compute_grads(img, padding=2):
    """
    Tính gradient bậc 1 và 2 theo 2 chiều x, y
    Args:
        img: ảnh theo channel Y (đã chuyển sang YIQ và đã được padding)
        padding_mod: 'zero' hoặc 'replicate'
    Returns:
        grad_x: gradient theo chiều x
        grad_y: gradient theo chiều y
        grad2_x: gradient bậc 2 theo chiều x
        grad2_y: gradient bậc 2 theo chiều y
    """
    gx = img[:, 3:-1] - img[:, 1:-3]
    gy = img[3:-1, :] - img[1:-3, :]
    g2x = img[:, 4:] - 2 * img[:, 2:-2] + img[:, :-4]
    g2y = img[4:, :] - 2 * img[2:-2, :] + img[:-4, :]

    gx = gx[2:-2, :]
    gy = gy[:, 2:-2]

    g2x = g2x[2:-2, :]
    g2y = g2y[:, 2:-2]
    H, W = gx.shape
    feat = np.zeros((H, W, 4), dtype=np.float32)
    feat[:, :, 0] = gx
    feat[:, :, 1] = gy
    feat[:, :, 2] = g2x
    feat[:, :, 3] = g2y
    return feat # list các features của ảnh


def build_knn_idx(LR_features_patches, k=5):
    """
    Xây dựng chỉ mục KNN từ các patch feature LR
    Args:
        LR_feature_patches: mảng numpy các patch feature LR
    Returns:
        knn_model: mô hình KNN đã được huấn luyện
    """
    N, h, w, _ = LR_features_patches.shape

    feat_vectors = LR_features_patches.reshape(N, -1)  # Chuyển mỗi patch thành vector 1D

    # fit knn
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
    knn.fit(feat_vectors)

    return knn, feat_vectors

def query_knn(knn_model, feat_vectors, query_feat, k=5):
    """
    Truy vấn KNN để tìm các patch tương tự nhất
    Args:
        knn_model: mô hình KNN đã được huấn luyện
        feat_vectors: mảng numpy các vector feature LR đã được reshape thành 1D (N, h*w*4)
        query_feat: vector feature của patch cần truy vấn
    Returns:
        distances: khoảng cách đến các patch gần nhất
        indices: chỉ số của các patch gần nhất trong tập dữ liệu gốc
    """
    query_feat = query_feat.reshape(1, -1)  # Đảm bảo query_feat là vector 1D
    distances, indices = knn_model.kneighbors(query_feat, n_neighbors=k)
    return distances[0], indices[0]

def compute_weights(target_vec, neighbor_vecs, epsilon=1e-6):
    """
    Tính trọng số cho các patch lân cận dựa trên khoảng cách
    Args:
        target_vec: vector feature của patch mục tiêu (1D)
        neighbor_vecs: mảng numpy các vector feature của các patch lân cận (k, h*w*4)
        epsilon: hằng số nhỏ để tránh chia cho 0
    Returns:
        weights: mảng numpy các trọng số tương ứng với các patch lân cận (k,)
    """
    k, D = neighbor_vecs.shape
    X = neighbor_vecs.T  # (D, k)
    x_q = target_vec.reshape(-1, 1)  # (D, 1)
    diff = np.dot(x_q, np.ones((1, k))) - X  # (D, k)
    G_q = np.dot(diff.T, diff) # (k, k)
    G_q += epsilon * np.eye(k)  # Thêm epsilon vào đường chéo chính để tránh chia cho 0
    ones = np.ones((k, 1))
    weights = (np.linalg.inv(G_q) @ ones) / (ones.T@np.linalg.inv(G_q)@ones)
    return weights
    
def map_hr_patches(neighbor_hr_patches, weights, lr_y_mean):
    """
    Ánh xạ các patch HR lân cận thành patch HR mục tiêu
    Args:
        neighbor_hr_patches: mảng numpy các patch HR lân cận (k, h*scale, w*scale)
        weights: mảng numpy các trọng số tương ứng với các patch lân cận (k,)
        lr_y_mean: giá trị mean của patch LR mục tiêu
    Returns:
        hr_patch: patch HR mục tiêu (h*scale, w*scale)
    """
    predicted_hr = np.tensordot(weights[:, 0], neighbor_hr_patches, axes=(0,0))  # (h*scale, w*scale)
    predicted_hr += lr_y_mean  # Thêm mean của patch LR mục tiêu
    return predicted_hr

def extract_patches_per_image(lr_img_path, hr_img_path=None, patch_size=5, scale=2, overlap=0.5):
    """
    Chia một ảnh LR (và HR nếu có) thành patch.
    Args:
        lr_img_path: đường dẫn ảnh LR
        hr_img_path: đường dẫn ảnh HR (nếu có)
        patch_size: kích thước patch LR
        scale: scale factor giữa HR và LR
        overlap: tỉ lệ chồng lấn
    Returns:
        LR_patches, LR_feature_patches, LR_y_means, HR_centered_patches
    """
    LR_patches, LR_features_patches, LR_y_means = [], [], []

    lr_img = cv2.imread(lr_img_path, cv2.IMREAD_COLOR)
    lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    lr_img = rgb2yiq(lr_img)
    lr_y_img = lr_img[:, :, 0]

    lr_y_img_pad = img_padding(lr_y_img, padding=2, padding_mode='replicate')
    lr_features = compute_grads(lr_y_img_pad)

    step = max(1, int(patch_size*(1-overlap)))
    H_lr, W_lr = lr_img.shape[:2]

    for i in range(0, H_lr - patch_size + 1, step):
        for j in range(0, W_lr - patch_size + 1, step):
            lr_patch = lr_img[i:i+patch_size, j:j+patch_size, :]
            feat_patch = lr_features[i:i+patch_size, j:j+patch_size, :]
            lr_patch_y = lr_patch[:, :, 0]
            lr_mean = float(lr_patch_y.mean())

            LR_patches.append(lr_patch)
            LR_features_patches.append(feat_patch)
            LR_y_means.append(lr_mean)

    return np.array(LR_patches), np.array(LR_features_patches), np.array(LR_y_means) 

def reconstruct_hr_rgb(predicted_y, lr_img_path, scale=2):
    """
    Kết hợp kênh Y đã dự đoán với kênh IQ từ ảnh LR để tái tạo ảnh RGB HR
    Args:
        predicted_y: kênh Y đã dự đoán (H*scale, W*scale)
        lr_yiq: ảnh LR theo kênh YIQ (H, W, 3)
        scale: scale factor giữa HR và LR
    Returns:
        hr_rgb: ảnh RGB HR tái tạo (H*scale, W*scale, 3)
    """
    lr_rgb = cv2.imread(lr_img_path, cv2.IMREAD_COLOR)
    lr_rgb = cv2.cvtColor(lr_rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lr_yiq = rgb2yiq(lr_rgb)
    lr_I = lr_yiq[:, :, 1]
    lr_Q = lr_yiq[:, :, 2]
    H_hr, W_hr = predicted_y.shape
    I_hr = cv2.resize(lr_I, (W_hr, H_hr), interpolation=cv2.INTER_CUBIC)
    Q_hr = cv2.resize(lr_Q, (W_hr, H_hr), interpolation=cv2.INTER_CUBIC)

    HR_YIQ = np.stack([predicted_y, I_hr, Q_hr], axis=-1)

    # YIQ -> RGB
    predicted_HR_rgb = yiq2rgb(HR_YIQ)
    predicted_HR_rgb = np.clip(predicted_HR_rgb, 0, 1)
    predicted_HR_rgb_uint8 = (predicted_HR_rgb*255).astype(np.uint8)
    return predicted_HR_rgb_uint8