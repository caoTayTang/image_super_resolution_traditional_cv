import cv2

def load_hr_image(path, scale=4):
    # 1. Đọc ảnh màu (Mặc định OpenCV là BGR)
    hr = cv2.imread(path)
    
    # 2. Chuyển từ BGR sang RGB để hiển thị đúng màu
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
    
    # 3. Chuẩn hóa về float [0, 1]
    hr = hr.astype("float32") / 255.0
    
    # 4. Cắt ảnh cho chẵn với scale (Cropping)
    H, W = hr.shape[:2] # Lấy H, W (bỏ qua kênh màu nếu có)
    Hc, Wc = H - (H % scale), W - (W % scale)
    
    # Trả về ảnh đã crop
    # Slice [:Hc, :Wc] sẽ tự động lấy hết các kênh màu ở chiều thứ 3
    return hr[:Hc, :Wc]
