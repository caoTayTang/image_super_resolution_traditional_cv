import sys, os

# Thêm thư mục project_root vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
# --- THÊM IMPORT MSE ---
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse 

from models import iterative_backprojection
from modules.degradation import degrade_image
from modules.upsample import upsample_bicubic

# =============================================================================
# CẤU HÌNH THAM SỐ (CONFIG)
# =============================================================================
OUTPUT_DIR = "../results/IBP"    # Thư mục để lưu kết quả
SCALE = 2                        # Hệ số phóng đại
ITERATIONS = 20                  # Số vòng lặp IBP
ALPHA = 0.2                      # Tốc độ học
NOISE_STD = 0.01                 # Độ lệch chuẩn nhiễu (Gaussian)

def save_image(path, img_float):
    """Chuyển từ Float RGB [0,1] sang UInt8 BGR [0,255] để lưu bằng OpenCV"""
    img_uint8 = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    if img_uint8.ndim == 3:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_uint8)

def run_evaluation():
    # Tạo thư mục output nếu chưa có
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Đã tạo thư mục: {OUTPUT_DIR}")
    
    IMAGE_PATH = "../data/degraded_lr/baby.png" 

    if not os.path.exists(IMAGE_PATH):
        print(f"Không tìm thấy ảnh HR tại: {IMAGE_PATH}")
        return

    image_files = [IMAGE_PATH]  

    results = []
    print(f"{'='*60}")
    print(f"BẮT ĐẦU ĐÁNH GIÁ (Bao gồm MSE, PSNR, SSIM)")
    print(f"Scale: x{SCALE} | Iterations: {ITERATIONS} | Alpha: {ALPHA}")
    print(f"{'='*60}\n")

    for idx, img_name in enumerate(image_files):
        filename_only = os.path.basename(img_name) 
        base_name = os.path.splitext(filename_only)[0]
        img_path = img_name
        
        print(f"[{idx+1}/{len(image_files)}] Đang xử lý: {img_name}...")
        
        # --- A. PRE-PROCESSING ---
        hr_orig = cv2.imread(img_path)
        if hr_orig is None: continue
        hr_orig = cv2.cvtColor(hr_orig, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        h, w = hr_orig.shape[:2]
        h_new, w_new = h - (h % SCALE), w - (w % SCALE)
        hr = hr_orig[:h_new, :w_new]
        
        # --- B. TẠO LR ---
        lr = degrade_image(hr, SCALE, noise_std=NOISE_STD)
        
        # --- C. BICUBIC ---
        sr_bicubic = upsample_bicubic(lr, SCALE)
        sr_bicubic = np.clip(sr_bicubic, 0.0, 1.0)
        
        # --- D. IBP (OURS) ---
        sr_ibp = np.zeros_like(sr_bicubic)
        if lr.ndim == 3:
            for c in range(3):
                sr_ibp[:,:,c] = iterative_backprojection(
                    lr[:,:,c], scale=SCALE, iterations=ITERATIONS, alpha=ALPHA
                )
        else:
             sr_ibp = iterative_backprojection(
                    lr, scale=SCALE, iterations=ITERATIONS, alpha=ALPHA
                )
            
        # --- E. TÍNH TOÁN CHỈ SỐ (THÊM MSE) ---
        # 1. Bicubic Metrics
        m_bic = mse(hr, sr_bicubic)  # <-- Tính MSE Bicubic
        p_bic = psnr(hr, sr_bicubic, data_range=1.0)
        s_bic = ssim(hr, sr_bicubic, channel_axis=2, data_range=1.0)
        
        # 2. IBP Metrics
        m_ibp = mse(hr, sr_ibp)      # <-- Tính MSE IBP
        p_ibp = psnr(hr, sr_ibp, data_range=1.0)
        s_ibp = ssim(hr, sr_ibp, channel_axis=2, data_range=1.0)
        
        results.append({
            "name": base_name,
            "bic_mse": m_bic, "bic_psnr": p_bic, "bic_ssim": s_bic,
            "ibp_mse": m_ibp, "ibp_psnr": p_ibp, "ibp_ssim": s_ibp
        })
        
        # --- F. LƯU ẢNH ---
        save_image(os.path.join(OUTPUT_DIR, f"{base_name}_HR.png"), hr)
        save_image(os.path.join(OUTPUT_DIR, f"{base_name}_LR.png"), lr)
        save_image(os.path.join(OUTPUT_DIR, f"{base_name}_Bicubic.png"), sr_bicubic)
        save_image(os.path.join(OUTPUT_DIR, f"{base_name}_IBP.png"), sr_ibp)

    # =========================================================================
    # 4. SINH MÃ LATEX
    # =========================================================================
    print(f"\n{'='*60}")
    print("ĐÃ HOÀN TẤT! MÃ LATEX (Kèm MSE):")
    print(f"{'='*60}\n")
    
    latex_code = generate_latex_table(results)
    print(latex_code)
    
    with open(os.path.join(OUTPUT_DIR, "report_table.tex"), "w", encoding="utf-8") as f:
        f.write(latex_code)
    print(f"\n(Đã lưu mã LaTeX vào {os.path.join(OUTPUT_DIR, 'report_table.tex')})")

def generate_latex_table(results):
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    # Cập nhật caption
    lines.append(rf"\caption{{So sánh MSE, PSNR (dB) và SSIM giữa Bicubic và IBP ($Scale={SCALE}\times$). Giá trị tốt nhất được \textbf{{in đậm}}.")
    lines.append(r"}")
    lines.append(r"\label{tab:ibp_comparison}")
    
    # Cập nhật cấu trúc bảng: thêm cột MSE
    # l | ccc | ccc (Image | Bicubic(3) | IBP(3))
    lines.append(r"\setlength{\tabcolsep}{4pt}") # Thu nhỏ khoảng cách cột chút cho vừa
    lines.append(r"\begin{tabular}{l|ccc|ccc}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{Image}} & \multicolumn{3}{c|}{\textbf{Bicubic}} & \multicolumn{3}{c}{\textbf{IBP (Ours)}} \\ \cline{2-7} ")
    # Dòng tiêu đề phụ
    lines.append(r" & MSE $\downarrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ & MSE $\downarrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ \\ \midrule")
    
    # Biến tính trung bình
    avg_data = [0] * 6 # [bic_mse, bic_psnr, bic_ssim, ibp_mse, ibp_psnr, ibp_ssim]
    
    for res in results:
        name = res['name'].capitalize()
        # Dữ liệu: 0:b_mse, 1:b_psnr, 2:b_ssim, 3:i_mse, 4:i_psnr, 5:i_ssim
        d = [
            res['bic_mse'], res['bic_psnr'], res['bic_ssim'],
            res['ibp_mse'], res['ibp_psnr'], res['ibp_ssim']
        ]
        for i in range(6): avg_data[i] += d[i]
        
        # --- LOGIC IN ĐẬM ---
        # MSE: Nhỏ hơn là tốt
        s_bm = f"\\textbf{{{d[0]:.4f}}}" if d[0] < d[3] else f"{d[0]:.4f}"
        s_im = f"\\textbf{{{d[3]:.4f}}}" if d[3] < d[0] else f"{d[3]:.4f}"
        
        # PSNR: Lớn hơn là tốt
        s_bp = f"\\textbf{{{d[1]:.2f}}}" if d[1] > d[4] else f"{d[1]:.2f}"
        s_ip = f"\\textbf{{{d[4]:.2f}}}" if d[4] > d[1] else f"{d[4]:.2f}"
        
        # SSIM: Lớn hơn là tốt
        s_bs = f"\\textbf{{{d[2]:.4f}}}" if d[2] > d[5] else f"{d[2]:.4f}"
        s_is = f"\\textbf{{{d[5]:.4f}}}" if d[5] > d[2] else f"{d[5]:.4f}"
        
        name_tex = name.replace("_", r"\_")
        # Format dòng: Image & MSE & PSNR & SSIM & MSE & PSNR & SSIM
        lines.append(f"{name_tex} & {s_bm} & {s_bp} & {s_bs} & {s_im} & {s_ip} & {s_is} \\\\")
        
    lines.append(r"\midrule")
    
    # Tính trung bình
    n = len(results)
    avg_data = [x/n for x in avg_data]
    d = avg_data
    
    # Logic in đậm cho dòng Average
    s_abm = f"\\textbf{{{d[0]:.4f}}}" if d[0] < d[3] else f"{d[0]:.4f}"
    s_aim = f"\\textbf{{{d[3]:.4f}}}" if d[3] < d[0] else f"{d[3]:.4f}"
    
    s_abp = f"\\textbf{{{d[1]:.2f}}}" if d[1] > d[4] else f"{d[1]:.2f}"
    s_aip = f"\\textbf{{{d[4]:.2f}}}" if d[4] > d[1] else f"{d[4]:.2f}"
    
    s_abs = f"\\textbf{{{d[2]:.4f}}}" if d[2] > d[5] else f"{d[2]:.4f}"
    s_ais = f"\\textbf{{{d[5]:.4f}}}" if d[5] > d[2] else f"{d[5]:.4f}"
    
    lines.append(f"\\textbf{{Average}} & {s_abm} & {s_abp} & {s_abs} & {s_aim} & {s_aip} & {s_ais} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)

if __name__ == "__main__":
    run_evaluation()