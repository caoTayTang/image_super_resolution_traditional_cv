import numpy as np
import cv2
from PIL import Image
import gradio as gr

# Giả sử bạn đã có file này
from src import sr_interpolation
from src.ibp import iterative_backprojection
from src.Wiener_Filter import wiener_unsupervised_mcmc

# Vì mình không có file src của bạn, mình tạo hàm dummy để code chạy được demo
# Bạn nhớ xóa 2 hàm dummy này khi chạy thật nhé
if 'sr_interpolation' not in globals():
    def sr_interpolation(lr, scale, method):
        h, w = lr.shape[:2]
        return cv2.resize(lr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

# ======================
# Helper Functions
# ======================
def pil_to_np(img: Image.Image):
    # SỬA: Không convert("L") nữa. 
    # Nếu là ảnh màu, giữ nguyên RGB. Nếu là ảnh xám, convert sang RGB để xử lý thống nhất hoặc giữ nguyên tùy logic.
    # Ở đây ta convert sang RGB để đảm bảo đầu ra luôn có 3 kênh nếu là ảnh màu.
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    arr = np.array(img) / 255.0
    return arr

def np_to_pil(arr: np.ndarray):
    # Clip giá trị trong khoảng 0-1 rồi nhân 255
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

# ======================
# Main SR Pipeline
# ======================
def process_pipeline(
    lr_img, scale,
    sr_method, interp_method, upsample_method,
    ibp_iter, ibp_alpha
):
    if lr_img is None:
        return None, None

    lr = pil_to_np(lr_img)
    # lr shape có thể là (H, W, 3) cho ảnh màu hoặc (H, W) cho ảnh xám
    
    H = lr.shape[0]
    W = lr.shape[1]
    new_H, new_W = int(H * scale), int(W * scale)

    # ----------------------
    # Super-Resolution
    # ----------------------
    if sr_method == "Interpolation":
        # Các hàm interpolation của OpenCV (dùng trong sr_interpolation giả định) 
        # tự động xử lý được cả ảnh màu và ảnh xám
        
        # Lưu ý: Cần đảm bảo sr_interpolation của bạn dùng cv2.resize hoặc tương tự hỗ trợ đa kênh
        # Nếu hàm sr_interpolation của bạn tự viết tay chỉ hỗ trợ 2D, bạn cũng cần tách kênh như IBP bên dưới
        
        # Mapping method string sang cv2 constant
        cv2_interp = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4
        }[interp_method]
        
        sr = cv2.resize(lr, (new_W, new_H), interpolation=cv2_interp)

    elif sr_method == "Iterative Back-projection":
        # IBP thường được viết cho ma trận 2D. 
        # Để xử lý ảnh màu, ta tách kênh (Split) -> Xử lý -> Gộp kênh (Merge)
        
        if lr.ndim == 3 and lr.shape[2] == 3: # Ảnh màu RGB
            channels = []
            for i in range(3):
                # Lấy từng kênh màu
                c_lr = lr[:, :, i]
                
                # Chạy IBP cho kênh đó
                c_sr = iterative_backprojection(
                    c_lr, scale=scale,
                    iterations=int(ibp_iter),
                    alpha=float(ibp_alpha)
                )
                channels.append(c_sr)
            
            # Gộp 3 kênh lại
            sr = np.stack(channels, axis=2)
            
        else: # Ảnh xám
            sr = iterative_backprojection(
                lr, scale=scale,
                iterations=int(ibp_iter),
                alpha=float(ibp_alpha)
            )


    elif sr_method == "Wiener (unsupervised)":
        upsampled = cv2.resize(lr, (new_W, new_H), interpolation=cv2.INTER_CUBIC)
        upsampled = upsampled.astype(np.float32)
        
        if upsampled.ndim == 3 and upsampled.shape[2] == 3:
            ycrcb = cv2.cvtColor(upsampled, cv2.COLOR_RGB2YCrCb)
            ycrcb = ycrcb.astype(np.float32)
            Y, Cr, Cb = cv2.split(ycrcb)
            Y_restored = wiener_unsupervised_mcmc(Y)
            Y_restored = np.clip(Y_restored, 0, 1)
            Y_restored = Y_restored.astype(np.float32)
            sr = cv2.cvtColor(cv2.merge([Y_restored, Cr, Cb]), cv2.COLOR_YCrCb2RGB)
        else:
            Y_restored = wiener_unsupervised_mcmc(upsampled)
            sr = np.clip(Y_restored, 0, 1)
 
    sr_pil = np_to_pil(sr)
    lr_pil = np_to_pil(lr)

    return lr_pil, sr_pil

# ======================
# Gradio UI
# ======================
def run_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("## 🧠 Single-Image Super-Resolution (Color Supported)")

        with gr.Row():
            with gr.Column():
                # Input LR image
                inp = gr.Image(type="pil", label="Upload LR Image")

                # Scale factor
                scale = gr.Slider(2, 8, value=4, step=1, label="Upscale Factor")

                # SR method
                sr_method = gr.Dropdown(
                    ["Interpolation", "Iterative Back-projection", "Wiener (unsupervised)"],
                    value="Interpolation",
                    label="Super-Resolution Method"
                )

                # Interpolation dropdown
                interp_method = gr.Dropdown(
                    ["nearest", "bilinear", "bicubic", "lanczos"],
                    value="bicubic",
                    label="Interpolation Method",
                    visible=True
                )

                # IBP options
                with gr.Group(visible=False) as ibp_group:
                    upsample_method = gr.Dropdown(
                        ["nearest", "bilinear", "bicubic"],
                        value="bicubic",
                        label="Upsample Method (IBP Post-process)"
                    )
                    ibp_iter = gr.Slider(10, 200, value=20, step=1, label="IBP Iterations")
                    ibp_alpha = gr.Slider(0.1, 2.0, value=0.5, step=0.1, label="IBP Step Size α")

                btn = gr.Button("Run SR", variant="primary")

            with gr.Column():
                out_lr = gr.Image(label="Input LR (Normalized)")
                out_sr = gr.Image(label="Reconstructed SR")

        # Toggle SR method
        def toggle_sr(method):
            if method == "Iterative Back-projection":
                return gr.update(visible=False), gr.update(visible=True)
            elif method == "Interpolation":
                return gr.update(visible=True), gr.update(visible=False)
            else:  # Wiener
                return gr.update(visible=False), gr.update(visible=False)

        sr_method.change(
            toggle_sr,
            inputs=[sr_method],
            outputs=[interp_method, ibp_group]
        )

        # Run button
        btn.click(
            process_pipeline,
            inputs=[inp, scale, sr_method, interp_method, upsample_method, ibp_iter, ibp_alpha],
            outputs=[out_lr, out_sr]
        )

    demo.launch()

if __name__ == "__main__":
    run_gradio()