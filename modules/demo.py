import cv2
import numpy as np
import gradio as gr
from PIL import Image
from src import sr_interpolation
from src.ibp import iterative_backprojection


# ======================
# Helper Functions
# ======================
def pil_to_np(img: Image.Image):
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
        cv2_interp = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4
        }[interp_method]
        
        sr = cv2.resize(lr, (new_W, new_H), interpolation=cv2_interp)

    elif sr_method == "Iterative Back-projection":
        
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
                    ["Interpolation", "Iterative Back-projection"],
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
                    ibp_alpha = gr.Slider(0.1, 1.0, value=0.2, step=0.1, label="IBP Step Size α")

                btn = gr.Button("Run SR", variant="primary")

            with gr.Column():
                out_lr = gr.Image(label="Input LR (Normalized)")
                out_sr = gr.Image(label="Reconstructed SR")

        # Toggle SR method
        def toggle_sr(method):
            if method == "Iterative Back-projection":
                return gr.update(visible=False), gr.update(visible=True)
            else:
                return gr.update(visible=True), gr.update(visible=False)

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