# import numpy as np
# import cv2
# from PIL import Image

# from .degradation.degradation import degrade_image
# from src import sr_interpolation   # same folder as demo.py
# import gradio as gr

# # Convert PIL → np.float32 [0,1]
# def pil_to_np(img: Image.Image):
#     arr = np.array(img.convert("L")) / 255.0   # grayscale
#     return arr

# # Convert np.float32 [0,1] → PIL
# def np_to_pil(arr: np.ndarray):
#     arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
#     return Image.fromarray(arr)

# def process_pipeline(hr_img, scale, noise_type, noise_std, interp_method):
#     # Convert input
#     hr = pil_to_np(hr_img)

#     # Degrade HR → LR
#     lr = degrade_image(hr, scale=scale, noise_type=noise_type, noise_std=noise_std)

#     # Interpolate LR → SR
#     sr = sr_interpolation(lr, scale=scale, method=interp_method)

#     # Convert all back to PIL
#     return np_to_pil(hr), np_to_pil(lr), np_to_pil(sr)

# # Gradio UI
# def run_gradio():
#     with gr.Blocks() as demo:
#         gr.Markdown("## 🔬 Image Super-Resolution Demo (Traditional)")

#         with gr.Row():
#             with gr.Column():
#                 inp = gr.Image(type="pil", label="Upload HR Image")
#                 scale = gr.Slider(2, 8, value=4, step=1, label="Downscale Factor")
#                 noise_type = gr.Dropdown(
#                     ["none", "gaussian", "rayleigh", "gamma", "exponential", "uniform", "saltpepper"],
#                     value="gaussian",
#                     label="Noise Type",
#                 )
#                 noise_std = gr.Slider(0.0, 0.2, value=0.01, step=0.01, label="Noise Std/Prob")
#                 interp_method = gr.Dropdown(
#                     ["nearest", "bilinear", "bicubic", "lanczos"],
#                     value="bicubic",
#                     label="Interpolation Method",
#                 )
#                 btn = gr.Button("Run SR")
#             with gr.Column():
#                 out_hr = gr.Image(label="Original HR")
#                 out_lr = gr.Image(label="Degraded LR")
#                 out_sr = gr.Image(label="Reconstructed SR")

#         btn.click(
#             process_pipeline,
#             inputs=[inp, scale, noise_type, noise_std, interp_method],
#             outputs=[out_hr, out_lr, out_sr],
#         )

#         demo.launch()

import numpy as np
import cv2
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
import io


from .degradation.degrade import degrade_image
from src import sr_interpolation
from src.models import iterative_backprojection 
from src.metrics import psnr

# ======================
# Helper Functions
# ======================
def pil_to_np(img: Image.Image):
    arr = np.array(img.convert("L")) / 255.0   # grayscale [0,1]
    return arr

def np_to_pil(arr: np.ndarray):
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def plot_mse_curve(mse_list):
    fig, ax = plt.subplots()
    ax.plot(mse_list, '-o', color='orange')
    ax.set_title("MSE Convergence Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Squared Error")
    ax.grid(True)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)  # ✅ chuyển buffer thành ảnh PIL
    plt.close(fig)
    return img

# ======================
# Main Pipeline
# ======================
def process_pipeline(hr_img, scale, noise_type, noise_std,
                     sr_method, interp_method, upsample_method,
                     ibp_iter, ibp_alpha, ibp_denoise, ibp_adaptive,
                     ibp_dynamic, ibp_early):
    hr = pil_to_np(hr_img)
    lr = degrade_image(hr, scale=scale, noise_type=noise_type, noise_std=noise_std)

    mse_list = None
    if sr_method == "Interpolation":
        sr = sr_interpolation(lr, scale=scale, method=interp_method)
        
        sr_pil_temp = np_to_pil(sr)
        sr_pil_temp = sr_pil_temp.resize((hr.shape[1], hr.shape[0]), resample={
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC
        }[interp_method])
        sr = pil_to_np(sr_pil_temp)
        
    elif sr_method == "IBP":
        sr, mse_list = iterative_backprojection(
            lr,
            scale=scale,
            iterations=int(ibp_iter),
            alpha=float(ibp_alpha),
            denoise=ibp_denoise,
            adaptive_alpha=ibp_adaptive,
            dynamic_blur = ibp_dynamic,
            early_stop = ibp_early,
            return_mse=True
        )
        sr_pil_temp = np_to_pil(sr)
        sr_pil_temp = sr_pil_temp.resize((hr.shape[1], hr.shape[0]), resample={
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC
        }[upsample_method])
        sr = pil_to_np(sr_pil_temp)
        
    else:
        sr = lr
    
    # Tính PSNR
    psnr_val = psnr(hr, sr)

    hr_pil, lr_pil, sr_pil = np_to_pil(hr), np_to_pil(lr), np_to_pil(sr)
    mse_plot = plot_mse_curve(mse_list) if mse_list is not None else None

    return hr_pil, lr_pil, sr_pil, mse_plot, f"PSNR: {psnr_val:.2f} dB"

# ======================
# Gradio UI
# ======================
def run_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("## 🧠 Image Super-Resolution Demo")

        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Upload HR Image")
                scale = gr.Slider(2, 8, value=4, step=1, label="Downscale Factor")
                noise_type = gr.Dropdown(
                    ["none", "gaussian", "rayleigh", "gamma", "exponential", "uniform", "saltpepper"],
                    value="gaussian",
                    label="Noise Type",
                )
                noise_std = gr.Slider(0.0, 0.2, value=0.01, step=0.01, label="Noise Std/Prob")

                sr_method = gr.Dropdown(
                    ["Interpolation", "Iterative Back-projection"],
                    value="Interpolation",
                    label="Super-Resolution Method",
                )

                # --- Parameters for Interpolation ---
                interp_method = gr.Dropdown(
                    ["nearest", "bilinear", "bicubic", "lanczos"],
                    value="bicubic",
                    label="Interpolation Method",
                    visible=True,
                )

                # --- Parameters for IBP ---
                with gr.Group(visible=False) as ibp_group:
                    upsample_method = gr.Dropdown(
                        ["nearest", "bilinear", "bicubic"],
                        value="bicubic",
                        label="Upsample Method (IBP only)"
                    )
                    ibp_iter = gr.Slider(20, 200, value=20, step=1, label="IBP Iterations")
                    ibp_alpha = gr.Slider(0.1, 1.0, value=0.1, step=0.1, label="IBP Step Size (α)")
                    ibp_denoise = gr.Checkbox(label="TV Denoising", value=False)
                    ibp_adaptive = gr.Checkbox(label="Adaptive α", value=False)
                    ibp_dynamic = gr.Checkbox(label="Dynamic Blur (σ decay)", value=False)
                    ibp_early = gr.Checkbox(label="Early Stop", value=False)

                btn = gr.Button("Run SR")

            with gr.Column():
                out_hr = gr.Image(label="Original HR")
                out_lr = gr.Image(label="Degraded LR")
                out_sr = gr.Image(label="Reconstructed SR")
                out_mse = gr.Image(label="MSE Convergence", visible=False)
                out_psnr = gr.Textbox(label="PSNR (HR vs SR)")

        # --- Toggle visibility ---
        def toggle_params(method):
            if method == "IBP":
                return gr.update(visible=False), gr.update(visible=True)
            else:
                return gr.update(visible=True), gr.update(visible=False)

        sr_method.change(toggle_params, inputs=[sr_method], outputs=[interp_method, ibp_group])

        # --- Run button ---
        btn.click(
            process_pipeline,
            inputs=[
                inp, scale, noise_type, noise_std,
                sr_method, interp_method, upsample_method,
                ibp_iter, ibp_alpha, ibp_denoise, ibp_adaptive, ibp_dynamic, ibp_early
            ],
            outputs=[out_hr, out_lr, out_sr, out_mse, out_psnr],
        )

        demo.launch()
