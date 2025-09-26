import numpy as np
import cv2
from PIL import Image

from .degradation import degrade_image
from src import sr_interpolation   # same folder as demo.py
import gradio as gr

# Convert PIL → np.float32 [0,1]
def pil_to_np(img: Image.Image):
    arr = np.array(img.convert("L")) / 255.0   # grayscale
    return arr

# Convert np.float32 [0,1] → PIL
def np_to_pil(arr: np.ndarray):
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def process_pipeline(hr_img, scale, noise_type, noise_std, interp_method):
    # Convert input
    hr = pil_to_np(hr_img)

    # Degrade HR → LR
    lr = degrade_image(hr, scale=scale, noise_type=noise_type, noise_std=noise_std)

    # Interpolate LR → SR
    sr = sr_interpolation(lr, scale=scale, method=interp_method)

    # Convert all back to PIL
    return np_to_pil(hr), np_to_pil(lr), np_to_pil(sr)

# Gradio UI
def run_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("## 🔬 Image Super-Resolution Demo (Traditional)")

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
                interp_method = gr.Dropdown(
                    ["nearest", "bilinear", "bicubic", "lanczos"],
                    value="bicubic",
                    label="Interpolation Method",
                )
                btn = gr.Button("Run SR")
            with gr.Column():
                out_hr = gr.Image(label="Original HR")
                out_lr = gr.Image(label="Degraded LR")
                out_sr = gr.Image(label="Reconstructed SR")

        btn.click(
            process_pipeline,
            inputs=[inp, scale, noise_type, noise_std, interp_method],
            outputs=[out_hr, out_lr, out_sr],
        )

        demo.launch()
