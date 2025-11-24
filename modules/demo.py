import gradio as gr
from .pipeline import run_pipeline


def process(hr_img, scale, noise_type, noise_std, interp_method):
    params = {
        "scale": scale,
        "noise_type": noise_type,
        "noise_std": noise_std,
        "interp_method": interp_method,
    }

    results = run_pipeline(
        hr_img,
        steps=["degrade", "interpolate"],
        params=params
    )

    return results["hr"], results["lr"], results["sr"]


def run_gradio():
    with gr.Blocks(title="SR Pipeline") as demo:
        gr.Markdown("## 🧩 Super-Resolution")

        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Upload HR Image")
                scale = gr.Slider(1, 8, value=2, step=1, label="Scale Factor")
                noise_type = gr.Dropdown(
                    ["none", "gaussian", "rayleigh", "gamma", "exponential", "uniform", "saltpepper"],
                    value="gaussian",
                    label="Noise Type",
                )
                noise_std = gr.Slider(0.0, 0.2, value=0.01, step=0.01, label="Noise Std/Prob")
                interp_method = gr.Dropdown(
                    ["nearest", "bilinear", "bicubic"],
                    value="bicubic",
                    label="Interpolation Method",
                )
                btn = gr.Button("Run Pipeline")

            with gr.Column(scale=2):
                out_hr = gr.Image(label="Original HR")
                out_lr = gr.Image(label="Degraded LR")
                out_sr = gr.Image(label="Reconstructed SR")

        btn.click(
            process,
            inputs=[inp, scale, noise_type, noise_std, interp_method],
            outputs=[out_hr, out_lr, out_sr],
        )

    demo.launch(share=True)
