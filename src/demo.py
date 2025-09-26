import gradio as gr
from PIL import Image
from src.interpolation import sr_interpolation

def degrade_image(img: Image.Image) -> Image.Image:
    """
    TODO: Implement degradation (blur + downsample + noise).
    For now, just return the same image.
    """
    return img

def sr_backprojection(img: Image.Image) -> Image.Image:
    """
    TODO: Implement iterative back-projection / Wiener filter.
    For now, just return the same image.
    """
    return img

def sr_patchbased(img: Image.Image) -> Image.Image:
    """
    TODO: Implement patch/example-based super-resolution.
    For now, just return the same image.
    """
    return img


# ======================
# SR Wrapper
# ======================
def run_sr(img: Image.Image, method: str) -> Image.Image:
    if img is None:
        return None
    if method == "Interpolation-Based":
        return sr_interpolation(img)
    elif method == "Reconstruction-Based":
        return sr_backprojection(img)
    elif method == "Patch-Based":
        return sr_patchbased(img)
    else:
        return img

# ======================
# Gradio UI
# ======================

def run_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("## 🖼️ Image Super-Resolution Demo (Traditional Methods)")

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Upload Image")
                degrade_btn = gr.Button("Degrade Image")
                degraded_img = gr.Image(type="pil", label="Degraded Image")

                method = gr.Dropdown(
                    ["Interpolation-Based", "Reconstruction-Based", "Patch-Based"],
                    value="Interpolation-Based",
                    label="Choose SR Method"
                )
                sr_btn = gr.Button("Run Super-Resolution")

            with gr.Column():
                output_img = gr.Image(type="pil", label="SR Output")

        # Events
        degrade_btn.click(fn=degrade_image, inputs=input_img, outputs=degraded_img)
        sr_btn.click(fn=run_sr, inputs=[degraded_img, method], outputs=output_img)

        demo.launch()