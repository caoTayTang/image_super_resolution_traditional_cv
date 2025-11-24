from .io_utils import pil_to_np, np_to_pil
from .degradation import degrade_image
from .interpolation import sr_interpolation


def run_pipeline(hr_img, steps, params):
    """
    Generic image-processing pipeline.
    Args:
        hr_img (PIL.Image)
        steps (list[str]): ordered list of steps to run ("degrade", "interpolate", etc.)
        params (dict): parameters for each step
    Returns:
        dict of intermediate results {step_name: PIL.Image}
    """
    results = {"hr": hr_img}

    img_np = pil_to_np(hr_img)

    for step in steps:
        if step == "degrade":
            img_np = degrade_image(
                img_np,
                scale=params.get("scale", 2),
                noise_type=params.get("noise_type", "gaussian"),
                noise_std=params.get("noise_std", 0.01),
            )
            results["lr"] = np_to_pil(img_np)

        elif step == "interpolate":
            interp_method = params.get("interp_method", "bicubic")
            scale = params.get("scale", 2)
            lr_pil = results.get("lr", np_to_pil(img_np))
            sr_pil = sr_interpolation(lr_pil, method=interp_method, scale=scale)
            results["sr"] = sr_pil

        # easy future extension:
        # elif step == "cnn_sr":
        #     img_np = run_cnn_sr(img_np)
        #     results["sr_cnn"] = np_to_pil(img_np)

        else:
            raise ValueError(f"Unknown pipeline step: {step}")

    return results
