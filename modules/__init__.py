from .degradation.degrade import degrade_image

# Tránh import các module nặng ở thời điểm import package để không gây circular import.
def get_run_gradio():
    """Lazy import: trả về hàm run_gradio khi cần."""
    from .demo import run_gradio
    return run_gradio

__all__ = ["get_run_gradio"]