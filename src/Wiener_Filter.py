# Wiener_Filter.py
"""
Traditional restoration pipeline (Wiener + RL + TV + optional pre/post denoise)

Dependencies:
    numpy, scipy, scikit-image (skimage), opencv-python
Install (if needed):
    pip install numpy scipy scikit-image opencv-python
"""

import os
import numpy as np
import cv2
from scipy.signal import wiener as adaptive_wiener
from numpy.fft import fft2, ifft2, ifftshift
from skimage import img_as_float, color
from skimage.restoration import (
    unsupervised_wiener,
    richardson_lucy,
    denoise_tv_chambolle,
    estimate_sigma as skimage_estimate_sigma,
)
from skimage.util import img_as_ubyte


def _to_luminance(img):
    """Return (luma, is_color) where luma is 2D float image [0..1]."""
    img_f = img_as_float(img)
    if img_f.ndim == 3 and img_f.shape[2] == 3:
        return color.rgb2ycbcr(img_f)[..., 0], True
    elif img_f.ndim == 3 and img_f.shape[2] == 4:
        img_rgb = img_f[..., :3]
        return color.rgb2ycbcr(img_rgb)[..., 0], True
    else:
        return img_f, False


def _from_luminance(luma, original_img):
    """If original_img was color, replace Y channel and convert back to RGB; else return luma."""
    if original_img.ndim == 3 and original_img.shape[2] >= 3:
        img_f = img_as_float(original_img)
        ycbcr = color.rgb2ycbcr(img_f)
        ycbcr[..., 0] = np.clip(luma, 0, 1)
        rgb = color.ycbcr2rgb(ycbcr)
        if original_img.shape[2] == 4:
            alpha = img_f[..., 3]
            rgba = np.dstack([rgb, alpha])
            return np.clip(rgba, 0, 1)
        return np.clip(rgb, 0, 1)
    else:
        return np.clip(luma, 0, 1)


def estimate_noise_variance(image, method='robust'):
    """
    Estimate noise variance from image (robust Laplacian-based).
    Returns float variance.
    """
    if method == 'robust':
        if image.ndim == 2:
            lap = cv2.Laplacian((image * 255).astype(np.float64), cv2.CV_64F)
            sigma = np.sqrt(2) * np.median(np.abs(lap - np.median(lap))) / 0.6745
        else:
            sigma = 0.0
            for c in range(image.shape[2]):
                lap = cv2.Laplacian((image[..., c] * 255).astype(np.float64), cv2.CV_64F)
                sigma += np.sqrt(2) * np.median(np.abs(lap - np.median(lap))) / 0.6745
            sigma /= image.shape[2]
    else:
        sigma = np.std(image)
    variance = max(sigma**2, 1e-12)
    return variance


def _match_shape_to_original(restored, original_img):
    """
    Ensure restored has exactly the same shape as original_img.
    - resized using cv2.resize if necessary
    - preserves number of channels where possible, re-attaches alpha if needed
    - input/outputs are floats in [0,1]
    """
    orig_shape = original_img.shape
    if restored.shape == orig_shape:
        return restored

    # Target size (width, height) for cv2.resize
    target_h, target_w = orig_shape[0], orig_shape[1]

    # If restored is grayscale (2D) but original is color, expand channels after resize
    if restored.ndim == 2:
        resized = cv2.resize(restored.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        if len(orig_shape) == 3:
            # expand to same number of channels as original
            channels = orig_shape[2]
            if channels == 1:
                resized = resized[..., np.newaxis]
            else:
                resized = np.stack([resized] * channels, axis=-1)
    else:
        # restored has channels
        resized = cv2.resize(restored.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        # handle channel count mismatch (e.g., restored=3ch but orig=4ch)
        if len(orig_shape) == 2 and resized.ndim == 3:
            # original grayscale but restored color -> convert to gray
            resized = color.rgb2gray(resized)
        elif len(orig_shape) == 3:
            orig_ch = orig_shape[2]
            res_ch = resized.shape[2]
            if res_ch != orig_ch:
                # if original has alpha and restored doesn't, re-attach original alpha
                if orig_ch == 4 and res_ch == 3:
                    alpha = img_as_float(original_img)[..., 3]
                    resized = np.dstack([resized, alpha])
                elif orig_ch == 3 and res_ch == 4:
                    # drop alpha
                    resized = resized[..., :3]
                else:
                    # generic fallback: either repeat or truncate channels
                    if res_ch < orig_ch:
                        resized = np.concatenate([resized] + [resized[..., -1:]] * (orig_ch - res_ch), axis=2)
                    else:
                        resized = resized[..., :orig_ch]

    resized = np.clip(resized, 0, 1)
    return resized


def restore_image(degraded_lr, scale=4, kernel_size=9, sigma=1.6, noise_var=None):
    """
    Simple restore image using upsampling + light denoising (for quick tests).
    Ensures output shape matches input upsampled shape (but test.py expects restored vs original:
    so caller should compare original <-> restored after resizing original to match if needed).
    """
    degraded = img_as_float(degraded_lr)
    try:
        # create simple gaussian PSF (for reference; not used in this simple pipeline)
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        psf = psf / (psf.sum() + 1e-12)

        # Upsample with bicubic
        if degraded.ndim == 2:
            h, w = degraded.shape
            upsampled = cv2.resize(degraded, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            # simple smoothing
            restored = cv2.GaussianBlur(upsampled, (3, 3), 0.5)
            restored = np.clip(restored, 0, 1)
        else:
            h, w, c = degraded.shape
            upsampled = cv2.resize(degraded, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            restored = np.zeros_like(upsampled)
            for ch in range(c):
                restored[..., ch] = cv2.GaussianBlur(upsampled[..., ch], (3, 3), 0.5)
            restored = np.clip(restored, 0, 1)

        # This function returns the upsampled restored image (shape determined by degraded and scale)
        # If you want to guarantee shape equals a specific original (before degradation), caller should pass that original.
        return restored

    except Exception as e:
        # fallback: return upscaled without filtering
        if degraded.ndim == 2:
            h, w = degraded.shape
            return cv2.resize(degraded, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        else:
            h, w, c = degraded.shape
            return cv2.resize(degraded, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def restore_image_advanced(img,
                           psf=None,
                           pre_denoise='nl_means',
                           nl_patch_size=7,
                           nl_patch_distance=11,
                           nl_h_factor=0.8,
                           use_unsupervised_wiener=True,
                           wiener_K=None,
                           rl_iters=10,
                           tv_weight=0.01,
                           final_adaptive_wiener=False,
                           return_psf=False):
    """
    Advanced restore function: pre-denoise -> (unsupervised) wiener -> optional RL -> TV -> final Wiener
    Returns restored image (and psf if return_psf=True).
    This version GUARANTEES the returned restored has the same shape as the input img.
    """
    img_in = img
    luma, was_color = _to_luminance(img_in)

    # estimate noise using skimage helper
    try:
        sigma_est = skimage_estimate_sigma(luma, multichannel=False)
    except Exception:
        sigma_est = np.std(luma)
    sigma2 = float(max(sigma_est**2, 1e-12))

    # pre-denoise
    luma_proc = luma.copy()
    if pre_denoise == 'adaptive':
        luma_proc = adaptive_wiener(luma_proc, mysize=(5, 5))
    elif pre_denoise == 'nl_means':
        try:
            from skimage.restoration import denoise_nl_means
            sigma_est_local = max(sigma_est, 1e-6)
            patch_kw = dict(patch_size=nl_patch_size,
                            patch_distance=nl_patch_distance,
                            multichannel=False)
            h = nl_h_factor * sigma_est_local
            luma_proc = denoise_nl_means(luma_proc, h=h, fast_mode=True, **patch_kw)
        except Exception:
            luma_proc = adaptive_wiener(luma_proc, mysize=(5, 5))

    # deblurring step
    psf_used = None
    if psf is None and use_unsupervised_wiener:
        init_psf = np.zeros((5, 5), dtype=float)
        init_psf[2, 2] = 1.0
        try:
            deconv, psf_est = unsupervised_wiener(luma_proc, init_psf)
            psf_used = psf_est
            luma_deblur = np.clip(deconv, 0, 1)
        except Exception:
            luma_deblur = luma_proc.copy()
    else:
        if psf is None:
            luma_deblur = luma_proc.copy()
        else:
            psf_arr = np.asarray(psf, dtype=float)
            psf_arr = psf_arr / (psf_arr.sum() + 1e-12)
            psf_used = psf_arr.copy()

            # Important: use s=img_shape everywhere so FFT sizes are consistent
            img_shape = luma_proc.shape
            H = fft2(ifftshift(psf_arr), s=img_shape)
            G = fft2(luma_proc, s=img_shape)
            H_conj = np.conj(H)
            H_abs2 = np.abs(H) ** 2

            if wiener_K is None:
                signal_power = max(np.var(luma_proc) - sigma2, 1e-12)
                K = sigma2 / signal_power if signal_power > 0 else 1e-3
            else:
                K = float(wiener_K)

            W = H_conj / (H_abs2 + K + 1e-12)
            F_hat = W * G
            luma_deblur = np.clip(np.real(ifft2(F_hat, s=img_shape)), 0, 1)

    # Richardson-Lucy refinement
    if rl_iters and rl_iters > 0 and psf_used is not None:
        try:
            psf_norm = psf_used / (psf_used.sum() + 1e-12)
            luma_deblur = richardson_lucy(luma_deblur, psf_norm, iterations=rl_iters, clip=True)
        except Exception:
            pass

    # TV denoise to reduce ringing
    if tv_weight and tv_weight > 0:
        try:
            luma_deblur = denoise_tv_chambolle(luma_deblur, weight=tv_weight, multichannel=False)
        except Exception:
            pass

    # final adaptive wiener
    if final_adaptive_wiener:
        try:
            luma_deblur = adaptive_wiener(luma_deblur, mysize=(3, 3))
        except Exception:
            pass

    # Convert back to color (or keep grayscale)
    restored = _from_luminance(luma_deblur, img_in)

    # === NEW: ensure shape exactly matches input img ===
    restored = _match_shape_to_original(restored, img_in)

    if return_psf:
        return restored, psf_used
    return restored
