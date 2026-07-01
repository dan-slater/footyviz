"""Crop super-resolution / enhancement for clearer VLM grids and OCR.

Real-ESRGAN (realesrgan+basicsr) would be the GPU-native choice, but its pins
clash hard with torch 2.12 / numpy 2.5 / torchvision 0.27 (basicsr imports the
removed torchvision.transforms.functional_tensor) and the install hangs in
resolution. So this uses a strong CLASSICAL enhancement that needs no extra deps
and runs anywhere: Lanczos x4 upscale + mild bilateral denoise + CLAHE (for the
dark night footage) + unsharp mask. It won't hallucinate detail like a GAN but it
makes numbers/hair/socks meaningfully more legible on these small crops.

Results are cached in-process by content hash so repeated grid builds don't redo
the work.
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np

_CACHE: dict[str, np.ndarray] = {}


def _enhance(img: np.ndarray, scale: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    up = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
    up = cv2.bilateralFilter(up, 5, 45, 45)                     # denoise, keep edges
    lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)  # lift dark detail
    up = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(up, (0, 0), 2.0)
    up = cv2.addWeighted(up, 1.5, blur, -0.5, 0)                # unsharp
    return up


def upscale(img: np.ndarray, scale: int = 4) -> np.ndarray:
    """Enhanced x`scale` crop, cached by content."""
    if img is None or img.size == 0:
        return img
    key = hashlib.md5(np.ascontiguousarray(img)).hexdigest() + f"_{scale}"
    cached = _CACHE.get(key)
    if cached is None:
        cached = _enhance(img, scale)
        _CACHE[key] = cached
    return cached


BACKEND = "lanczos+clahe+unsharp (classical; Real-ESRGAN unavailable)"
