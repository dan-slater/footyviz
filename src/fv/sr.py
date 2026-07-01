"""Crop super-resolution for clearer VLM grids and jersey OCR.

Uses Real-ESRGAN_x4plus loaded via **spandrel** (no basicsr — basicsr's pins are
incompatible with torch 2.12 / torchvision 0.27 and its install hangs). Runs on
CUDA. Falls back to a classical Lanczos+CLAHE+unsharp pipeline only if the model
or weights are unavailable. Results cached in-process by content hash.

Weights: data/models/RealESRGAN_x4plus.pth (fetched once from the Real-ESRGAN
release). `BACKEND` reports what actually ran.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

WEIGHTS = Path("data/models/RealESRGAN_x4plus.pth")
_CACHE: dict[str, np.ndarray] = {}
_MODEL = None            # None=untried, False=unavailable, else the spandrel model
BACKEND = "uninitialised"


def _load_model():
    global _MODEL, BACKEND
    if _MODEL is not None:
        return _MODEL
    try:
        import torch
        from spandrel import ModelLoader
        if not WEIGHTS.exists():
            raise FileNotFoundError(WEIGHTS)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = ModelLoader().load_from_file(str(WEIGHTS)).to(dev).eval()
        _MODEL._fv_device = dev
        BACKEND = f"RealESRGAN_x4plus (spandrel, {dev})"
    except Exception as e:  # keep the pipeline alive on any SR failure
        _MODEL = False
        BACKEND = f"classical lanczos+clahe+unsharp (Real-ESRGAN unavailable: {repr(e)[:70]})"
    return _MODEL


def _classical(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    up = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
    up = cv2.bilateralFilter(up, 5, 45, 45)
    lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    up = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(up, (0, 0), 2.0)
    return cv2.addWeighted(up, 1.5, blur, -0.5, 0)


def _real_esrgan(img: np.ndarray) -> np.ndarray:
    import torch
    m = _load_model()
    x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
    x = x.float().div(255).unsqueeze(0).to(m._fv_device)
    with torch.no_grad():
        y = m(x)
    out = (y.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def upscale(img: np.ndarray, scale: int = 4) -> np.ndarray:
    """x`scale` super-resolved crop (Real-ESRGAN x4, else classical), cached."""
    if img is None or img.size == 0:
        return img
    key = hashlib.md5(np.ascontiguousarray(img)).hexdigest() + f"_{scale}"
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    m = _load_model()
    if m:
        try:
            out = _real_esrgan(img)                 # native x4
            if scale != 4:
                h, w = img.shape[:2]
                out = cv2.resize(out, (w * scale, h * scale), interpolation=cv2.INTER_AREA)
        except Exception:
            out = _classical(img, scale)
    else:
        out = _classical(img, scale)
    _CACHE[key] = out
    return out
