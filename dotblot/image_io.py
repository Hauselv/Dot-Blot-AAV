from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from .types import ImageInfo

SUPPORTED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def _to_numpy_from_bytes(file_bytes: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Nicht unterstuetztes Dateiformat: {suffix}")

    if suffix in {".tif", ".tiff"}:
        image = tifffile.imread(BytesIO(file_bytes))
    else:
        with Image.open(BytesIO(file_bytes)) as img:
            image = np.array(img)
    return np.asarray(image)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(np.float32, copy=False)

    if image.ndim != 3:
        raise ValueError(f"Unerwartete Bildform: {image.shape}")

    color = image[..., :3].astype(np.float32, copy=False)
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.tensordot(color, weights, axes=([-1], [0])).astype(np.float32, copy=False)


def load_image(file_bytes: bytes, filename: str) -> tuple[np.ndarray, np.ndarray, ImageInfo]:
    raw = _to_numpy_from_bytes(file_bytes, filename)
    gray = to_grayscale(raw)
    image_info = ImageInfo(
        filename=filename,
        shape=tuple(raw.shape),
        dtype=str(raw.dtype),
        intensity_min=float(np.min(raw)),
        intensity_max=float(np.max(raw)),
        is_color=raw.ndim == 3,
        file_hash=compute_file_hash(file_bytes),
    )
    return raw, gray, image_info
