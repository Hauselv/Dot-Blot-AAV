from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def normalize_for_display(image: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.5) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo = float(np.percentile(image, lower_pct))
    hi = float(np.percentile(image, upper_pct))
    if hi <= lo:
        hi = lo + 1e-6
    scaled = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return scaled


def invert_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return float(np.min(image) + np.max(image)) - image


def _positive_blob_score(image: np.ndarray, sigma_small: float = 1.0, sigma_large: float = 7.0) -> float:
    smoothed = gaussian_filter(np.asarray(image, dtype=np.float32), sigma=sigma_small)
    background = gaussian_filter(smoothed, sigma=sigma_large)
    residual = smoothed - background
    threshold = float(np.percentile(residual, 99.5))
    peaks = residual[residual > threshold]
    if peaks.size == 0:
        return 0.0
    return float(np.sum(peaks - threshold))


def detect_spot_polarity(image: np.ndarray, sigma_small: float = 1.0) -> dict:
    sigma_large = max(4.0, sigma_small * 6.0)
    original_score = _positive_blob_score(image, sigma_small=sigma_small, sigma_large=sigma_large)
    inverted = invert_image(image)
    inverted_score = _positive_blob_score(inverted, sigma_small=sigma_small, sigma_large=sigma_large)
    total = original_score + inverted_score + 1e-9
    confidence = abs(original_score - inverted_score) / total

    if original_score >= inverted_score:
        detected = "Spots hell"
    else:
        detected = "Spots dunkel"

    return {
        "detected_polarity": detected,
        "original_score": original_score,
        "inverted_score": inverted_score,
        "confidence": float(confidence),
    }


def orient_image(image: np.ndarray, polarity_mode: str, sigma_small: float = 1.0) -> tuple[np.ndarray, dict]:
    image = np.asarray(image, dtype=np.float32)
    auto_result = detect_spot_polarity(image, sigma_small=sigma_small)

    if polarity_mode == "Auto":
        effective_mode = auto_result["detected_polarity"]
        invert_applied = effective_mode == "Spots dunkel"
    elif polarity_mode == "Spots hell":
        effective_mode = "Spots hell"
        invert_applied = False
    elif polarity_mode in {"Spots dunkel", "Bild invertieren"}:
        effective_mode = "Spots dunkel"
        invert_applied = True
    else:
        raise ValueError(f"Unbekannter Polarity-Modus: {polarity_mode}")

    oriented = invert_image(image) if invert_applied else image.copy()
    metadata = {
        "selected_mode": polarity_mode,
        "effective_mode": effective_mode,
        "invert_applied": invert_applied,
        "auto_detected_polarity": auto_result["detected_polarity"],
        "polarity_confidence": auto_result["confidence"],
        "auto_scores": {
            "original": auto_result["original_score"],
            "inverted": auto_result["inverted_score"],
        },
    }
    return oriented, metadata
