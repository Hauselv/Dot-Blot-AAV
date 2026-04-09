from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def summarize_background(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            "background_mean": np.nan,
            "background_median": np.nan,
            "background_std": np.nan,
            "background_n_pixels": 0,
        }

    return {
        "background_mean": float(np.mean(values)),
        "background_median": float(np.median(values)),
        "background_std": float(np.std(values, ddof=0)),
        "background_n_pixels": int(values.size),
        "background_fit_rmse": np.nan,
    }


def make_local_window_mask(
    shape: tuple[int, int],
    center_x: float,
    center_y: float,
    roi_radius: float,
    window_scale: float,
    spot_mask: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    half_size = max(roi_radius * window_scale, roi_radius + 2.0)
    xmin = max(0, int(np.floor(center_x - half_size)))
    xmax = min(shape[1], int(np.ceil(center_x + half_size + 1)))
    ymin = max(0, int(np.floor(center_y - half_size)))
    ymax = min(shape[0], int(np.ceil(center_y + half_size + 1)))
    mask[ymin:ymax, xmin:xmax] = True
    mask &= ~spot_mask
    return mask


def estimate_background_surface(image: np.ndarray, sigma: float) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    sigma = max(float(sigma), 1.0)
    return gaussian_filter(image, sigma=sigma)


def fit_background_plane(image: np.ndarray, mask: np.ndarray) -> dict:
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    yy, xx = np.nonzero(mask)
    if xx.size < 3:
        summary = summarize_background(image[mask])
        summary["background_fit_rmse"] = np.nan
        summary["background_plane_coefficients"] = None
        return summary

    values = image[yy, xx].astype(np.float32)
    design = np.column_stack([xx.astype(np.float32), yy.astype(np.float32), np.ones_like(xx, dtype=np.float32)])
    coeffs, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    fitted = design @ coeffs
    residual = values - fitted
    summary = summarize_background(values)
    summary["background_fit_rmse"] = float(np.sqrt(np.mean(residual**2)))
    summary["background_plane_coefficients"] = coeffs.astype(np.float32)
    return summary


def evaluate_background_plane(summary: dict, x: float, y: float) -> float:
    coeffs = summary.get("background_plane_coefficients")
    if coeffs is None:
        return float(summary.get("background_median", np.nan))
    a, b, c = [float(value) for value in coeffs]
    return float(a * x + b * y + c)


def make_surface_corrected_preview(image: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.float32)
    background = estimate_background_surface(image, sigma=sigma)
    corrected = np.clip(image - background, 0.0, None)
    return background, corrected


def select_background_value(summary: dict, mode: str) -> float:
    if mode in {"local_annulus_median", "local_window_median", "surface_median", "global_median"}:
        return float(summary["background_median"])
    if mode in {"local_annulus_mean", "local_window_mean", "surface_mean", "global_mean"}:
        return float(summary["background_mean"])
    if mode in {"local_annulus_plane", "local_window_plane"}:
        return float(summary["background_plane_value"])
    raise ValueError(f"Unbekannter Background-Modus: {mode}")
