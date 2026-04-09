from __future__ import annotations

import numpy as np


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
    }


def select_background_value(summary: dict, mode: str) -> float:
    if mode in {"local_annulus_median", "global_median"}:
        return float(summary["background_median"])
    if mode in {"local_annulus_mean", "global_mean"}:
        return float(summary["background_mean"])
    raise ValueError(f"Unbekannter Background-Modus: {mode}")
