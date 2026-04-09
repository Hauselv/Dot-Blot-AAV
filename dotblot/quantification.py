from __future__ import annotations

import numpy as np

from .background import select_background_value, summarize_background
from .masks import estimate_expected_area
from .types import AnalysisConfig, GridConfig


def quantify_spot(
    image: np.ndarray,
    spot_mask: np.ndarray,
    background_mask: np.ndarray,
    grid_config: GridConfig,
    analysis_config: AnalysisConfig,
    global_background_summary: dict | None = None,
) -> dict:
    image = np.asarray(image, dtype=np.float32)
    spot_values = image[spot_mask]
    local_background_values = image[background_mask]

    if analysis_config.background_mode.startswith("global"):
        if global_background_summary is None:
            raise ValueError("Globaler Background wurde nicht uebergeben.")
        background_summary = global_background_summary
    else:
        background_summary = summarize_background(local_background_values)

    background_value = select_background_value(background_summary, analysis_config.background_mode)

    raw_sum = float(np.sum(spot_values)) if spot_values.size else np.nan
    raw_mean = float(np.mean(spot_values)) if spot_values.size else np.nan
    corrected_sum = raw_sum - spot_values.size * background_value if spot_values.size else np.nan
    corrected_mean = raw_mean - background_value if spot_values.size else np.nan
    dynamic_range = float(np.max(image) - np.min(image) + 1e-9)
    saturation_threshold = float(np.min(image) + analysis_config.saturation_dynamic_fraction * dynamic_range)

    saturated_fraction = float(np.mean(spot_values >= saturation_threshold)) if spot_values.size else 0.0
    spot_std = float(np.std(spot_values, ddof=0)) if spot_values.size else np.nan
    background_std = float(background_summary["background_std"]) if background_summary["background_n_pixels"] else np.nan
    snr = corrected_mean / (background_std + 1e-9) if np.isfinite(corrected_mean) and np.isfinite(background_std) else np.nan

    expected_area = estimate_expected_area(grid_config.roi_radius, grid_config.roi_shape)
    actual_area = int(spot_values.size)
    edge_clipped = actual_area < 0.9 * expected_area
    is_saturated = bool(
        spot_values.size
        and np.nanmax(spot_values) >= saturation_threshold
        and (
            saturated_fraction >= analysis_config.saturation_fraction_threshold
            or spot_std <= 0.01 * dynamic_range
        )
    )
    is_weak = bool(np.isfinite(snr) and snr < analysis_config.weak_spot_snr_threshold)

    return {
        "spot_area_pixels": actual_area,
        "raw_integrated_intensity": raw_sum,
        "raw_mean_intensity": raw_mean,
        "background_value": background_value,
        "background_mean": background_summary["background_mean"],
        "background_median": background_summary["background_median"],
        "background_std": background_summary["background_std"],
        "background_n_pixels": background_summary["background_n_pixels"],
        "corrected_integrated_intensity": corrected_sum,
        "corrected_mean_intensity": corrected_mean,
        "snr_estimate": snr,
        "saturated_fraction": saturated_fraction,
        "is_saturated": is_saturated,
        "is_weak": is_weak,
        "is_negative_after_background": bool(np.isfinite(corrected_sum) and corrected_sum < 0),
        "edge_clipped": edge_clipped,
    }
