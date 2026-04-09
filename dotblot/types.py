from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ImageInfo:
    filename: str
    shape: tuple[int, ...]
    dtype: str
    intensity_min: float
    intensity_max: float
    is_color: bool
    file_hash: str


@dataclass
class GridConfig:
    rows: int
    cols: int
    anchor_x: float
    anchor_y: float
    pitch_x: float
    pitch_y: float
    rotation_deg: float
    roi_radius: float
    roi_shape: str = "circle"
    annulus_inner_scale: float = 1.35
    annulus_outer_scale: float = 2.15


@dataclass
class AnalysisConfig:
    polarity_mode: str
    auto_detected_polarity: str
    invert_applied: bool
    polarity_confidence: float
    background_mode: str
    smooth_sigma: float
    saturation_fraction_threshold: float
    saturation_dynamic_fraction: float
    weak_spot_snr_threshold: float
    reference_sample: Optional[str]
    normalization_mode: str
    log_dilution_axis: bool = False
    center_refinement_enabled: bool = True
    center_refinement_radius: float = 5.0
    linear_range_x_mode: str = "inverse_dilution"
    linear_range_min_points: int = 3
    linear_range_exclude_saturated: bool = True
    local_window_scale: float = 3.5
    surface_sigma: float = 25.0
