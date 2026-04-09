from __future__ import annotations

from dataclasses import asdict

from .types import AnalysisConfig, GridConfig

APP_NAME = "Dot Blot Quantifier"
APP_VERSION = "0.1.0"

POLARITY_OPTIONS = (
    "Auto",
    "Spots hell",
    "Spots dunkel",
    "Bild invertieren",
)

BACKGROUND_OPTIONS = {
    "local_annulus_median": "Lokaler Annulus (Median)",
    "local_annulus_mean": "Lokaler Annulus (Mean)",
    "local_annulus_plane": "Lokaler Annulus (Planarer Fit)",
    "local_window_median": "Lokales Fenster (Median)",
    "local_window_mean": "Lokales Fenster (Mean)",
    "local_window_plane": "Lokales Fenster (Planarer Fit)",
    "surface_median": "Background Surface (Median)",
    "surface_mean": "Background Surface (Mean)",
    "global_median": "Globaler Hintergrund (Median)",
    "global_mean": "Globaler Hintergrund (Mean)",
}

METADATA_PRESETS = (
    "Flexibel / manuell",
    "Zeilen = Samples",
    "Spalten = Samples",
)

ROI_SHAPES = ("circle", "square")
LAYOUT_MODES = {
    "grid": "Regulaeres Grid",
    "free_spots": "Freie Spot-Detektion",
}

DEFAULT_ROWS = 4
DEFAULT_COLS = 6


def default_grid_config() -> GridConfig:
    return GridConfig(
        rows=DEFAULT_ROWS,
        cols=DEFAULT_COLS,
        anchor_x=80.0,
        anchor_y=80.0,
        pitch_x=70.0,
        pitch_y=70.0,
        rotation_deg=0.0,
        roi_radius=18.0,
        roi_shape="circle",
        annulus_inner_scale=1.35,
        annulus_outer_scale=2.15,
    )


def default_analysis_config() -> AnalysisConfig:
    return AnalysisConfig(
        layout_mode="grid",
        polarity_mode="Auto",
        auto_detected_polarity="Unbestimmt",
        invert_applied=False,
        polarity_confidence=0.0,
        background_mode="local_annulus_median",
        smooth_sigma=1.0,
        saturation_fraction_threshold=0.18,
        saturation_dynamic_fraction=0.985,
        weak_spot_snr_threshold=1.0,
        reference_sample=None,
        normalization_mode="per_dilution",
        log_dilution_axis=False,
        center_refinement_enabled=True,
        center_refinement_radius=5.0,
        linear_range_x_mode="inverse_dilution",
        linear_range_min_points=3,
        linear_range_exclude_saturated=True,
        local_window_scale=3.5,
        surface_sigma=25.0,
        detection_background_sigma=20.0,
        detection_min_sigma=1.5,
        detection_max_sigma=8.0,
        detection_num_sigma=12,
        detection_threshold_rel=0.12,
        detection_overlap=0.5,
        detection_min_distance=8.0,
        detection_crop_enabled=False,
        detection_crop_box=None,
        adaptive_roi_enabled=True,
        adaptive_roi_threshold_rel=0.28,
        adaptive_roi_min_radius=3.0,
        adaptive_roi_max_radius=18.0,
    )


def config_as_dict(grid_config: GridConfig, analysis_config: AnalysisConfig) -> dict:
    return {
        "grid": asdict(grid_config),
        "analysis": asdict(analysis_config),
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
    }
