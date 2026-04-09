import numpy as np
import pandas as pd

from dotblot.analysis import run_quantification
from dotblot.types import AnalysisConfig, GridConfig


def _base_analysis_config(background_mode: str) -> AnalysisConfig:
    return AnalysisConfig(
        polarity_mode="Spots hell",
        auto_detected_polarity="Spots hell",
        invert_applied=False,
        polarity_confidence=1.0,
        background_mode=background_mode,
        smooth_sigma=1.0,
        saturation_fraction_threshold=0.18,
        saturation_dynamic_fraction=0.985,
        weak_spot_snr_threshold=1.0,
        reference_sample=None,
        normalization_mode="per_dilution",
        log_dilution_axis=False,
        local_window_scale=3.0,
        surface_sigma=12.0,
    )


def _single_spot_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "spot_id": "R01C01",
                "row": 0,
                "col": 0,
                "sample": "Sample_1",
                "dilution": "1",
                "replicate": 1,
                "exclude": False,
            }
        ]
    )


def test_local_window_background_tracks_local_gradient():
    yy, xx = np.mgrid[:90, :120]
    image = 5.0 + 0.05 * xx
    image += 8.0 * np.exp(-(((xx - 60) ** 2 + (yy - 45) ** 2) / (2 * 5.0**2)))

    grid_config = GridConfig(
        rows=1,
        cols=1,
        anchor_x=60.0,
        anchor_y=45.0,
        pitch_x=20.0,
        pitch_y=20.0,
        rotation_deg=0.0,
        roi_radius=7.0,
    )
    results = run_quantification(image.astype(np.float32), grid_config, _single_spot_metadata(), _base_analysis_config("local_window_median"))

    assert len(results) == 1
    assert 7.0 <= results.loc[0, "background_value"] <= 9.0
    assert results.loc[0, "background_method"] == "local_window_median"


def test_surface_background_estimates_smooth_background_level():
    yy, xx = np.mgrid[:90, :90]
    background = 12.0 + 0.03 * xx + 0.04 * yy
    image = background + 25.0 * np.exp(-(((xx - 45) ** 2 + (yy - 45) ** 2) / (2 * 4.0**2)))

    grid_config = GridConfig(
        rows=1,
        cols=1,
        anchor_x=45.0,
        anchor_y=45.0,
        pitch_x=20.0,
        pitch_y=20.0,
        rotation_deg=0.0,
        roi_radius=6.0,
    )
    results = run_quantification(image.astype(np.float32), grid_config, _single_spot_metadata(), _base_analysis_config("surface_median"))

    expected_background = float(background[45, 45])
    assert len(results) == 1
    assert abs(results.loc[0, "background_value"] - expected_background) < 3.5
    assert results.loc[0, "background_method"] == "surface_median"


def test_local_window_plane_follows_gradient_better_than_median():
    yy, xx = np.mgrid[:120, :120]
    background = 8.0 + 0.09 * xx + 0.04 * yy
    image = background + 15.0 * np.exp(-(((xx - 72) ** 2 + (yy - 50) ** 2) / (2 * 4.5**2)))

    grid_config = GridConfig(
        rows=1,
        cols=1,
        anchor_x=72.0,
        anchor_y=50.0,
        pitch_x=20.0,
        pitch_y=20.0,
        rotation_deg=0.0,
        roi_radius=6.0,
    )
    median_results = run_quantification(
        image.astype(np.float32),
        grid_config,
        _single_spot_metadata(),
        _base_analysis_config("local_window_median"),
    )
    plane_results = run_quantification(
        image.astype(np.float32),
        grid_config,
        _single_spot_metadata(),
        _base_analysis_config("local_window_plane"),
    )

    expected_background = float(background[50, 72])
    median_error = abs(median_results.loc[0, "background_value"] - expected_background)
    plane_error = abs(plane_results.loc[0, "background_value"] - expected_background)

    assert plane_results.loc[0, "background_method"] == "local_window_plane"
    assert plane_results.loc[0, "background_fit_rmse"] >= 0
    assert plane_error <= median_error
