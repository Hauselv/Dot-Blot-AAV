import json

import pandas as pd

from dotblot.export import analysis_manifest_bytes
from dotblot.types import AnalysisConfig, GridConfig, ImageInfo


def test_analysis_manifest_contains_linear_range_fields():
    image_info = ImageInfo(
        filename="example.png",
        shape=(10, 10),
        dtype="uint8",
        intensity_min=0.0,
        intensity_max=255.0,
        is_color=False,
        file_hash="abc123",
    )
    grid_config = GridConfig(
        rows=1,
        cols=1,
        anchor_x=5.0,
        anchor_y=5.0,
        pitch_x=10.0,
        pitch_y=10.0,
        rotation_deg=0.0,
        roi_radius=3.0,
    )
    analysis_config = AnalysisConfig(
        polarity_mode="Auto",
        auto_detected_polarity="Spots hell",
        invert_applied=False,
        polarity_confidence=0.9,
        background_mode="local_annulus_median",
        smooth_sigma=1.0,
        saturation_fraction_threshold=0.18,
        saturation_dynamic_fraction=0.985,
        weak_spot_snr_threshold=1.0,
        reference_sample=None,
        normalization_mode="per_dilution",
        linear_range_x_mode="inverse_dilution",
        linear_range_min_points=3,
        linear_range_exclude_saturated=True,
    )
    metadata = pd.DataFrame([{"spot_id": "R01C01", "sample": "A"}])
    linear_summary = pd.DataFrame([{"sample": "A", "slope": 1.2, "r_squared": 0.99}])
    results = pd.DataFrame([{"spot_id": "R01C01", "linear_fit_include": True}])

    manifest = json.loads(
        analysis_manifest_bytes(
            image_info,
            grid_config,
            analysis_config,
            metadata,
            linear_summary=linear_summary,
            results=results,
        ).decode("utf-8")
    )

    assert manifest["analysis_config"]["linear_range_x_mode"] == "inverse_dilution"
    assert manifest["linear_range_summary"][0]["sample"] == "A"
    assert manifest["linear_fit_selection"][0]["linear_fit_include"] is True
