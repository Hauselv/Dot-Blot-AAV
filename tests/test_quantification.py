import numpy as np
import pandas as pd

from dotblot.analysis import run_quantification
from dotblot.types import AnalysisConfig, GridConfig


def test_local_annulus_quantification_returns_positive_signal():
    image = np.ones((64, 64), dtype=np.float32) * 5.0
    yy, xx = np.mgrid[:64, :64]
    image += 12.0 * np.exp(-(((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 4.0**2)))

    grid_config = GridConfig(
        rows=1,
        cols=1,
        anchor_x=32.0,
        anchor_y=32.0,
        pitch_x=20.0,
        pitch_y=20.0,
        rotation_deg=0.0,
        roi_radius=6.0,
        annulus_inner_scale=1.4,
        annulus_outer_scale=2.0,
    )
    analysis_config = AnalysisConfig(
        polarity_mode="Spots hell",
        auto_detected_polarity="Spots hell",
        invert_applied=False,
        polarity_confidence=1.0,
        background_mode="local_annulus_median",
        smooth_sigma=1.0,
        saturation_fraction_threshold=0.18,
        saturation_dynamic_fraction=0.985,
        weak_spot_snr_threshold=1.0,
        reference_sample=None,
        normalization_mode="per_dilution",
        log_dilution_axis=False,
    )
    metadata = pd.DataFrame(
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

    results = run_quantification(image, grid_config, metadata, analysis_config)
    assert len(results) == 1
    assert 5.0 <= results.loc[0, "background_value"] < 6.5
    assert results.loc[0, "corrected_integrated_intensity"] > 0
    assert bool(results.loc[0, "is_saturated"]) is False
