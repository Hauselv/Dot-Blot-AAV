import numpy as np
import pandas as pd

from dotblot.linear_range import apply_manual_linear_selection, auto_select_linear_range


def test_auto_select_linear_range_picks_contiguous_linear_window():
    results = pd.DataFrame(
        {
            "spot_id": ["s1", "s2", "s3", "s4", "s5"],
            "sample": ["A"] * 5,
            "dilution": [1, 2, 4, 8, 16],
            "exclude": [False] * 5,
            "is_saturated": [True, False, False, False, False],
            "corrected_integrated_intensity": [1000, 520, 250, 130, 20],
        }
    )

    enriched, summary = auto_select_linear_range(
        results,
        value_column="corrected_integrated_intensity",
        x_mode="inverse_dilution",
        min_points=3,
        exclude_saturated=True,
    )

    selected = enriched.loc[enriched["linear_fit_auto_include"], "spot_id"].tolist()
    assert selected == ["s2", "s3", "s4", "s5"]
    assert summary.loc[0, "fit_status"] == "auto"
    assert summary.loc[0, "n_points"] == 4
    assert summary.loc[0, "r_squared"] > 0.99


def test_manual_linear_selection_computes_summary():
    results = pd.DataFrame(
        {
            "spot_id": ["a", "b", "c", "d"],
            "sample": ["Sample_1"] * 4,
            "dilution": [1, 2, 4, 8],
            "exclude": [False] * 4,
            "is_saturated": [False] * 4,
            "corrected_integrated_intensity": [8.0, 4.1, 2.0, 0.9],
            "linear_fit_include": [False, True, True, True],
        }
    )

    enriched, summary = apply_manual_linear_selection(
        results,
        value_column="corrected_integrated_intensity",
        x_mode="inverse_dilution",
        min_points=3,
    )

    assert np.isfinite(enriched["linear_x"]).all()
    assert summary.loc[0, "fit_status"] == "manual"
    assert summary.loc[0, "n_points"] == 3
    assert summary.loc[0, "r_squared"] > 0.99
