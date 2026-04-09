from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_to_reference(results: pd.DataFrame, reference_sample: str | None) -> pd.DataFrame:
    results = results.copy()
    if results.empty:
        results["reference_value"] = np.nan
        results["normalized_signal"] = np.nan
        results["normalization_note"] = "Keine Spots vorhanden"
        return results

    results["reference_value"] = np.nan
    results["normalized_signal"] = np.nan
    results["normalization_note"] = ""

    if not reference_sample:
        results["normalization_note"] = "Keine Referenz gewaehlt"
        return results

    included = results[~results["exclude"]].copy()
    reference_rows = included[included["sample"] == reference_sample]

    if reference_rows.empty:
        results["normalization_note"] = f"Referenzsample '{reference_sample}' nicht vorhanden"
        return results

    ref_values = (
        reference_rows.groupby("dilution", dropna=False)["corrected_integrated_intensity"]
        .median()
        .replace(0, np.nan)
    )
    results["reference_value"] = results["dilution"].map(ref_values)
    results["normalized_signal"] = results["corrected_integrated_intensity"] / results["reference_value"]
    results["normalization_note"] = f"Pro Verdunnung auf Referenzsample '{reference_sample}' normalisiert"
    return results
