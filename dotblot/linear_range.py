from __future__ import annotations

import numpy as np
import pandas as pd


LINEAR_RANGE_X_OPTIONS = {
    "input": "Verdunnung wie eingegeben",
    "inverse_dilution": "1 / Verdunnung",
    "log10_input": "log10(Verdunnung)",
    "log10_inverse_dilution": "log10(1 / Verdunnung)",
}


def add_linear_x(results: pd.DataFrame, x_mode: str) -> pd.DataFrame:
    df = results.copy()
    df["dilution_numeric"] = pd.to_numeric(df["dilution"], errors="coerce")
    x = df["dilution_numeric"].astype(float)

    if x_mode == "input":
        linear_x = x
    elif x_mode == "inverse_dilution":
        linear_x = 1.0 / x.replace(0, np.nan)
    elif x_mode == "log10_input":
        linear_x = np.where(x > 0, np.log10(x), np.nan)
    elif x_mode == "log10_inverse_dilution":
        inv = 1.0 / x.replace(0, np.nan)
        linear_x = np.where(inv > 0, np.log10(inv), np.nan)
    else:
        raise ValueError(f"Unbekannter X-Modus fuer linearen Bereich: {x_mode}")

    df["linear_x"] = linear_x
    return df


def _fit_line(x: np.ndarray, y: np.ndarray) -> dict | None:
    if len(x) < 2 or np.allclose(x, x[0]):
        return None

    slope, intercept = np.polyfit(x, y, deg=1)
    prediction = slope * x + intercept
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "predicted": prediction,
    }


def _best_contiguous_window(sample_df: pd.DataFrame, value_column: str, min_points: int) -> tuple[list[int], dict | None]:
    best_indices: list[int] = []
    best_fit: dict | None = None
    best_score: tuple[float, int, float] | None = None

    if len(sample_df) < min_points:
        return best_indices, best_fit

    for start in range(0, len(sample_df) - min_points + 1):
        for end in range(start + min_points, len(sample_df) + 1):
            window = sample_df.iloc[start:end]
            x = window["linear_x"].to_numpy(dtype=float)
            y = window[value_column].to_numpy(dtype=float)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                continue
            fit = _fit_line(x, y)
            if fit is None:
                continue
            adjusted_r2 = fit["r_squared"] + 0.02 * max(len(window) - min_points, 0)
            score = (adjusted_r2, len(window), abs(fit["slope"]))
            if best_score is None or score > best_score:
                best_score = score
                best_indices = window.index.tolist()
                best_fit = fit

    return best_indices, best_fit


def auto_select_linear_range(
    results: pd.DataFrame,
    value_column: str,
    x_mode: str,
    min_points: int = 3,
    exclude_saturated: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_linear_x(results, x_mode=x_mode)
    df["linear_fit_auto_include"] = False
    df["linear_fit_include"] = False
    df["linear_fit_eligible"] = False

    summaries: list[dict] = []
    for sample, sample_df in df.groupby("sample", dropna=False):
        working = sample_df.loc[~sample_df["exclude"]].copy()
        if exclude_saturated:
            working = working.loc[~working["is_saturated"]].copy()
        working = working.loc[np.isfinite(working["linear_x"]) & np.isfinite(working[value_column])].copy()
        working = working.sort_values("linear_x")
        df.loc[working.index, "linear_fit_eligible"] = True

        indices, fit = _best_contiguous_window(working, value_column=value_column, min_points=min_points)
        if indices:
            df.loc[indices, "linear_fit_auto_include"] = True
            df.loc[indices, "linear_fit_include"] = True
            selected = working.loc[indices]
            summaries.append(
                {
                    "sample": sample,
                    "fit_status": "auto",
                    "n_points": int(len(selected)),
                    "x_mode": x_mode,
                    "slope": fit["slope"],
                    "intercept": fit["intercept"],
                    "r_squared": fit["r_squared"],
                    "x_min": float(selected["linear_x"].min()),
                    "x_max": float(selected["linear_x"].max()),
                    "selected_spots": ", ".join(selected["spot_id"].astype(str).tolist()),
                }
            )
        else:
            summaries.append(
                {
                    "sample": sample,
                    "fit_status": "insufficient_data",
                    "n_points": int(len(working)),
                    "x_mode": x_mode,
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r_squared": np.nan,
                    "x_min": np.nan,
                    "x_max": np.nan,
                    "selected_spots": "",
                }
            )

    return df, pd.DataFrame(summaries)


def apply_manual_linear_selection(
    results: pd.DataFrame,
    value_column: str,
    x_mode: str,
    min_points: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_linear_x(results, x_mode=x_mode)
    if "linear_fit_include" not in df.columns:
        df["linear_fit_include"] = False

    summaries: list[dict] = []
    for sample, sample_df in df.groupby("sample", dropna=False):
        selected = sample_df.loc[
            (~sample_df["exclude"])
            & sample_df["linear_fit_include"].fillna(False)
            & np.isfinite(sample_df["linear_x"])
            & np.isfinite(sample_df[value_column])
        ].copy()
        selected = selected.sort_values("linear_x")

        if len(selected) >= min_points:
            fit = _fit_line(
                selected["linear_x"].to_numpy(dtype=float),
                selected[value_column].to_numpy(dtype=float),
            )
        else:
            fit = None

        if fit is None:
            summaries.append(
                {
                    "sample": sample,
                    "fit_status": "manual_insufficient_data",
                    "n_points": int(len(selected)),
                    "x_mode": x_mode,
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r_squared": np.nan,
                    "x_min": np.nan,
                    "x_max": np.nan,
                    "selected_spots": ", ".join(selected["spot_id"].astype(str).tolist()),
                }
            )
            continue

        summaries.append(
            {
                "sample": sample,
                "fit_status": "manual",
                "n_points": int(len(selected)),
                "x_mode": x_mode,
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "r_squared": fit["r_squared"],
                "x_min": float(selected["linear_x"].min()),
                "x_max": float(selected["linear_x"].max()),
                "selected_spots": ", ".join(selected["spot_id"].astype(str).tolist()),
            }
        )

    return df, pd.DataFrame(summaries)
