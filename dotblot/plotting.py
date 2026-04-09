from __future__ import annotations

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def _filtered_results(results: pd.DataFrame) -> pd.DataFrame:
    return results.loc[~results["exclude"]].copy()


def plot_signal_by_spot(results: pd.DataFrame, value_column: str, title: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(results))
    colors = ["#d62728" if exclude else "#1f77b4" for exclude in results["exclude"]]
    ax.bar(x, results[value_column].fillna(0.0), color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(results["spot_id"], rotation=90, fontsize=8)
    ax.set_ylabel(value_column)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_dilution_series(results: pd.DataFrame, value_column: str, log_x: bool = False):
    plot_df = _filtered_results(results)
    fig, ax = plt.subplots(figsize=(8, 5))

    if plot_df.empty:
        ax.set_title("Keine Daten fuer Verdunnungsplot verfuegbar")
        ax.set_axis_off()
        return fig

    plot_df["dilution_numeric"] = pd.to_numeric(plot_df["dilution"], errors="coerce")
    use_numeric = plot_df["dilution_numeric"].notna().all()

    for sample, sample_df in plot_df.groupby("sample"):
        sample_df = sample_df.copy()
        if use_numeric:
            sample_df = sample_df.sort_values("dilution_numeric")
            x = sample_df["dilution_numeric"].to_numpy()
        else:
            sample_df = sample_df.sort_values(["dilution", "replicate"])
            x = np.arange(len(sample_df))
        y = sample_df[value_column].to_numpy()
        marker_style = ["s" if saturated else "o" for saturated in sample_df["is_saturated"].fillna(False).tolist()]
        for xi, yi, marker in zip(x, y, marker_style):
            ax.scatter([xi], [yi], marker=marker, s=55)
        ax.plot(x, y, linewidth=1.5, label=sample)

    if use_numeric:
        ax.set_xlabel("Verdunnung")
        if log_x and np.all(plot_df["dilution_numeric"] > 0):
            ax.set_xscale("log")
    else:
        ax.set_xlabel("Verdunnungsstufe (geordnet)")

    ax.set_ylabel(value_column)
    ax.set_title(f"{value_column} ueber Verdunnung")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_grid_heatmap(results: pd.DataFrame, value_column: str):
    if results.empty:
        fig, ax = plt.subplots()
        ax.set_axis_off()
        return fig

    if "row" not in results.columns or "col" not in results.columns:
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.set_title("Keine Grid-Heatmap fuer diesen Analysemodus verfuegbar")
        return fig

    duplicate_positions = results.duplicated(subset=["row", "col"]).any()
    if duplicate_positions:
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.set_title("Grid-Heatmap ist fuer freie Spot-Layouts nicht eindeutig")
        return fig

    heatmap = results.pivot(index="row", columns="col", values=value_column).sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(heatmap.to_numpy(), cmap="viridis")
    ax.set_title(f"Grid Heatmap: {value_column}")
    ax.set_xlabel("Spalte")
    ax.set_ylabel("Zeile")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_linear_range_fits(results: pd.DataFrame, summaries: pd.DataFrame, value_column: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = results.loc[~results["exclude"]].copy()

    if plot_df.empty:
        ax.set_axis_off()
        ax.set_title("Keine Daten fuer linearen Bereich verfuegbar")
        return fig

    for sample, sample_df in plot_df.groupby("sample"):
        sample_df = sample_df.sort_values("linear_x")
        x = sample_df["linear_x"].to_numpy(dtype=float)
        y = sample_df[value_column].to_numpy(dtype=float)
        include_mask = sample_df["linear_fit_include"].fillna(False).to_numpy(dtype=bool)

        ax.plot(x, y, alpha=0.35, linewidth=1.0)
        ax.scatter(x[~include_mask], y[~include_mask], s=30, alpha=0.45, label=None)
        ax.scatter(x[include_mask], y[include_mask], s=60, marker="o", label=sample)

        summary = summaries.loc[summaries["sample"] == sample]
        if not summary.empty and np.isfinite(summary.iloc[0]["slope"]):
            slope = float(summary.iloc[0]["slope"])
            intercept = float(summary.iloc[0]["intercept"])
            x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ax.plot(x_fit, slope * x_fit + intercept, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Linear-range X")
    ax.set_ylabel(value_column)
    ax.set_title("Lineare Bereiche und Fits")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_linear_summary_metric(summaries: pd.DataFrame, metric: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_df = summaries.copy()
    if plot_df.empty or metric not in plot_df.columns:
        ax.set_axis_off()
        ax.set_title(title)
        return fig

    plot_df = plot_df.loc[np.isfinite(plot_df[metric])].copy()
    if plot_df.empty:
        ax.set_axis_off()
        ax.set_title(title)
        return fig

    ax.bar(plot_df["sample"].astype(str), plot_df[metric], color="#1f77b4", alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
