from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
import pandas as pd

from .preprocessing import normalize_for_display
from .types import GridConfig


def _spot_color(row: pd.Series) -> str:
    if bool(row.get("exclude", False)):
        return "#d62728"
    if bool(row.get("is_saturated", False)):
        return "#ff7f0e"
    if bool(row.get("is_weak", False)):
        return "#f1c40f"
    return "#00bcd4"


def create_qc_overlay(image, results: pd.DataFrame, grid_config: GridConfig, title: str = "QC Overlay"):
    display = normalize_for_display(image)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(display, cmap="gray")

    for _, row in results.iterrows():
        color = _spot_color(row)
        center = (row["center_x"], row["center_y"])
        if grid_config.roi_shape == "circle":
            roi_patch = Circle(center, radius=grid_config.roi_radius, fill=False, linewidth=1.5, edgecolor=color)
        else:
            roi_patch = Rectangle(
                (row["center_x"] - grid_config.roi_radius, row["center_y"] - grid_config.roi_radius),
                width=2 * grid_config.roi_radius,
                height=2 * grid_config.roi_radius,
                fill=False,
                linewidth=1.5,
                edgecolor=color,
            )
        bg_patch = Circle(
            center,
            radius=grid_config.roi_radius * grid_config.annulus_outer_scale,
            fill=False,
            linewidth=1.0,
            linestyle="--",
            edgecolor=color,
            alpha=0.8,
        )
        ax.add_patch(roi_patch)
        ax.add_patch(bg_patch)
        ax.text(
            row["center_x"] + grid_config.roi_radius * 0.2,
            row["center_y"] - grid_config.roi_radius * 0.2,
            str(row["spot_id"]),
            color=color,
            fontsize=8,
            ha="left",
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 1},
        )

    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def summarize_qc_flags(results: pd.DataFrame) -> pd.DataFrame:
    summary = {
        "Anzahl Spots": int(len(results)),
        "Ausgeschlossen": int(results["exclude"].sum()),
        "Gesaettigt": int(results["is_saturated"].sum()),
        "Schwach": int(results["is_weak"].sum()),
        "Negativ nach Background": int(results["is_negative_after_background"].sum()),
        "Randbeschnitten": int(results["edge_clipped"].sum()),
    }
    return pd.DataFrame({"Metrik": list(summary.keys()), "Wert": list(summary.values())})
