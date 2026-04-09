from __future__ import annotations

import json
from dataclasses import asdict
from io import BytesIO

import pandas as pd

from .config import APP_NAME, APP_VERSION
from .types import AnalysisConfig, GridConfig, ImageInfo


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def figure_to_png_bytes(fig) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def analysis_manifest_bytes(
    image_info: ImageInfo,
    grid_config: GridConfig,
    analysis_config: AnalysisConfig,
    metadata: pd.DataFrame,
    linear_summary: pd.DataFrame | None = None,
    results: pd.DataFrame | None = None,
) -> bytes:
    payload = {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "image_info": asdict(image_info),
        "grid_config": asdict(grid_config),
        "analysis_config": asdict(analysis_config),
        "spot_metadata": metadata.to_dict(orient="records"),
    }
    if linear_summary is not None:
        payload["linear_range_summary"] = linear_summary.to_dict(orient="records")
    if results is not None and "linear_fit_include" in results.columns:
        payload["linear_fit_selection"] = results.loc[:, ["spot_id", "linear_fit_include"]].to_dict(orient="records")
    return json.dumps(payload, indent=2).encode("utf-8")
