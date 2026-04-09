from __future__ import annotations

import numpy as np
import pandas as pd

from .background import estimate_background_surface, summarize_background
from .grid import generate_grid_centers, refine_grid_centers
from .masks import build_masks
from .normalization import normalize_to_reference
from .quantification import quantify_spot
from .types import AnalysisConfig, GridConfig


def run_quantification(
    oriented_image: np.ndarray,
    grid_config: GridConfig,
    metadata: pd.DataFrame,
    analysis_config: AnalysisConfig,
) -> pd.DataFrame:
    metadata = metadata.sort_values(["row", "col"]).reset_index(drop=True).copy()
    centers = generate_grid_centers(grid_config)
    if analysis_config.center_refinement_enabled and analysis_config.center_refinement_radius > 0:
        centers = refine_grid_centers(
            oriented_image,
            centers,
            roi_radius=grid_config.roi_radius,
            search_radius=analysis_config.center_refinement_radius,
        )
    image_shape = oriented_image.shape[:2]

    global_summary = None
    surface_background_map = None
    if analysis_config.background_mode.startswith("global"):
        global_summary = summarize_background(oriented_image.reshape(-1))
    if analysis_config.background_mode.startswith("surface"):
        surface_background_map = estimate_background_surface(
            oriented_image,
            sigma=analysis_config.surface_sigma,
        )

    records = []
    for idx, meta_row in metadata.iterrows():
        center_x, center_y = centers[idx]
        spot_mask, background_mask = build_masks(image_shape, float(center_x), float(center_y), grid_config)
        metrics = quantify_spot(
            oriented_image,
            spot_mask,
            background_mask,
            grid_config=grid_config,
            analysis_config=analysis_config,
            global_background_summary=global_summary,
            center_x=float(center_x),
            center_y=float(center_y),
            surface_background_map=surface_background_map,
        )
        record = meta_row.to_dict()
        record.update({"center_x": float(center_x), "center_y": float(center_y)})
        record.update(metrics)
        records.append(record)

    results = pd.DataFrame.from_records(records)
    results = normalize_to_reference(results, analysis_config.reference_sample)
    return results
