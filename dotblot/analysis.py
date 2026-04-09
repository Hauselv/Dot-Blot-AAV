from __future__ import annotations

import numpy as np
import pandas as pd

from .background import estimate_background_surface, summarize_background
from .detection import detect_spots
from .grid import generate_grid_centers, refine_grid_centers
from .masks import build_masks, build_masks_for_spot
from .normalization import normalize_to_reference
from .quantification import quantify_spot
from .types import AnalysisConfig, GridConfig


def run_quantification(
    oriented_image: np.ndarray,
    grid_config: GridConfig,
    metadata: pd.DataFrame,
    analysis_config: AnalysisConfig,
) -> pd.DataFrame:
    metadata = metadata.copy()

    if analysis_config.layout_mode == "free_spots":
        if metadata.empty:
            return normalize_to_reference(metadata.copy(), analysis_config.reference_sample)
        required = {"center_x", "center_y"}
        if not required.issubset(metadata.columns):
            raise ValueError("Freie Spot-Detektion erwartet center_x und center_y in den Metadaten.")
        if "roi_radius" not in metadata.columns:
            metadata["roi_radius"] = float(grid_config.roi_radius)
        metadata["row"] = pd.to_numeric(metadata.get("row", 0), errors="coerce").fillna(0).astype(int)
        metadata["col"] = pd.to_numeric(metadata.get("col", 0), errors="coerce").fillna(0).astype(int)
        metadata = metadata.sort_values(["row", "col", "center_y", "center_x"]).reset_index(drop=True)
    else:
        metadata = metadata.sort_values(["row", "col"]).reset_index(drop=True).copy()
        centers = generate_grid_centers(grid_config)
        if analysis_config.center_refinement_enabled and analysis_config.center_refinement_radius > 0:
            centers = refine_grid_centers(
                oriented_image,
                centers,
                roi_radius=grid_config.roi_radius,
                search_radius=analysis_config.center_refinement_radius,
            )
        metadata["center_x"] = centers[:, 0]
        metadata["center_y"] = centers[:, 1]
        metadata["roi_radius"] = float(grid_config.roi_radius)

    if analysis_config.layout_mode == "free_spots" and analysis_config.center_refinement_enabled:
        centers = metadata.loc[:, ["center_x", "center_y"]].to_numpy(dtype=np.float32)
        radii = metadata["roi_radius"].to_numpy(dtype=np.float32)
        refined_centers = []
        for center, roi_radius in zip(centers, radii):
            refined = refine_grid_centers(
                oriented_image,
                np.asarray([center], dtype=np.float32),
                roi_radius=float(roi_radius),
                search_radius=max(analysis_config.center_refinement_radius, roi_radius),
            )
            refined_centers.append(refined[0])
        refined_centers = np.asarray(refined_centers, dtype=np.float32)
        metadata["center_x"] = refined_centers[:, 0]
        metadata["center_y"] = refined_centers[:, 1]

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
    for _, meta_row in metadata.iterrows():
        center_x = float(meta_row["center_x"])
        center_y = float(meta_row["center_y"])
        roi_radius = float(meta_row.get("roi_radius", grid_config.roi_radius))
        roi_shape = str(meta_row.get("roi_shape", grid_config.roi_shape))
        spot_mask, background_mask = build_masks_for_spot(
            image_shape,
            center_x,
            center_y,
            roi_radius=roi_radius,
            roi_shape=roi_shape,
            annulus_inner_scale=grid_config.annulus_inner_scale,
            annulus_outer_scale=grid_config.annulus_outer_scale,
        )
        metrics = quantify_spot(
            oriented_image,
            spot_mask,
            background_mask,
            roi_radius=roi_radius,
            roi_shape=roi_shape,
            analysis_config=analysis_config,
            global_background_summary=global_summary,
            center_x=float(center_x),
            center_y=float(center_y),
            surface_background_map=surface_background_map,
        )
        record = meta_row.to_dict()
        record.update({"center_x": float(center_x), "center_y": float(center_y), "roi_radius": roi_radius, "roi_shape": roi_shape})
        record.update(metrics)
        records.append(record)

    results = pd.DataFrame.from_records(records)
    results = normalize_to_reference(results, analysis_config.reference_sample)
    return results


def auto_detect_spot_table(oriented_image: np.ndarray, analysis_config: AnalysisConfig) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    return detect_spots(
        oriented_image=oriented_image,
        background_sigma=analysis_config.detection_background_sigma,
        min_sigma=analysis_config.detection_min_sigma,
        max_sigma=analysis_config.detection_max_sigma,
        num_sigma=analysis_config.detection_num_sigma,
        threshold_rel=analysis_config.detection_threshold_rel,
        overlap=analysis_config.detection_overlap,
        min_distance=analysis_config.detection_min_distance,
        crop_box=analysis_config.detection_crop_box if analysis_config.detection_crop_enabled else None,
        adaptive_roi_enabled=analysis_config.adaptive_roi_enabled,
        adaptive_roi_threshold_rel=analysis_config.adaptive_roi_threshold_rel,
        adaptive_roi_min_radius=analysis_config.adaptive_roi_min_radius,
        adaptive_roi_max_radius=analysis_config.adaptive_roi_max_radius,
    )
