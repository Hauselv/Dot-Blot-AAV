from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_log
from skimage.measure import label, regionprops


def estimate_detection_background(image: np.ndarray, sigma: float) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    sigma = max(float(sigma), 1.0)
    return gaussian_filter(image, sigma=sigma)


def make_detection_image(image: np.ndarray, background_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.float32)
    background = estimate_detection_background(image, sigma=background_sigma)
    detection = np.clip(image - background, 0.0, None)
    return detection, background


def sanitize_crop_box(crop_box: tuple[float, float, float, float] | None, image_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if crop_box is None:
        return None

    height, width = image_shape[:2]
    x0, y0, x1, y1 = [float(value) for value in crop_box]
    xmin = int(np.clip(min(x0, x1), 0, width - 1))
    xmax = int(np.clip(max(x0, x1), xmin + 1, width))
    ymin = int(np.clip(min(y0, y1), 0, height - 1))
    ymax = int(np.clip(max(y0, y1), ymin + 1, height))
    if xmax - xmin < 2 or ymax - ymin < 2:
        return None
    return xmin, ymin, xmax, ymax


def apply_crop_to_image(image: np.ndarray, crop_box: tuple[float, float, float, float] | None) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    sanitized = sanitize_crop_box(crop_box, image.shape[:2])
    if sanitized is None:
        return image
    xmin, ymin, xmax, ymax = sanitized
    cropped = np.zeros_like(image, dtype=np.float32)
    cropped[ymin:ymax, xmin:xmax] = image[ymin:ymax, xmin:xmax]
    return cropped


def _component_candidates(
    detection_image: np.ndarray,
    min_sigma: float,
    max_sigma: float,
    threshold_rel: float,
    min_distance: float,
    adaptive_roi_min_radius: float,
) -> pd.DataFrame:
    smooth_sigma = max(0.9, min(float(min_sigma), 2.5))
    smooth = gaussian_filter(np.asarray(detection_image, dtype=np.float32), sigma=smooth_sigma)
    percentile = float(np.clip(99.45 + 2.2 * float(threshold_rel), 98.8, 99.96))
    threshold_value = float(np.percentile(smooth, percentile))
    binary = smooth >= threshold_value

    min_area = max(8, int(round(math.pi * max(min_sigma, 1.0) ** 2)))
    labels = label(binary)
    if labels.max() == 0:
        return pd.DataFrame(columns=["center_x", "center_y", "roi_radius", "detection_sigma", "detection_score"])

    h, w = smooth.shape
    border_margin = int(max(8.0, min_distance * 0.75, adaptive_roi_min_radius * 1.5))
    max_core_area = math.pi * max(max_sigma * 3.2, adaptive_roi_min_radius * 2.5) ** 2

    records: list[dict] = []
    for region in regionprops(labels, intensity_image=smooth):
        if region.area < min_area or region.area > max_core_area:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        if (
            min_row <= border_margin
            or min_col <= border_margin
            or max_row >= h - border_margin
            or max_col >= w - border_margin
        ):
            continue
        if region.eccentricity > 0.985:
            continue

        center_y, center_x = region.centroid_weighted
        detection_score = float(region.intensity_max)
        base_radius = math.sqrt(max(float(region.area), 1.0) / math.pi) * 1.5
        records.append(
            {
                "center_x": float(center_x),
                "center_y": float(center_y),
                "roi_radius": float(base_radius),
                "detection_sigma": float(max(min_sigma, base_radius / math.sqrt(2.0))),
                "detection_score": detection_score,
            }
        )

    return pd.DataFrame.from_records(records)


def _blob_candidates(
    detection_image: np.ndarray,
    min_sigma: float,
    max_sigma: float,
    num_sigma: int,
    threshold_rel: float,
    overlap: float,
    adaptive_roi_min_radius: float,
) -> pd.DataFrame:
    max_value = float(np.max(detection_image))
    if max_value <= 0:
        return pd.DataFrame(columns=["center_x", "center_y", "roi_radius", "detection_sigma", "detection_score"])

    robust_scale = float(np.percentile(detection_image, 99.9))
    robust_scale = max(robust_scale, max_value * 0.2, 1e-6)
    normalized = np.clip(detection_image / robust_scale, 0.0, 1.0)
    blobs = blob_log(
        normalized,
        min_sigma=max(float(min_sigma), 0.5),
        max_sigma=max(float(max_sigma), float(min_sigma) + 0.5),
        num_sigma=max(int(num_sigma), 3),
        threshold=max(float(threshold_rel) * 0.5, 0.02),
        overlap=float(np.clip(overlap, 0.0, 0.95)),
        exclude_border=False,
    )
    records: list[dict] = []
    for blob in blobs:
        center_y, center_x, sigma = [float(value) for value in blob[:3]]
        base_radius = max(math.sqrt(2.0) * sigma, adaptive_roi_min_radius)
        gx = int(np.clip(round(center_x), 0, detection_image.shape[1] - 1))
        gy = int(np.clip(round(center_y), 0, detection_image.shape[0] - 1))
        records.append(
            {
                "center_x": center_x,
                "center_y": center_y,
                "roi_radius": float(base_radius),
                "detection_sigma": sigma,
                "detection_score": float(detection_image[gy, gx]),
            }
        )
    return pd.DataFrame.from_records(records)


def _deduplicate_candidates(spots: pd.DataFrame, min_distance: float) -> pd.DataFrame:
    if spots.empty:
        return spots

    min_distance = max(float(min_distance), 1.0)
    ordered = spots.sort_values("detection_score", ascending=False).reset_index(drop=True)
    keep_indices: list[int] = []
    kept_centers: list[tuple[float, float]] = []
    for index, row in ordered.iterrows():
        center = (float(row["center_x"]), float(row["center_y"]))
        if any(math.hypot(center[0] - existing[0], center[1] - existing[1]) < min_distance for existing in kept_centers):
            continue
        keep_indices.append(index)
        kept_centers.append(center)
    return ordered.loc[keep_indices].reset_index(drop=True)


def _estimate_row_col(spots: pd.DataFrame, row_tolerance: float) -> pd.DataFrame:
    if spots.empty:
        return spots

    ordered = spots.sort_values(["center_y", "center_x"]).reset_index(drop=True).copy()
    row_tolerance = max(float(row_tolerance), 3.0)
    current_row = -1
    row_anchor = None
    row_assignments: list[int] = []

    for _, row in ordered.iterrows():
        center_y = float(row["center_y"])
        if row_anchor is None or abs(center_y - row_anchor) > row_tolerance:
            current_row += 1
            row_anchor = center_y
        else:
            row_anchor = (row_anchor + center_y) / 2.0
        row_assignments.append(current_row)

    ordered["row"] = row_assignments
    parts = []
    for _, row_df in ordered.groupby("row", sort=True):
        row_df = row_df.sort_values("center_x").reset_index(drop=True).copy()
        row_df["col"] = np.arange(len(row_df), dtype=int)
        parts.append(row_df)

    return pd.concat(parts, ignore_index=True)


def _component_around_seed(binary_patch: np.ndarray, seed_x: int, seed_y: int) -> np.ndarray | None:
    labels = label(binary_patch)
    if labels.size == 0:
        return None
    seed_y = int(np.clip(seed_y, 0, labels.shape[0] - 1))
    seed_x = int(np.clip(seed_x, 0, labels.shape[1] - 1))
    label_id = int(labels[seed_y, seed_x])
    if label_id == 0:
        return None
    return labels == label_id


def estimate_spot_geometry(
    oriented_image: np.ndarray,
    detection_image: np.ndarray,
    center_x: float,
    center_y: float,
    base_radius: float,
    threshold_rel: float,
    min_radius: float,
    max_radius: float,
) -> dict:
    max_radius = max(float(max_radius), float(min_radius) + 0.5)
    patch_radius = int(np.ceil(max(max_radius * 2.0, base_radius * 2.5, 8.0)))
    xmin = max(0, int(np.floor(center_x - patch_radius)))
    xmax = min(oriented_image.shape[1], int(np.ceil(center_x + patch_radius + 1)))
    ymin = max(0, int(np.floor(center_y - patch_radius)))
    ymax = min(oriented_image.shape[0], int(np.ceil(center_y + patch_radius + 1)))

    patch = np.asarray(detection_image[ymin:ymax, xmin:xmax], dtype=np.float32)
    if patch.size == 0:
        return {"center_x": float(center_x), "center_y": float(center_y), "roi_radius": float(base_radius)}

    local_x = float(center_x - xmin)
    local_y = float(center_y - ymin)
    local_xi = int(np.clip(round(local_x), 0, patch.shape[1] - 1))
    local_yi = int(np.clip(round(local_y), 0, patch.shape[0] - 1))

    patch_background = float(np.percentile(patch, 55))
    patch_peak = float(np.max(patch))
    threshold = patch_background + max(float(threshold_rel), 0.05) * max(patch_peak - patch_background, 0.0)
    binary = patch >= threshold
    component = _component_around_seed(binary, local_xi, local_yi)

    if component is None or int(np.sum(component)) < 6:
        refined_x = float(center_x)
        refined_y = float(center_y)
        roi_radius = float(np.clip(base_radius, min_radius, max_radius))
        return {"center_x": refined_x, "center_y": refined_y, "roi_radius": roi_radius}

    props = regionprops(component.astype(np.uint8))
    region = max(props, key=lambda prop: prop.area)
    refined_y = float(ymin + region.centroid[0])
    refined_x = float(xmin + region.centroid[1])
    roi_radius = math.sqrt(max(float(region.area), 1.0) / math.pi) * 1.2
    roi_radius = float(np.clip(roi_radius, min_radius, max_radius))
    return {"center_x": refined_x, "center_y": refined_y, "roi_radius": roi_radius}


def detect_spots(
    oriented_image: np.ndarray,
    background_sigma: float,
    min_sigma: float,
    max_sigma: float,
    num_sigma: int,
    threshold_rel: float,
    overlap: float,
    min_distance: float,
    adaptive_roi_enabled: bool,
    adaptive_roi_threshold_rel: float,
    adaptive_roi_min_radius: float,
    adaptive_roi_max_radius: float,
    crop_box: tuple[float, float, float, float] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    detection_image, background = make_detection_image(oriented_image, background_sigma=background_sigma)
    detection_image = apply_crop_to_image(detection_image, crop_box)
    if float(np.max(detection_image)) <= 0:
        return pd.DataFrame(columns=["spot_id", "row", "col", "sample", "dilution", "replicate", "exclude", "center_x", "center_y", "roi_radius", "detection_score"]), detection_image, background

    primary_candidates = _component_candidates(
        detection_image=detection_image,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold_rel=threshold_rel,
        min_distance=min_distance,
        adaptive_roi_min_radius=adaptive_roi_min_radius,
    )
    if len(primary_candidates) < 6:
        fallback_candidates = _blob_candidates(
            detection_image=detection_image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold_rel=threshold_rel,
            overlap=overlap,
            adaptive_roi_min_radius=adaptive_roi_min_radius,
        )
        candidate_df = pd.concat([primary_candidates, fallback_candidates], ignore_index=True)
    else:
        candidate_df = primary_candidates

    candidate_df = _deduplicate_candidates(candidate_df, min_distance=min_distance)
    if candidate_df.empty:
        return pd.DataFrame(columns=["spot_id", "row", "col", "sample", "dilution", "replicate", "exclude", "center_x", "center_y", "roi_radius", "detection_score"]), detection_image, background

    records: list[dict] = []
    for _, candidate in candidate_df.iterrows():
        center_x = float(candidate["center_x"])
        center_y = float(candidate["center_y"])
        base_radius = float(candidate["roi_radius"])
        geometry = (
            estimate_spot_geometry(
                oriented_image=oriented_image,
                detection_image=detection_image,
                center_x=center_x,
                center_y=center_y,
                base_radius=base_radius,
                threshold_rel=adaptive_roi_threshold_rel,
                min_radius=adaptive_roi_min_radius,
                max_radius=adaptive_roi_max_radius,
            )
            if adaptive_roi_enabled
            else {
                "center_x": center_x,
                "center_y": center_y,
                "roi_radius": float(np.clip(base_radius, adaptive_roi_min_radius, adaptive_roi_max_radius)),
            }
        )

        gx = int(np.clip(round(geometry["center_x"]), 0, detection_image.shape[1] - 1))
        gy = int(np.clip(round(geometry["center_y"]), 0, detection_image.shape[0] - 1))
        detection_score = float(detection_image[gy, gx])
        records.append(
            {
                "center_x": geometry["center_x"],
                "center_y": geometry["center_y"],
                "roi_radius": geometry["roi_radius"],
                "detection_sigma": float(candidate.get("detection_sigma", base_radius / math.sqrt(2.0))),
                "detection_score": detection_score,
            }
        )

    spots = pd.DataFrame.from_records(records)
    spots = _deduplicate_candidates(spots, min_distance=min_distance)
    if spots.empty:
        return pd.DataFrame(columns=["spot_id", "row", "col", "sample", "dilution", "replicate", "exclude", "center_x", "center_y", "roi_radius", "detection_score"]), detection_image, background

    row_tolerance = max(float(np.median(spots["roi_radius"])) * 1.8, min_distance)
    spots = _estimate_row_col(spots, row_tolerance=row_tolerance)
    spots["spot_id"] = [f"S{i + 1:03d}" for i in range(len(spots))]
    spots["sample"] = ""
    spots["dilution"] = ""
    spots["replicate"] = 1
    spots["exclude"] = False
    ordered_columns = [
        "spot_id",
        "row",
        "col",
        "sample",
        "dilution",
        "replicate",
        "exclude",
        "center_x",
        "center_y",
        "roi_radius",
        "detection_score",
        "detection_sigma",
    ]
    spots = spots.loc[:, ordered_columns].copy()
    return spots, detection_image, background
