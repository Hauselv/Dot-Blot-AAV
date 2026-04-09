from __future__ import annotations

from math import cos, radians, sin

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, gaussian_filter, uniform_filter
from scipy.signal import find_peaks

from .types import GridConfig


def suggest_grid_config(image_shape: tuple[int, ...], rows: int, cols: int) -> GridConfig:
    height, width = image_shape[:2]
    margin_x = width * 0.15
    margin_y = height * 0.15
    usable_width = max(width - 2 * margin_x, width * 0.5)
    usable_height = max(height - 2 * margin_y, height * 0.5)
    pitch_x = usable_width / max(cols - 1, 1)
    pitch_y = usable_height / max(rows - 1, 1)
    roi_radius = min(pitch_x, pitch_y) * 0.24

    return GridConfig(
        rows=rows,
        cols=cols,
        anchor_x=float(margin_x),
        anchor_y=float(margin_y),
        pitch_x=float(pitch_x),
        pitch_y=float(pitch_y),
        rotation_deg=0.0,
        roi_radius=float(max(6.0, roi_radius)),
        roi_shape="circle",
        annulus_inner_scale=1.35,
        annulus_outer_scale=2.15,
    )


def generate_grid_centers(config: GridConfig) -> np.ndarray:
    theta = radians(config.rotation_deg)
    col_step = np.array([config.pitch_x * cos(theta), config.pitch_x * sin(theta)], dtype=np.float32)
    row_step = np.array([-config.pitch_y * sin(theta), config.pitch_y * cos(theta)], dtype=np.float32)
    centers = []
    for row in range(config.rows):
        for col in range(config.cols):
            center = np.array([config.anchor_x, config.anchor_y], dtype=np.float32)
            center = center + col * col_step + row * row_step
            centers.append(center)
    return np.vstack(centers) if centers else np.zeros((0, 2), dtype=np.float32)


def _estimate_peak_positions(profile: np.ndarray, expected_count: int) -> np.ndarray:
    profile = np.asarray(profile, dtype=np.float32)
    n = profile.size
    if expected_count <= 0 or n == 0:
        return np.array([], dtype=np.float32)

    sigma = max(1.0, n / max(expected_count * 10, 8))
    smooth = gaussian_filter1d(profile, sigma=sigma)
    smooth = smooth - np.percentile(smooth, 25)
    smooth = np.clip(smooth, 0.0, None)

    min_distance = max(3, int(n / max(expected_count * 2, 2)))
    prominence = max(float(np.max(smooth)) * 0.1, 1e-6)
    peaks, properties = find_peaks(smooth, distance=min_distance, prominence=prominence)

    if peaks.size >= expected_count:
        order = np.argsort(properties["prominences"])[-expected_count:]
        selected = np.sort(peaks[order].astype(np.float32))
        return selected

    if peaks.size > 1:
        return np.sort(peaks.astype(np.float32))

    margin = n * 0.15
    return np.linspace(margin, n - margin, expected_count, dtype=np.float32)


def _regularize_positions(positions: np.ndarray, expected_count: int, axis_length: int) -> np.ndarray:
    positions = np.sort(np.asarray(positions, dtype=np.float32))
    if expected_count <= 0:
        return np.array([], dtype=np.float32)
    if positions.size == expected_count:
        return positions
    if positions.size <= 1:
        margin = axis_length * 0.15
        return np.linspace(margin, axis_length - margin, expected_count, dtype=np.float32)

    pitch = float(np.median(np.diff(positions)))
    pitch = max(pitch, 2.0)
    anchor = float(positions[0])

    if positions.size < expected_count:
        candidate = anchor + pitch * np.arange(expected_count, dtype=np.float32)
        if candidate[-1] > axis_length - 1:
            anchor = max(0.0, axis_length - 1 - pitch * (expected_count - 1))
            candidate = anchor + pitch * np.arange(expected_count, dtype=np.float32)
        return candidate

    best = positions[:expected_count]
    best_error = np.inf
    for start in range(0, positions.size - expected_count + 1):
        window = positions[start : start + expected_count]
        local_pitch = float(np.median(np.diff(window))) if expected_count > 1 else pitch
        local_pitch = max(local_pitch, 2.0)
        candidate = window[0] + local_pitch * np.arange(expected_count, dtype=np.float32)
        error = float(np.mean(np.abs(window - candidate)))
        if error < best_error:
            best_error = error
            best = candidate
    return best


def _template_response(image: np.ndarray, radius: float) -> np.ndarray:
    sigma_signal = max(1.0, radius / 2.5)
    sigma_background = max(sigma_signal * 2.5, sigma_signal + 2.0)
    signal = gaussian_filter(image.astype(np.float32), sigma=sigma_signal)
    background = gaussian_filter(signal, sigma=sigma_background)
    response = signal - background
    return np.clip(response, 0.0, None)


def auto_initialize_grid(image: np.ndarray, rows: int, cols: int, current_config: GridConfig | None = None) -> GridConfig:
    image = np.asarray(image, dtype=np.float32)
    base = suggest_grid_config(image.shape, rows, cols) if current_config is None else current_config
    signal = _template_response(image, radius=base.roi_radius)
    if not np.any(signal > 0):
        signal = np.clip(image - float(np.percentile(image, 40)), 0.0, None)
        signal = gaussian_filter(signal, sigma=1.0)

    col_profile = np.sum(signal, axis=0)
    row_profile = np.sum(signal, axis=1)
    x_positions = _regularize_positions(_estimate_peak_positions(col_profile, cols), cols, image.shape[1])
    y_positions = _regularize_positions(_estimate_peak_positions(row_profile, rows), rows, image.shape[0])

    pitch_x = float(np.median(np.diff(x_positions))) if x_positions.size > 1 else base.pitch_x
    pitch_y = float(np.median(np.diff(y_positions))) if y_positions.size > 1 else base.pitch_y
    roi_radius = min(pitch_x, pitch_y) * 0.24 if min(pitch_x, pitch_y) > 0 else base.roi_radius

    # A second pass anchors the regular grid to local evidence, which helps when
    # one or more spots are faint enough to disappear from the 1D peak profiles.
    provisional = GridConfig(
        rows=rows,
        cols=cols,
        anchor_x=float(x_positions[0]) if x_positions.size else base.anchor_x,
        anchor_y=float(y_positions[0]) if y_positions.size else base.anchor_y,
        pitch_x=float(max(2.0, pitch_x)),
        pitch_y=float(max(2.0, pitch_y)),
        rotation_deg=0.0 if current_config is None else current_config.rotation_deg,
        roi_radius=float(max(4.0, roi_radius if np.isfinite(roi_radius) else base.roi_radius)),
        roi_shape=base.roi_shape,
        annulus_inner_scale=base.annulus_inner_scale,
        annulus_outer_scale=base.annulus_outer_scale,
    )
    provisional_centers = generate_grid_centers(provisional)
    if provisional_centers.size:
        search_radius = max(
            provisional.roi_radius,
            min(provisional.pitch_x, provisional.pitch_y) * 0.35,
        )
        refined = refine_grid_centers(
            image,
            provisional_centers,
            roi_radius=provisional.roi_radius,
            search_radius=search_radius,
        ).reshape(rows, cols, 2)
        x_positions = _regularize_positions(np.median(refined[:, :, 0], axis=0), cols, image.shape[1])
        y_positions = _regularize_positions(np.median(refined[:, :, 1], axis=1), rows, image.shape[0])
        pitch_x = float(np.median(np.diff(x_positions))) if x_positions.size > 1 else provisional.pitch_x
        pitch_y = float(np.median(np.diff(y_positions))) if y_positions.size > 1 else provisional.pitch_y
        roi_radius = min(pitch_x, pitch_y) * 0.24 if min(pitch_x, pitch_y) > 0 else provisional.roi_radius

    return GridConfig(
        rows=rows,
        cols=cols,
        anchor_x=float(x_positions[0]) if x_positions.size else base.anchor_x,
        anchor_y=float(y_positions[0]) if y_positions.size else base.anchor_y,
        pitch_x=float(max(2.0, pitch_x)),
        pitch_y=float(max(2.0, pitch_y)),
        rotation_deg=0.0 if current_config is None else current_config.rotation_deg,
        roi_radius=float(max(4.0, roi_radius if np.isfinite(roi_radius) else base.roi_radius)),
        roi_shape=base.roi_shape,
        annulus_inner_scale=base.annulus_inner_scale,
        annulus_outer_scale=base.annulus_outer_scale,
    )


def refine_grid_centers(
    image: np.ndarray,
    centers: np.ndarray,
    roi_radius: float,
    search_radius: float,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    response_map = _template_response(image, radius=roi_radius)
    refined = []
    window_radius = int(np.ceil(max(roi_radius, search_radius) + 2))

    for center_x, center_y in centers:
        cx = float(center_x)
        cy = float(center_y)
        xmin = max(0, int(np.floor(cx - window_radius)))
        xmax = min(image.shape[1], int(np.ceil(cx + window_radius + 1)))
        ymin = max(0, int(np.floor(cy - window_radius)))
        ymax = min(image.shape[0], int(np.ceil(cy + window_radius + 1)))
        patch = image[ymin:ymax, xmin:xmax]
        response_patch = response_map[ymin:ymax, xmin:xmax]
        if patch.size == 0:
            refined.append([cx, cy])
            continue

        background = float(np.median(patch))
        signal = np.clip(patch - background, 0.0, None)
        signal = gaussian_filter(signal, sigma=max(1.0, roi_radius / 4.0))
        signal = 0.45 * signal + 0.55 * response_patch
        signal = uniform_filter(signal, size=max(2, int(np.ceil(roi_radius / 2.5))))

        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
        search_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= search_radius**2
        if not np.any(search_mask):
            refined.append([cx, cy])
            continue

        masked_signal = np.where(search_mask, signal, 0.0)
        peak_value = float(np.max(masked_signal))
        if peak_value <= 0:
            refined.append([cx, cy])
            continue

        threshold = max(peak_value * 0.7, float(np.percentile(masked_signal[search_mask], 85)))
        weights = np.where(masked_signal >= threshold, masked_signal, 0.0)
        total = float(np.sum(weights))
        if total <= 0:
            local_index = np.unravel_index(np.argmax(masked_signal), masked_signal.shape)
            refined_x = float(xmin + local_index[1])
            refined_y = float(ymin + local_index[0])
        else:
            refined_x = float(np.sum(xx * weights) / total)
            refined_y = float(np.sum(yy * weights) / total)
        refined.append([refined_x, refined_y])

    return np.asarray(refined, dtype=np.float32)


def make_default_metadata(rows: int, cols: int, preset: str) -> pd.DataFrame:
    records = []
    for row in range(rows):
        for col in range(cols):
            if preset == "Zeilen = Samples":
                sample = f"Sample_{row + 1}"
                dilution = str(col + 1)
            elif preset == "Spalten = Samples":
                sample = f"Sample_{col + 1}"
                dilution = str(row + 1)
            else:
                sample = f"Sample_{row + 1}"
                dilution = str(col + 1)

            records.append(
                {
                    "spot_id": f"R{row + 1:02d}C{col + 1:02d}",
                    "row": row,
                    "col": col,
                    "sample": sample,
                    "dilution": dilution,
                    "replicate": 1,
                    "exclude": False,
                }
            )
    return pd.DataFrame.from_records(records)


def ensure_metadata_shape(metadata: pd.DataFrame, rows: int, cols: int, preset: str) -> pd.DataFrame:
    expected = rows * cols
    if metadata is None or len(metadata) != expected:
        return make_default_metadata(rows, cols, preset)

    metadata = metadata.copy()
    required_columns = ["spot_id", "row", "col", "sample", "dilution", "replicate", "exclude"]
    defaults = make_default_metadata(rows, cols, preset)
    for column in required_columns:
        if column not in metadata.columns:
            metadata[column] = defaults[column]

    metadata = metadata[required_columns].copy()
    metadata["row"] = metadata["row"].astype(int)
    metadata["col"] = metadata["col"].astype(int)
    metadata["exclude"] = metadata["exclude"].astype(bool)
    return metadata
