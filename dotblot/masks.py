from __future__ import annotations

import numpy as np
from skimage.draw import disk, rectangle

from .types import GridConfig


def create_spot_mask(shape: tuple[int, int], center_x: float, center_y: float, radius: float, shape_name: str) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if shape_name == "circle":
        rr, cc = disk((center_y, center_x), radius=radius, shape=shape)
    elif shape_name == "square":
        start = (int(round(center_y - radius)), int(round(center_x - radius)))
        extent = (int(round(2 * radius)), int(round(2 * radius)))
        rr, cc = rectangle(start=start, extent=extent, shape=shape)
    else:
        raise ValueError(f"Unbekannte ROI-Form: {shape_name}")
    mask[rr, cc] = True
    return mask


def create_annulus_mask(shape: tuple[int, int], center_x: float, center_y: float, inner_radius: float, outer_radius: float) -> np.ndarray:
    outer_mask = create_spot_mask(shape, center_x, center_y, outer_radius, "circle")
    inner_mask = create_spot_mask(shape, center_x, center_y, inner_radius, "circle")
    return outer_mask & ~inner_mask


def estimate_expected_area(radius: float, shape_name: str) -> float:
    if shape_name == "circle":
        return float(np.pi * radius * radius)
    return float((2 * radius) ** 2)


def build_masks_for_spot(
    shape: tuple[int, int],
    center_x: float,
    center_y: float,
    roi_radius: float,
    roi_shape: str,
    annulus_inner_scale: float,
    annulus_outer_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    spot_mask = create_spot_mask(shape, center_x, center_y, roi_radius, roi_shape)
    annulus_mask = create_annulus_mask(
        shape,
        center_x,
        center_y,
        inner_radius=roi_radius * annulus_inner_scale,
        outer_radius=roi_radius * annulus_outer_scale,
    )
    annulus_mask &= ~spot_mask
    return spot_mask, annulus_mask


def build_masks(shape: tuple[int, int], center_x: float, center_y: float, config: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    return build_masks_for_spot(
        shape=shape,
        center_x=center_x,
        center_y=center_y,
        roi_radius=config.roi_radius,
        roi_shape=config.roi_shape,
        annulus_inner_scale=config.annulus_inner_scale,
        annulus_outer_scale=config.annulus_outer_scale,
    )
