from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.elements.image as st_image_module

from dotblot.analysis import auto_detect_spot_table, run_quantification
from dotblot.background import make_surface_corrected_preview
from dotblot.config import (
    APP_NAME,
    APP_VERSION,
    BACKGROUND_OPTIONS,
    LAYOUT_MODES,
    METADATA_PRESETS,
    POLARITY_OPTIONS,
    ROI_SHAPES,
    default_analysis_config,
)
from dotblot.export import analysis_manifest_bytes, dataframe_to_csv_bytes, figure_to_png_bytes
from dotblot.grid import (
    auto_initialize_grid,
    ensure_metadata_shape,
    generate_grid_centers,
    make_default_metadata,
    suggest_grid_config,
)
from dotblot.image_io import load_image
from dotblot.linear_range import (
    LINEAR_RANGE_X_OPTIONS,
    apply_manual_linear_selection,
    auto_select_linear_range,
)
from dotblot.plotting import (
    plot_dilution_series,
    plot_grid_heatmap,
    plot_linear_range_fits,
    plot_linear_summary_metric,
    plot_signal_by_spot,
)
from dotblot.preprocessing import normalize_for_display, orient_image
from dotblot.qc import create_qc_overlay, summarize_qc_flags
from dotblot.types import AnalysisConfig, GridConfig

try:
    from streamlit_image_coordinates import streamlit_image_coordinates

    HAS_IMAGE_COORDINATES = True
except ImportError:
    HAS_IMAGE_COORDINATES = False

try:
    if not hasattr(st_image_module, "image_to_url"):
        from streamlit.elements.lib.image_utils import image_to_url as _image_to_url

        st_image_module.image_to_url = _image_to_url
    from streamlit_drawable_canvas import st_canvas

    HAS_DRAWABLE_CANVAS = True
except ImportError:
    HAS_DRAWABLE_CANVAS = False


st.set_page_config(page_title=APP_NAME, layout="wide")


@st.cache_data(show_spinner=False)
def load_uploaded_image(file_bytes: bytes, filename: str):
    return load_image(file_bytes, filename)


@st.cache_data(show_spinner=False)
def load_sample_image(sample_path: str):
    path = Path(sample_path)
    return load_image(path.read_bytes(), path.name)


def available_sample_images() -> dict[str, str]:
    candidates = []
    for pattern in ("sample_data/*.png", "sample_data/*.jpg", "sample_data/*.jpeg", "sample_data/*.tif", "sample_data/*.tiff"):
        candidates.extend(Path(".").glob(pattern))
    for pattern in ("sample_data/external/*.png", "sample_data/external/*.jpg", "sample_data/external/*.jpeg", "sample_data/external/*.tif", "sample_data/external/*.tiff"):
        candidates.extend(Path(".").glob(pattern))
    for pattern in ("dotblottestdata/*.png", "dotblottestdata/*.jpg", "dotblottestdata/*.jpeg", "dotblottestdata/*.tif", "dotblottestdata/*.tiff"):
        candidates.extend(Path(".").glob(pattern))

    files = sorted({str(path.resolve()): path for path in candidates}.items(), key=lambda item: Path(item[0]).name.lower())
    return {
        (
            f"Lokale Testdaten: {Path(path_str).name}"
            if "dotblottestdata" in path_str.lower()
            else Path(path_str).name
        ): path_str
        for path_str, _ in files
    }


def recommend_detection_settings(gray_image, image_info) -> tuple[str, dict[str, float]]:
    gray_image = gray_image.astype("float32", copy=False)
    dynamic_range = float(image_info.intensity_max - image_info.intensity_min)
    high_dynamic = ("16" in image_info.dtype) or image_info.intensity_max > 4095
    clipped_threshold = image_info.intensity_max - max(dynamic_range * 0.01, 1.0)
    clipped_fraction = float((gray_image >= clipped_threshold).mean()) if dynamic_range > 0 else 0.0

    if high_dynamic:
        return (
            "Raw TIFF / hohe Dynamik",
            {
                "detection_background_sigma": 22.0,
                "detection_min_sigma": 1.2,
                "detection_max_sigma": 7.0,
                "detection_num_sigma": 12,
                "detection_threshold_rel": 0.16,
                "detection_overlap": 0.45,
                "detection_min_distance": 9.0,
                "adaptive_roi_min_radius": 2.5,
                "adaptive_roi_max_radius": 14.0,
                "adaptive_roi_threshold_rel": 0.24,
            },
        )
    if clipped_fraction > 0.08 or image_info.intensity_max <= 255:
        return (
            "Scan PNG / komprimierter Export",
            {
                "detection_background_sigma": 18.0,
                "detection_min_sigma": 1.2,
                "detection_max_sigma": 9.0,
                "detection_num_sigma": 12,
                "detection_threshold_rel": 0.08,
                "detection_overlap": 0.45,
                "detection_min_distance": 10.0,
                "adaptive_roi_min_radius": 4.0,
                "adaptive_roi_max_radius": 20.0,
                "adaptive_roi_threshold_rel": 0.22,
            },
        )
    return (
        "Allgemeiner Default",
        {
            "detection_background_sigma": 20.0,
            "detection_min_sigma": 1.5,
            "detection_max_sigma": 8.0,
            "detection_num_sigma": 12,
            "detection_threshold_rel": 0.12,
            "detection_overlap": 0.5,
            "detection_min_distance": 8.0,
            "adaptive_roi_min_radius": 3.0,
            "adaptive_roi_max_radius": 18.0,
            "adaptive_roi_threshold_rel": 0.28,
        },
    )


def apply_detection_recommendation(settings: dict[str, float]) -> None:
    for key, value in settings.items():
        st.session_state[key] = value


def default_crop_box(image_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    height, width = image_shape[:2]
    margin_x = width * 0.1
    margin_y = height * 0.1
    return (margin_x, margin_y, width - margin_x, height - margin_y)


def crop_box_to_canvas_payload(crop_box: tuple[float, float, float, float] | None, image_shape: tuple[int, int]) -> dict:
    box = crop_box or default_crop_box(image_shape)
    x0, y0, x1, y1 = [float(value) for value in box]
    left = min(x0, x1)
    top = min(y0, y1)
    width = max(abs(x1 - x0), 2.0)
    height = max(abs(y1 - y0), 2.0)
    return {
        "version": "4.4.0",
        "objects": [
            {
                "type": "rect",
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "fill": "rgba(0, 0, 0, 0)",
                "stroke": "#00bcd4",
                "strokeWidth": 2,
                "strokeUniform": True,
                "name": "detection_crop",
            }
        ],
    }


def crop_box_from_canvas_json(json_data: dict | None, image_shape: tuple[int, int]) -> tuple[float, float, float, float] | None:
    if not json_data:
        return None
    objects = json_data.get("objects") or []
    if not objects:
        return None

    rect = None
    for obj in objects:
        if obj.get("type") == "rect":
            rect = obj
            break
    if rect is None:
        return None

    left = float(rect.get("left", 0.0))
    top = float(rect.get("top", 0.0))
    width = float(rect.get("width", 0.0)) * float(rect.get("scaleX", 1.0))
    height = float(rect.get("height", 0.0)) * float(rect.get("scaleY", 1.0))
    x0 = max(0.0, left)
    y0 = max(0.0, top)
    x1 = min(float(image_shape[1]), x0 + max(width, 1.0))
    y1 = min(float(image_shape[0]), y0 + max(height, 1.0))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1, y1)


def ensure_canvas_spot_metadata(metadata: pd.DataFrame, roi_shape: str) -> pd.DataFrame:
    metadata = ensure_free_spot_metadata(metadata)
    if "roi_shape" not in metadata.columns:
        metadata["roi_shape"] = roi_shape
    return metadata


def spots_to_canvas_payload(metadata: pd.DataFrame) -> dict:
    objects = []
    for _, row in metadata.iterrows():
        radius = float(row.get("roi_radius", 6.0))
        center_x = float(row.get("center_x", 0.0))
        center_y = float(row.get("center_y", 0.0))
        exclude = bool(row.get("exclude", False))
        objects.append(
            {
                "type": "circle",
                "left": center_x - radius,
                "top": center_y - radius,
                "radius": radius,
                "fill": "rgba(0, 0, 0, 0)",
                "stroke": "#d62728" if exclude else "#00bcd4",
                "strokeWidth": 2,
                "strokeUniform": True,
                "name": str(row.get("spot_id", "")),
            }
        )
    return {"version": "4.4.0", "objects": objects}


def update_spots_from_canvas(metadata: pd.DataFrame, json_data: dict | None) -> pd.DataFrame:
    metadata = ensure_free_spot_metadata(metadata)
    if not json_data or not json_data.get("objects"):
        return metadata

    objects = [obj for obj in json_data.get("objects", []) if obj.get("type") == "circle"]
    updated = metadata.copy()
    for row_index, obj in enumerate(objects):
        if row_index >= len(updated):
            break
        radius = float(obj.get("radius", 0.0))
        scale_x = float(obj.get("scaleX", 1.0))
        scale_y = float(obj.get("scaleY", 1.0))
        effective_radius = max(radius * (scale_x + scale_y) * 0.5, 1.0)
        left = float(obj.get("left", 0.0))
        top = float(obj.get("top", 0.0))
        updated.loc[row_index, "center_x"] = left + effective_radius
        updated.loc[row_index, "center_y"] = top + effective_radius
        updated.loc[row_index, "roi_radius"] = effective_radius
    return ensure_free_spot_metadata(updated)


def empty_free_spot_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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
        ]
    )


def ensure_free_spot_metadata(metadata: pd.DataFrame | None) -> pd.DataFrame:
    required_columns = [
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
    ]
    if metadata is None or len(metadata) == 0:
        return empty_free_spot_metadata()

    metadata = pd.DataFrame(metadata).copy()
    for column in required_columns:
        if column not in metadata.columns:
            if column in {"sample", "dilution"}:
                metadata[column] = ""
            elif column == "replicate":
                metadata[column] = 1
            elif column == "exclude":
                metadata[column] = False
            else:
                metadata[column] = 0
    metadata = metadata[required_columns].copy()
    metadata["row"] = pd.to_numeric(metadata["row"], errors="coerce").fillna(0).astype(int)
    metadata["col"] = pd.to_numeric(metadata["col"], errors="coerce").fillna(0).astype(int)
    metadata["replicate"] = pd.to_numeric(metadata["replicate"], errors="coerce").fillna(1).astype(int)
    metadata["exclude"] = metadata["exclude"].astype(bool)
    metadata["center_x"] = pd.to_numeric(metadata["center_x"], errors="coerce").fillna(0.0).astype(float)
    metadata["center_y"] = pd.to_numeric(metadata["center_y"], errors="coerce").fillna(0.0).astype(float)
    metadata["roi_radius"] = pd.to_numeric(metadata["roi_radius"], errors="coerce").fillna(6.0).astype(float)
    metadata["spot_id"] = metadata["spot_id"].astype(str)
    return metadata


def apply_qc_click_action(
    metadata: pd.DataFrame,
    clicked: dict | None,
    action: str,
    selected_spot_id: str | None,
    default_radius: float,
) -> pd.DataFrame:
    metadata = ensure_free_spot_metadata(metadata)
    if not clicked:
        return metadata

    x = float(clicked["x"])
    y = float(clicked["y"])
    if action == "move" and selected_spot_id:
        mask = metadata["spot_id"].astype(str) == str(selected_spot_id)
        if mask.any():
            metadata.loc[mask, "center_x"] = x
            metadata.loc[mask, "center_y"] = y
        return metadata

    if action == "add":
        next_index = len(metadata) + 1
        new_row = pd.DataFrame(
            [
                {
                    "spot_id": f"S{next_index:03d}",
                    "row": int(metadata["row"].max() + 1) if not metadata.empty else 0,
                    "col": 0,
                    "sample": "",
                    "dilution": "",
                    "replicate": 1,
                    "exclude": False,
                    "center_x": x,
                    "center_y": y,
                    "roi_radius": float(default_radius),
                }
            ]
        )
        return pd.concat([metadata, new_row], ignore_index=True)

    return metadata


def initialize_state(image_hash: str, gray_image_shape: tuple[int, ...]) -> None:
    suggested = suggest_grid_config(gray_image_shape, 4, 6)
    if st.session_state.get("current_image_hash") != image_hash:
        st.session_state["current_image_hash"] = image_hash
        st.session_state["grid_anchor_x"] = suggested.anchor_x
        st.session_state["grid_anchor_y"] = suggested.anchor_y
        st.session_state["grid_pitch_x"] = suggested.pitch_x
        st.session_state["grid_pitch_y"] = suggested.pitch_y
        st.session_state["grid_rotation_deg"] = suggested.rotation_deg
        st.session_state["grid_roi_radius"] = suggested.roi_radius
        st.session_state["grid_annulus_inner_scale"] = suggested.annulus_inner_scale
        st.session_state["grid_annulus_outer_scale"] = suggested.annulus_outer_scale
        st.session_state["grid_rows"] = suggested.rows
        st.session_state["grid_cols"] = suggested.cols
        st.session_state["grid_roi_shape"] = suggested.roi_shape
        st.session_state["metadata_preset"] = METADATA_PRESETS[1]
        st.session_state["spot_metadata"] = make_default_metadata(
            suggested.rows,
            suggested.cols,
            st.session_state["metadata_preset"],
        )
        st.session_state["free_spot_metadata"] = empty_free_spot_metadata()
        st.session_state["last_qc_click_signature"] = ""
        st.session_state["detection_crop_box"] = default_crop_box(gray_image_shape)
        st.session_state["qc_canvas_version"] = 0
        st.session_state["last_anchor_click_signature"] = ""
        st.session_state.pop("linear_fit_selection", None)

    pending_anchor_x = st.session_state.pop("pending_grid_anchor_x", None)
    pending_anchor_y = st.session_state.pop("pending_grid_anchor_y", None)
    if pending_anchor_x is not None and pending_anchor_y is not None:
        st.session_state["grid_anchor_x"] = float(pending_anchor_x)
        st.session_state["grid_anchor_y"] = float(pending_anchor_y)

    st.session_state.setdefault("grid_anchor_x", suggested.anchor_x)
    st.session_state.setdefault("grid_anchor_y", suggested.anchor_y)
    st.session_state.setdefault("grid_pitch_x", suggested.pitch_x)
    st.session_state.setdefault("grid_pitch_y", suggested.pitch_y)
    st.session_state.setdefault("grid_rotation_deg", suggested.rotation_deg)
    st.session_state.setdefault("grid_roi_radius", suggested.roi_radius)
    st.session_state.setdefault("grid_annulus_inner_scale", suggested.annulus_inner_scale)
    st.session_state.setdefault("grid_annulus_outer_scale", suggested.annulus_outer_scale)
    st.session_state.setdefault("grid_rows", suggested.rows)
    st.session_state.setdefault("grid_cols", suggested.cols)
    st.session_state.setdefault("grid_roi_shape", suggested.roi_shape)
    st.session_state.setdefault("metadata_preset", METADATA_PRESETS[1])
    st.session_state.setdefault("spot_metadata", make_default_metadata(suggested.rows, suggested.cols, METADATA_PRESETS[1]))
    st.session_state.setdefault("free_spot_metadata", empty_free_spot_metadata())
    st.session_state.setdefault("detection_crop_box", default_crop_box(gray_image_shape))

    defaults = default_analysis_config()
    st.session_state.setdefault("layout_mode", defaults.layout_mode)
    st.session_state.setdefault("polarity_mode", defaults.polarity_mode)
    st.session_state.setdefault("background_mode", defaults.background_mode)
    st.session_state.setdefault("smooth_sigma", defaults.smooth_sigma)
    st.session_state.setdefault("saturation_fraction_threshold", defaults.saturation_fraction_threshold)
    st.session_state.setdefault("saturation_dynamic_fraction", defaults.saturation_dynamic_fraction)
    st.session_state.setdefault("weak_spot_snr_threshold", defaults.weak_spot_snr_threshold)
    st.session_state.setdefault("log_dilution_axis", defaults.log_dilution_axis)
    st.session_state.setdefault("center_refinement_enabled", defaults.center_refinement_enabled)
    st.session_state.setdefault("center_refinement_radius", defaults.center_refinement_radius)
    st.session_state.setdefault("linear_range_x_mode", defaults.linear_range_x_mode)
    st.session_state.setdefault("linear_range_min_points", defaults.linear_range_min_points)
    st.session_state.setdefault("linear_range_exclude_saturated", defaults.linear_range_exclude_saturated)
    st.session_state.setdefault("local_window_scale", defaults.local_window_scale)
    st.session_state.setdefault("surface_sigma", defaults.surface_sigma)
    st.session_state.setdefault("detection_background_sigma", defaults.detection_background_sigma)
    st.session_state.setdefault("detection_min_sigma", defaults.detection_min_sigma)
    st.session_state.setdefault("detection_max_sigma", defaults.detection_max_sigma)
    st.session_state.setdefault("detection_num_sigma", defaults.detection_num_sigma)
    st.session_state.setdefault("detection_threshold_rel", defaults.detection_threshold_rel)
    st.session_state.setdefault("detection_overlap", defaults.detection_overlap)
    st.session_state.setdefault("detection_min_distance", defaults.detection_min_distance)
    st.session_state.setdefault("detection_crop_enabled", defaults.detection_crop_enabled)
    st.session_state.setdefault("adaptive_roi_enabled", defaults.adaptive_roi_enabled)
    st.session_state.setdefault("adaptive_roi_threshold_rel", defaults.adaptive_roi_threshold_rel)
    st.session_state.setdefault("adaptive_roi_min_radius", defaults.adaptive_roi_min_radius)
    st.session_state.setdefault("adaptive_roi_max_radius", defaults.adaptive_roi_max_radius)
    st.session_state.setdefault("qc_click_action", "move")
    st.session_state.setdefault("qc_selected_spot_id", "")
    st.session_state.setdefault("last_qc_click_signature", "")
    st.session_state.setdefault("qc_canvas_version", 0)
    st.session_state.setdefault("last_anchor_click_signature", "")


def make_grid_config() -> GridConfig:
    return GridConfig(
        rows=int(st.session_state["grid_rows"]),
        cols=int(st.session_state["grid_cols"]),
        anchor_x=float(st.session_state["grid_anchor_x"]),
        anchor_y=float(st.session_state["grid_anchor_y"]),
        pitch_x=float(st.session_state["grid_pitch_x"]),
        pitch_y=float(st.session_state["grid_pitch_y"]),
        rotation_deg=float(st.session_state["grid_rotation_deg"]),
        roi_radius=float(st.session_state["grid_roi_radius"]),
        roi_shape=str(st.session_state["grid_roi_shape"]),
        annulus_inner_scale=float(st.session_state["grid_annulus_inner_scale"]),
        annulus_outer_scale=float(st.session_state["grid_annulus_outer_scale"]),
    )


def make_analysis_config(reference_sample: str | None) -> AnalysisConfig:
    defaults = default_analysis_config()
    return replace(
        defaults,
        layout_mode=str(st.session_state["layout_mode"]),
        polarity_mode=st.session_state["polarity_mode"],
        background_mode=st.session_state["background_mode"],
        smooth_sigma=float(st.session_state["smooth_sigma"]),
        saturation_fraction_threshold=float(st.session_state["saturation_fraction_threshold"]),
        saturation_dynamic_fraction=float(st.session_state["saturation_dynamic_fraction"]),
        weak_spot_snr_threshold=float(st.session_state["weak_spot_snr_threshold"]),
        reference_sample=reference_sample,
        log_dilution_axis=bool(st.session_state["log_dilution_axis"]),
        center_refinement_enabled=bool(st.session_state["center_refinement_enabled"]),
        center_refinement_radius=float(st.session_state["center_refinement_radius"]),
        linear_range_x_mode=str(st.session_state["linear_range_x_mode"]),
        linear_range_min_points=int(st.session_state["linear_range_min_points"]),
        linear_range_exclude_saturated=bool(st.session_state["linear_range_exclude_saturated"]),
        local_window_scale=float(st.session_state["local_window_scale"]),
        surface_sigma=float(st.session_state["surface_sigma"]),
        detection_background_sigma=float(st.session_state["detection_background_sigma"]),
        detection_min_sigma=float(st.session_state["detection_min_sigma"]),
        detection_max_sigma=float(st.session_state["detection_max_sigma"]),
        detection_num_sigma=int(st.session_state["detection_num_sigma"]),
        detection_threshold_rel=float(st.session_state["detection_threshold_rel"]),
        detection_overlap=float(st.session_state["detection_overlap"]),
        detection_min_distance=float(st.session_state["detection_min_distance"]),
        detection_crop_enabled=bool(st.session_state["detection_crop_enabled"]),
        detection_crop_box=tuple(st.session_state["detection_crop_box"]) if st.session_state.get("detection_crop_box") else None,
        adaptive_roi_enabled=bool(st.session_state["adaptive_roi_enabled"]),
        adaptive_roi_threshold_rel=float(st.session_state["adaptive_roi_threshold_rel"]),
        adaptive_roi_min_radius=float(st.session_state["adaptive_roi_min_radius"]),
        adaptive_roi_max_radius=float(st.session_state["adaptive_roi_max_radius"]),
    )


def apply_grid_config_to_state(config: GridConfig) -> None:
    st.session_state["grid_anchor_x"] = float(config.anchor_x)
    st.session_state["grid_anchor_y"] = float(config.anchor_y)
    st.session_state["grid_pitch_x"] = float(config.pitch_x)
    st.session_state["grid_pitch_y"] = float(config.pitch_y)
    st.session_state["grid_rotation_deg"] = float(config.rotation_deg)
    st.session_state["grid_roi_radius"] = float(config.roi_radius)
    st.session_state["grid_roi_shape"] = str(config.roi_shape)
    st.session_state["grid_annulus_inner_scale"] = float(config.annulus_inner_scale)
    st.session_state["grid_annulus_outer_scale"] = float(config.annulus_outer_scale)


def merge_linear_selection(auto_results: pd.DataFrame, force_reset: bool) -> pd.DataFrame:
    auto_selection = auto_results[["spot_id", "linear_fit_include"]].copy()
    current = st.session_state.get("linear_fit_selection")
    if force_reset or current is None:
        st.session_state["linear_fit_selection"] = auto_selection
        return auto_results

    current = pd.DataFrame(current).copy()
    if set(current["spot_id"].astype(str)) != set(auto_selection["spot_id"].astype(str)):
        st.session_state["linear_fit_selection"] = auto_selection
        return auto_results

    selection_map = current.set_index("spot_id")["linear_fit_include"].astype(bool).to_dict()
    merged = auto_results.copy()
    merged["linear_fit_include"] = (
        merged["spot_id"].map(selection_map).fillna(merged["linear_fit_auto_include"]).astype(bool)
    )
    st.session_state["linear_fit_selection"] = merged[["spot_id", "linear_fit_include"]].copy()
    return merged


def render_anchor_picker(display_image_pil: Image.Image) -> None:
    st.markdown("**Ankerpunkt im Bild setzen**")
    if HAS_IMAGE_COORDINATES:
        clicked = streamlit_image_coordinates(
            display_image_pil,
            key="anchor_picker",
            width=min(display_image_pil.width, 900),
        )
        if clicked:
            click_signature = f"{float(clicked['x']):.1f},{float(clicked['y']):.1f}"
            if click_signature != st.session_state.get("last_anchor_click_signature", ""):
                st.session_state["pending_grid_anchor_x"] = float(clicked["x"])
                st.session_state["pending_grid_anchor_y"] = float(clicked["y"])
                st.session_state["last_anchor_click_signature"] = click_signature
                st.rerun()
        st.caption("Klick auf das Bild setzt den Mittelpunkt des ersten Spots (oben links).")
    else:
        st.info(
            "`streamlit-image-coordinates` ist nicht installiert. "
            "Der MVP funktioniert weiterhin, der Ankerpunkt wird dann numerisch gesetzt."
        )
        st.image(display_image_pil, caption="Grid-Referenzbild", use_container_width=True)


def render_help_tab() -> None:
    st.markdown("**Kurzstart**")
    st.markdown(
        "\n".join(
            [
                "1. Bild laden oder Beispielbild waehlen.",
                "2. Polarity im Tab `Preprocessing` pruefen.",
                "3. Im Tab `Grid Setup` entweder ein regulaeres Grid initialisieren oder freie Spots automatisch erkennen.",
                "4. Spot-Metadaten bearbeiten und problematische Spots ueber `exclude` markieren.",
                "5. Background-Methode und optionale Spot-Verfeinerung in `Quantification` einstellen.",
                "6. Im `QC`-Tab Overlay und Flags pruefen.",
                "7. Im `Linear Range`-Tab den vorgeschlagenen Fit kontrollieren und bei Bedarf manuell anpassen.",
                "8. Ergebnisse und Exporte im Anschluss speichern.",
            ]
        )
    )

    with st.expander("Empfohlene Defaults", expanded=True):
        st.markdown(
            "\n".join(
                [
                    "- Polarity: `Auto`",
                    "- Background: `local_annulus_median`",
                    "- Spot-Zentren lokal verfeinern: aktiviert",
                    "- Linearer Bereich: Auto-Vorschlag als Startpunkt",
                    "- Normalisierung: Referenzsample pro Verdunnungsstufe, falls vorhanden",
                ]
            )
        )

    with st.expander("Wie die automatische Spot-Erkennung aktuell arbeitet"):
        st.markdown(
            "\n".join(
                [
                    "Die App unterstuetzt jetzt zwei Wege:",
                    "",
                    "- regulaeres Grid: sinnvoll, wenn das Layout gut organisiert ist",
                    "- freie Spot-Detektion: sinnvoll, wenn Spotgroessen und Abstaende variieren",
                    "",
                    "Im freien Modus werden:",
                    "- ein background-subtrahiertes Detection-Bild berechnet",
                    "- die Suche optional auf eine gezeichnete Begrenzungs-Box eingeschraenkt",
                    "- blob-aehnliche Kandidaten gesucht",
                    "- nahe Duplikate zusammengefasst",
                    "- lokale ROI-Radien adaptiv aus dem lokalen Signal geschaetzt",
                    "- Spots im QC-Bereich interaktiv korrigierbar gemacht",
                ]
            )
        )

    with st.expander("Background-Methoden"):
        st.markdown(
            "\n".join(
                [
                    "- `local_annulus_median`: guter Default fuer viele Membranen",
                    "- `local_annulus_plane`: gut bei lokalem Gradient um den Spot",
                    "- `local_window_median`: nuetzlich, wenn benachbarte Spots den Annulus stoeren",
                    "- `local_window_plane`: gut bei lokalem Gradient und unruhigem Umfeld",
                    "- `surface_median`: sinnvoll bei breitem Hintergrundgradienten",
                    "- `global_*`: nur bei sehr homogenem Hintergrund empfehlenswert",
                ]
            )
        )

    with st.expander("QC und Troubleshooting"):
        st.markdown(
            "\n".join(
                [
                    "- Gelb markierte Spots sind schwach.",
                    "- Orange markierte Spots sind potenziell gesaettigt.",
                    "- Rot markierte Spots sind ausgeschlossen.",
                    "- Negative korrigierte Werte werden bewusst nicht auf 0 gesetzt.",
                    "",
                    "Wenn schwache Spots schlecht getroffen werden:",
                    "- zuerst den Modus `Freie Spot-Detektion` probieren",
                    "- eine Begrenzungs-Box nur ueber den eigentlichen Blot ziehen",
                    "- fuer PNG-Scans bzw. Raw-TIFFs den empfohlenen Parameter-Startwert anwenden",
                    "- `Detection background sigma` und `Detection threshold rel` anpassen",
                    "- `Spot-Zentren lokal verfeinern` aktiviert lassen",
                    "- adaptive ROI-Grenzen pruefen",
                    "- `local_window_median` gegen `local_annulus_median` vergleichen",
                    "- Polarity manuell pruefen",
                ]
            )
        )

    st.info(
        "Ausfuehrlichere Dokumentation liegt im Repo in `README.md` und `docs/USER_GUIDE.md`."
    )


def main() -> None:
    st.title(APP_NAME)
    st.caption(f"Version {APP_VERSION} | Lokale, reproduzierbare Dot-Blot-Quantifizierung")

    sample_images = available_sample_images()
    source_col1, source_col2 = st.columns([1.4, 1])
    with source_col1:
        uploaded = st.file_uploader("Dot-Blot-Bild laden", type=["tif", "tiff", "png", "jpg", "jpeg"])
    with source_col2:
        sample_label = st.selectbox(
            "Oder Beispielbild laden",
            options=["Keins"] + list(sample_images.keys()),
            help="Nuetzlich zum Testen der App, solange noch keine eigenen Blot-Bilder vorliegen.",
        )

    if uploaded is None and sample_label == "Keins":
        st.info("Ein Bild hochladen oder ein Beispielbild auswaehlen, um die Analyse zu starten.")
        return

    if uploaded is not None:
        raw_image, gray_image, image_info = load_uploaded_image(uploaded.getvalue(), uploaded.name)
        image_source_label = f"Upload: {uploaded.name}"
    else:
        raw_image, gray_image, image_info = load_sample_image(sample_images[sample_label])
        image_source_label = f"Beispielbild: {sample_label}"

    initialize_state(image_info.file_hash, gray_image.shape)
    recommended_detection_label, recommended_detection_settings = recommend_detection_settings(gray_image, image_info)

    display_image = normalize_for_display(gray_image)
    display_image_pil = Image.fromarray((display_image * 255).astype("uint8"))

    tabs = st.tabs(["Upload", "Preprocessing", "Grid Setup", "Quantification", "QC", "Linear Range", "Results", "Export", "Help"])

    with tabs[0]:
        col1, col2 = st.columns([1.4, 1])
        with col1:
            st.image(display_image, caption="Bildvorschau", clamp=True, use_container_width=True)
        with col2:
            info_table = pd.DataFrame(
                [
                    {"Eigenschaft": "Quelle", "Wert": image_source_label},
                    {"Eigenschaft": "Datei", "Wert": image_info.filename},
                    {"Eigenschaft": "Shape", "Wert": str(image_info.shape)},
                    {"Eigenschaft": "Datentyp", "Wert": image_info.dtype},
                    {"Eigenschaft": "Intensitaetsminimum", "Wert": image_info.intensity_min},
                    {"Eigenschaft": "Intensitaetsmaximum", "Wert": image_info.intensity_max},
                    {"Eigenschaft": "Farbbild", "Wert": image_info.is_color},
                ]
            )
            st.dataframe(info_table, use_container_width=True, hide_index=True)

    with tabs[1]:
        control_col, preview_col = st.columns([1, 1.2])
        with control_col:
            st.selectbox("Polarity", POLARITY_OPTIONS, key="polarity_mode")
            st.caption("Die Analyse arbeitet intern immer auf einem orientierten Bild mit positivem Signal.")
        oriented_image, polarity_meta = orient_image(
            gray_image,
            st.session_state["polarity_mode"],
            sigma_small=float(st.session_state["smooth_sigma"]),
        )
        with preview_col:
            st.image(normalize_for_display(oriented_image), caption="Orientiertes Analysebild", clamp=True, use_container_width=True)
        st.dataframe(
            pd.DataFrame(
                {
                    "Parameter": ["Auto erkannt", "Confidence", "Invertierung angewendet", "Effektiver Modus"],
                    "Wert": [
                        polarity_meta["auto_detected_polarity"],
                        round(polarity_meta["polarity_confidence"], 3),
                        polarity_meta["invert_applied"],
                        polarity_meta["effective_mode"],
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[2]:
        left, right = st.columns([1, 1.4])
        with left:
            st.selectbox(
                "Layout-Modus",
                options=list(LAYOUT_MODES.keys()),
                format_func=lambda value: LAYOUT_MODES[value],
                key="layout_mode",
            )
            st.number_input("ROI-Radius (Default)", min_value=2.0, step=1.0, key="grid_roi_radius")
            st.selectbox("ROI-Form", ROI_SHAPES, key="grid_roi_shape")

            if st.session_state["layout_mode"] == "grid":
                st.number_input("Zeilen", min_value=1, max_value=24, step=1, key="grid_rows")
                st.number_input("Spalten", min_value=1, max_value=24, step=1, key="grid_cols")
                st.selectbox("Metadaten-Preset", METADATA_PRESETS, key="metadata_preset")
                if st.button("Grid automatisch initialisieren", use_container_width=True):
                    auto_grid = auto_initialize_grid(
                        oriented_image,
                        rows=int(st.session_state["grid_rows"]),
                        cols=int(st.session_state["grid_cols"]),
                        current_config=make_grid_config(),
                    )
                    apply_grid_config_to_state(auto_grid)
                if st.button("Metadaten aus Preset neu erzeugen", use_container_width=True):
                    st.session_state["spot_metadata"] = make_default_metadata(
                        int(st.session_state["grid_rows"]),
                        int(st.session_state["grid_cols"]),
                        st.session_state["metadata_preset"],
                    )
                st.number_input("Anker X", min_value=0.0, step=1.0, key="grid_anchor_x")
                st.number_input("Anker Y", min_value=0.0, step=1.0, key="grid_anchor_y")
                st.number_input("Abstand X", min_value=1.0, step=1.0, key="grid_pitch_x")
                st.number_input("Abstand Y", min_value=1.0, step=1.0, key="grid_pitch_y")
                st.number_input("Rotation (Grad)", min_value=-45.0, max_value=45.0, step=0.1, key="grid_rotation_deg")
            else:
                st.caption(f"Empfohlener Startwert fuer dieses Bild: {recommended_detection_label}")
                if st.button("Empfohlene Detektionsparameter anwenden", use_container_width=True):
                    apply_detection_recommendation(recommended_detection_settings)
                st.checkbox("Detektion auf Begrenzungs-Box beschraenken", key="detection_crop_enabled")
                if st.button("Begrenzungs-Box zuruecksetzen", use_container_width=True):
                    st.session_state["detection_crop_box"] = default_crop_box(gray_image.shape)
                st.number_input("Detection background sigma", min_value=1.0, max_value=200.0, step=1.0, key="detection_background_sigma")
                st.number_input("Detection min sigma", min_value=0.5, max_value=20.0, step=0.1, key="detection_min_sigma")
                st.number_input("Detection max sigma", min_value=1.0, max_value=40.0, step=0.1, key="detection_max_sigma")
                st.number_input("Detection num sigma", min_value=3, max_value=30, step=1, key="detection_num_sigma")
                st.number_input("Detection threshold rel", min_value=0.01, max_value=0.9, step=0.01, key="detection_threshold_rel")
                st.number_input("Detection overlap", min_value=0.0, max_value=0.95, step=0.05, key="detection_overlap")
                st.number_input("Detection min distance", min_value=1.0, max_value=100.0, step=1.0, key="detection_min_distance")
                st.checkbox("Adaptive ROI aktiv", key="adaptive_roi_enabled")
                st.number_input("Adaptive ROI threshold rel", min_value=0.05, max_value=0.95, step=0.01, key="adaptive_roi_threshold_rel")
                st.number_input("Adaptive ROI min radius", min_value=1.0, max_value=50.0, step=0.5, key="adaptive_roi_min_radius")
                st.number_input("Adaptive ROI max radius", min_value=2.0, max_value=100.0, step=0.5, key="adaptive_roi_max_radius")
                preview_detection_config = make_analysis_config(reference_sample=None)
                preview_detection_config.auto_detected_polarity = polarity_meta["auto_detected_polarity"]
                preview_detection_config.invert_applied = polarity_meta["invert_applied"]
                preview_detection_config.polarity_confidence = polarity_meta["polarity_confidence"]
                if st.button("Spots automatisch erkennen", use_container_width=True):
                    detected_spots, _, _ = auto_detect_spot_table(oriented_image, preview_detection_config)
                    st.session_state["free_spot_metadata"] = detected_spots
        with right:
            if st.session_state["layout_mode"] == "grid":
                render_anchor_picker(display_image_pil)
            else:
                if st.session_state["detection_crop_enabled"]:
                    st.markdown("**Detektionsbereich**")
                    if HAS_DRAWABLE_CANVAS:
                        crop_canvas = st_canvas(
                            fill_color="rgba(0, 0, 0, 0)",
                            stroke_width=2,
                            stroke_color="#00bcd4",
                            background_image=display_image_pil,
                            update_streamlit=True,
                            height=display_image_pil.height,
                            width=display_image_pil.width,
                            drawing_mode="rect",
                            initial_drawing=crop_box_to_canvas_payload(
                                st.session_state.get("detection_crop_box"),
                                gray_image.shape,
                            ),
                            key=f"detection_crop_canvas_{image_info.file_hash}",
                        )
                        crop_box = crop_box_from_canvas_json(crop_canvas.json_data, gray_image.shape)
                        if crop_box is not None:
                            st.session_state["detection_crop_box"] = crop_box
                        if st.session_state.get("detection_crop_box"):
                            x0, y0, x1, y1 = st.session_state["detection_crop_box"]
                            st.caption(f"Aktive Box: x={x0:.0f}-{x1:.0f}, y={y0:.0f}-{y1:.0f}")
                    else:
                        st.info("Fuer die Zieh-Box wird `streamlit-drawable-canvas` benoetigt.")

                preview_detection_config = make_analysis_config(reference_sample=None)
                detected_preview, detection_preview, background_preview = auto_detect_spot_table(
                    oriented_image,
                    preview_detection_config,
                )
                preview_cols = st.columns(3)
                with preview_cols[0]:
                    st.image(
                        normalize_for_display(oriented_image),
                        caption="Orientiertes Bild",
                        clamp=True,
                        use_container_width=True,
                    )
                with preview_cols[1]:
                    st.image(
                        normalize_for_display(background_preview),
                        caption="Geschaetzter Detection-Background",
                        clamp=True,
                        use_container_width=True,
                    )
                with preview_cols[2]:
                    st.image(
                        normalize_for_display(detection_preview),
                        caption=f"Detection-Bild ({len(detected_preview)} Kandidaten)",
                        clamp=True,
                        use_container_width=True,
                    )
                st.caption("Die Spot-Detektion arbeitet auf dem background-subtrahierten Detection-Bild, quantifiziert wird spaeter wieder auf dem Signalbild.")

        current_grid = make_grid_config()
        if st.session_state["layout_mode"] == "grid":
            metadata = ensure_metadata_shape(
                st.session_state.get("spot_metadata"),
                current_grid.rows,
                current_grid.cols,
                st.session_state["metadata_preset"],
            )
            centers = generate_grid_centers(current_grid)
            preview_df = metadata.copy()
            preview_df["center_x"] = centers[:, 0]
            preview_df["center_y"] = centers[:, 1]
            st.markdown("**Spot-Metadaten**")
            edited_metadata = st.data_editor(
                preview_df.drop(columns=["center_x", "center_y"]),
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key="metadata_editor",
            )
            st.session_state["spot_metadata"] = pd.DataFrame(edited_metadata)
        else:
            free_metadata = ensure_free_spot_metadata(st.session_state.get("free_spot_metadata"))
            st.markdown("**Gefundene Spots und Metadaten**")
            edited_free_metadata = st.data_editor(
                free_metadata,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                key="free_metadata_editor",
            )
            st.session_state["free_spot_metadata"] = ensure_free_spot_metadata(pd.DataFrame(edited_free_metadata))

    with tabs[3]:
        quant_background_preview, quant_corrected_preview = make_surface_corrected_preview(
            oriented_image,
            sigma=float(st.session_state["surface_sigma"]),
        )
        st.markdown("**Background-Vorschau**")
        bg_preview_cols = st.columns(3)
        with bg_preview_cols[0]:
            st.image(
                normalize_for_display(oriented_image),
                caption="Signalbild fuer Quantifizierung",
                clamp=True,
                use_container_width=True,
            )
        with bg_preview_cols[1]:
            st.image(
                normalize_for_display(quant_background_preview),
                caption="Surface-Background Preview",
                clamp=True,
                use_container_width=True,
            )
        with bg_preview_cols[2]:
            st.image(
                normalize_for_display(quant_corrected_preview),
                caption="Surface-korrigierte Vorschau",
                clamp=True,
                use_container_width=True,
            )
        st.caption(
            "Diese Vorschau zeigt eine globale Surface-Subtraktion als QC-Hilfe. "
            "Die eigentliche Quantifizierung nutzt weiterhin die pro Spot gewaehlte Background-Methode."
        )

        left, right = st.columns(2)
        with left:
            st.selectbox(
                "Background-Methode",
                options=list(BACKGROUND_OPTIONS.keys()),
                format_func=lambda value: BACKGROUND_OPTIONS[value],
                key="background_mode",
            )
            st.number_input("Glattung sigma", min_value=0.0, step=0.1, key="smooth_sigma")
            st.number_input(
                "Saturation fraction threshold",
                min_value=0.01,
                max_value=1.0,
                step=0.01,
                key="saturation_fraction_threshold",
            )
            st.number_input(
                "Saturation dynamic fraction",
                min_value=0.5,
                max_value=0.9999,
                step=0.001,
                format="%.4f",
                key="saturation_dynamic_fraction",
            )
            st.number_input(
                "Surface sigma",
                min_value=1.0,
                max_value=200.0,
                step=1.0,
                key="surface_sigma",
            )
        with right:
            st.number_input(
                "Annulus inner scale",
                min_value=1.01,
                max_value=5.0,
                step=0.05,
                key="grid_annulus_inner_scale",
            )
            st.number_input(
                "Annulus outer scale",
                min_value=1.1,
                max_value=6.0,
                step=0.05,
                key="grid_annulus_outer_scale",
            )
            st.number_input(
                "Weak-Spot SNR threshold",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                key="weak_spot_snr_threshold",
            )
            st.number_input(
                "Lokales Fenster (Scale x ROI)",
                min_value=1.5,
                max_value=10.0,
                step=0.1,
                key="local_window_scale",
            )
            st.checkbox("Spot-Zentren lokal verfeinern", key="center_refinement_enabled")
            st.number_input(
                "Suchradius fuer Verfeinerung",
                min_value=0.0,
                max_value=30.0,
                step=1.0,
                key="center_refinement_radius",
            )
            st.checkbox("Log-Skalierung fuer Verdunnungsplot", key="log_dilution_axis")

    with tabs[5]:
        left, right = st.columns(2)
        with left:
            st.selectbox(
                "X-Achse fuer linearen Bereich",
                options=list(LINEAR_RANGE_X_OPTIONS.keys()),
                format_func=lambda value: LINEAR_RANGE_X_OPTIONS[value],
                key="linear_range_x_mode",
            )
            st.number_input(
                "Minimale Punkte pro linearem Bereich",
                min_value=2,
                max_value=12,
                step=1,
                key="linear_range_min_points",
            )
        with right:
            st.checkbox(
                "Gesattigte Spots im Auto-Vorschlag ausschliessen",
                key="linear_range_exclude_saturated",
            )
            reset_linear_selection = st.button("Auto-Vorschlag uebernehmen", use_container_width=True)

    if st.session_state["layout_mode"] == "grid":
        metadata = ensure_metadata_shape(
            st.session_state.get("spot_metadata"),
            int(st.session_state["grid_rows"]),
            int(st.session_state["grid_cols"]),
            st.session_state["metadata_preset"],
        )
    else:
        metadata = ensure_free_spot_metadata(st.session_state.get("free_spot_metadata"))
    sample_options = sorted([sample for sample in metadata["sample"].astype(str).unique() if sample])

    if st.session_state["layout_mode"] == "free_spots" and metadata.empty:
        with tabs[4]:
            st.info("Im Modus `Freie Spot-Detektion` zuerst `Spots automatisch erkennen` ausfuehren oder Spots manuell hinzufuegen.")
        with tabs[8]:
            render_help_tab()
        return

    with tabs[3]:
        reference_sample = st.selectbox("Referenzsample", options=[""] + sample_options)
        if reference_sample == "":
            reference_sample = None

    analysis_config = make_analysis_config(reference_sample)
    analysis_config.auto_detected_polarity = polarity_meta["auto_detected_polarity"]
    analysis_config.invert_applied = polarity_meta["invert_applied"]
    analysis_config.polarity_confidence = polarity_meta["polarity_confidence"]

    grid_config = make_grid_config()
    results = run_quantification(oriented_image, grid_config, metadata, analysis_config)
    auto_linear_results, auto_linear_summary = auto_select_linear_range(
        results,
        value_column="corrected_integrated_intensity",
        x_mode=analysis_config.linear_range_x_mode,
        min_points=analysis_config.linear_range_min_points,
        exclude_saturated=analysis_config.linear_range_exclude_saturated,
    )
    linear_results = merge_linear_selection(auto_linear_results, force_reset=reset_linear_selection)
    results, linear_summary = apply_manual_linear_selection(
        linear_results,
        value_column="corrected_integrated_intensity",
        x_mode=analysis_config.linear_range_x_mode,
        min_points=analysis_config.linear_range_min_points,
    )

    qc_figure = create_qc_overlay(gray_image, results, grid_config)
    raw_plot = plot_signal_by_spot(results, "raw_integrated_intensity", "Rohsignale pro Spot")
    background_plot = plot_signal_by_spot(results, "background_value", "Background-Werte pro Spot")
    corrected_plot = plot_signal_by_spot(results, "corrected_integrated_intensity", "Background-korrigierte Signale")
    background_fit_plot = None
    if "background_fit_rmse" in results.columns and results["background_fit_rmse"].notna().any():
        background_fit_plot = plot_signal_by_spot(results, "background_fit_rmse", "Background-Fit RMSE pro Spot")
    dilution_plot = plot_dilution_series(
        results,
        "corrected_integrated_intensity",
        log_x=analysis_config.log_dilution_axis,
    )
    heatmap_plot = plot_grid_heatmap(results, "corrected_integrated_intensity")
    linear_fit_plot = plot_linear_range_fits(results, linear_summary, "corrected_integrated_intensity")
    linear_slope_plot = plot_linear_summary_metric(linear_summary, "slope", "Steigung pro Sample im linearen Bereich")
    normalized_plot = None
    if results["normalized_signal"].notna().any():
        normalized_plot = plot_signal_by_spot(results, "normalized_signal", "Normalisierte Signale")

    with tabs[2]:
        st.markdown("**ROI-Overlay Vorschau**")
        st.pyplot(qc_figure, use_container_width=True)

    with tabs[4]:
        if analysis_config.layout_mode == "free_spots":
            st.markdown("**Interaktive QC-Korrektur**")
            qc_left, qc_right = st.columns([1, 1.25])
            qc_canvas_result = None
            with qc_right:
                if HAS_DRAWABLE_CANVAS:
                    qc_canvas_result = st_canvas(
                        fill_color="rgba(0, 0, 0, 0)",
                        stroke_width=2,
                        background_image=display_image_pil,
                        update_streamlit=True,
                        height=display_image_pil.height,
                        width=display_image_pil.width,
                        drawing_mode="transform",
                        initial_drawing=spots_to_canvas_payload(metadata),
                        key=f"qc_canvas_{image_info.file_hash}_{st.session_state.get('qc_canvas_version', 0)}",
                    )
                    st.caption("Kreise koennen direkt verschoben und in der Groesse angepasst werden. Danach links `Canvas-Aenderungen uebernehmen` klicken.")
                else:
                    st.pyplot(qc_figure, use_container_width=True)

            with qc_left:
                spot_ids = metadata["spot_id"].astype(str).tolist()
                selected_spot_id = st.selectbox(
                    "Ausgewaehlter Spot",
                    options=[""] + spot_ids,
                    key="qc_selected_spot_id",
                )
                if HAS_DRAWABLE_CANVAS:
                    if st.button("Canvas-Aenderungen uebernehmen", use_container_width=True):
                        updated_metadata = update_spots_from_canvas(metadata, qc_canvas_result.json_data if qc_canvas_result else None)
                        st.session_state["free_spot_metadata"] = updated_metadata
                        st.session_state["qc_canvas_version"] = int(st.session_state.get("qc_canvas_version", 0)) + 1
                        st.rerun()
                st.selectbox(
                    "Klick-Aktion",
                    options=["move", "add"],
                    format_func=lambda value: "Spot verschieben" if value == "move" else "Neuen Spot hinzufuegen",
                    key="qc_click_action",
                )
                if selected_spot_id:
                    remove_col, toggle_col = st.columns(2)
                    with remove_col:
                        if st.button("Spot loeschen", use_container_width=True):
                            st.session_state["free_spot_metadata"] = ensure_free_spot_metadata(
                                metadata.loc[metadata["spot_id"].astype(str) != selected_spot_id].copy()
                            )
                            st.session_state["qc_canvas_version"] = int(st.session_state.get("qc_canvas_version", 0)) + 1
                            st.rerun()
                    with toggle_col:
                        if st.button("Exclude umschalten", use_container_width=True):
                            updated_metadata = metadata.copy()
                            mask = updated_metadata["spot_id"].astype(str) == selected_spot_id
                            updated_metadata.loc[mask, "exclude"] = ~updated_metadata.loc[mask, "exclude"].astype(bool)
                            st.session_state["free_spot_metadata"] = ensure_free_spot_metadata(updated_metadata)
                            st.session_state["qc_canvas_version"] = int(st.session_state.get("qc_canvas_version", 0)) + 1
                            st.rerun()
                if HAS_IMAGE_COORDINATES:
                    qc_clicked = streamlit_image_coordinates(
                        display_image_pil,
                        key="qc_click_image",
                        width=min(display_image_pil.width, 900),
                    )
                    if qc_clicked:
                        click_signature = (
                            f"{qc_clicked['x']:.1f},{qc_clicked['y']:.1f},"
                            f"{st.session_state['qc_click_action']},{selected_spot_id}"
                        )
                        if click_signature != st.session_state.get("last_qc_click_signature", ""):
                            updated_metadata = apply_qc_click_action(
                                metadata,
                                qc_clicked,
                                action=st.session_state["qc_click_action"],
                                selected_spot_id=selected_spot_id or None,
                                default_radius=float(grid_config.roi_radius),
                            )
                            st.session_state["last_qc_click_signature"] = click_signature
                            st.session_state["free_spot_metadata"] = ensure_free_spot_metadata(updated_metadata)
                            st.session_state["qc_canvas_version"] = int(st.session_state.get("qc_canvas_version", 0)) + 1
                            st.rerun()
                    st.caption("Fuer neue Spots oder schnelle Korrekturen kannst du weiterhin direkt ins Bild klicken.")
                else:
                    st.info("Fuer klickbare QC-Korrekturen wird `streamlit-image-coordinates` benoetigt.")

        formula_df = pd.DataFrame(
            {
                "Schritt": [
                    "Signalorientierung",
                    "Detection-Background",
                    "Background-Korrektur",
                    "Normalisierung",
                    "Linearer Kennwert",
                ],
                "Beschreibung": [
                    "Intern bedeutet groessere Intensitaet immer staerkeres Signal.",
                    "Fuer die freie Spot-Suche wird ein separates background-subtrahiertes Detection-Bild verwendet.",
                    "corrected_integrated_intensity = sum(spot_roi) - area(spot_roi) * background_value",
                    results["normalization_note"].iloc[0] if "normalization_note" in results.columns else "Keine",
                    "Default: Steigung des linearen Fits im gewaehlten Verdunnungsbereich",
                ],
            }
        )
        st.dataframe(formula_df, use_container_width=True, hide_index=True)
        st.caption(f"Gewaehlte Background-Methode: {analysis_config.background_mode}")
        st.pyplot(qc_figure, use_container_width=True)
        st.dataframe(summarize_qc_flags(results), use_container_width=True, hide_index=True)
        flagged = results[
            results["exclude"]
            | results["is_saturated"]
            | results["is_weak"]
            | results["is_negative_after_background"]
            | results["edge_clipped"]
        ]
        if not flagged.empty:
            st.markdown("**Auffaellige Spots**")
            st.dataframe(flagged, use_container_width=True, hide_index=True)

    with tabs[5]:
        st.markdown("**Auto-Vorschlag und manuelle Auswahl**")
        linear_editor_df = results[
            [
                "spot_id",
                "sample",
                "dilution",
                "replicate",
                "corrected_integrated_intensity",
                "is_saturated",
                "linear_fit_eligible",
                "linear_fit_auto_include",
                "linear_fit_include",
            ]
        ].copy()
        edited_linear = st.data_editor(
            linear_editor_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            disabled=[
                "spot_id",
                "sample",
                "dilution",
                "replicate",
                "corrected_integrated_intensity",
                "is_saturated",
                "linear_fit_eligible",
                "linear_fit_auto_include",
            ],
            key="linear_fit_editor",
        )
        st.session_state["linear_fit_selection"] = pd.DataFrame(edited_linear)[["spot_id", "linear_fit_include"]].copy()
        results["linear_fit_include"] = pd.DataFrame(edited_linear)["linear_fit_include"].astype(bool).to_numpy()
        results, linear_summary = apply_manual_linear_selection(
            results,
            value_column="corrected_integrated_intensity",
            x_mode=analysis_config.linear_range_x_mode,
            min_points=analysis_config.linear_range_min_points,
        )
        linear_fit_plot = plot_linear_range_fits(results, linear_summary, "corrected_integrated_intensity")
        linear_slope_plot = plot_linear_summary_metric(linear_summary, "slope", "Steigung pro Sample im linearen Bereich")
        st.dataframe(linear_summary, use_container_width=True, hide_index=True)
        st.pyplot(linear_fit_plot, use_container_width=True)
        st.pyplot(linear_slope_plot, use_container_width=True)

    with tabs[6]:
        st.markdown("**Ergebnisse**")
        sample_summary = linear_summary.copy()
        if results["normalized_signal"].notna().any():
            normalized_agg = (
                results.loc[~results["exclude"]]
                .groupby("sample", dropna=False)["normalized_signal"]
                .median()
                .rename("median_normalized_signal")
                .reset_index()
            )
            sample_summary = sample_summary.merge(normalized_agg, on="sample", how="left")
        st.markdown("**Sample-Zusammenfassung**")
        st.dataframe(sample_summary, use_container_width=True, hide_index=True)
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.pyplot(raw_plot, use_container_width=True)
        st.pyplot(background_plot, use_container_width=True)
        if background_fit_plot is not None:
            st.pyplot(background_fit_plot, use_container_width=True)
        st.pyplot(corrected_plot, use_container_width=True)
        if normalized_plot is not None:
            st.pyplot(normalized_plot, use_container_width=True)
        st.pyplot(dilution_plot, use_container_width=True)
        st.pyplot(heatmap_plot, use_container_width=True)

    with tabs[7]:
        st.download_button(
            "CSV exportieren",
            data=dataframe_to_csv_bytes(results),
            file_name="dotblot_results.csv",
            mime="text/csv",
        )
        st.download_button(
            "Analyseparameter als JSON exportieren",
            data=analysis_manifest_bytes(
                image_info,
                grid_config,
                analysis_config,
                metadata,
                linear_summary=linear_summary,
                results=results,
            ),
            file_name="dotblot_analysis_parameters.json",
            mime="application/json",
        )
        st.download_button(
            "QC-Overlay als PNG exportieren",
            data=figure_to_png_bytes(qc_figure),
            file_name="dotblot_qc_overlay.png",
            mime="image/png",
        )
        st.download_button(
            "Plot: korrigierte Signale",
            data=figure_to_png_bytes(corrected_plot),
            file_name="dotblot_corrected_signals.png",
            mime="image/png",
        )
        st.download_button(
            "Plot: Verdunnungsreihe",
            data=figure_to_png_bytes(dilution_plot),
            file_name="dotblot_dilution_series.png",
            mime="image/png",
        )
        st.download_button(
            "CSV: Linear-Range Summary",
            data=dataframe_to_csv_bytes(linear_summary),
            file_name="dotblot_linear_range_summary.csv",
            mime="text/csv",
        )
        st.download_button(
            "Plot: Linear-Range Fits",
            data=figure_to_png_bytes(linear_fit_plot),
            file_name="dotblot_linear_range_fits.png",
            mime="image/png",
        )

    with tabs[8]:
        render_help_tab()


if __name__ == "__main__":
    main()
