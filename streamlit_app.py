from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st

from dotblot.analysis import run_quantification
from dotblot.config import (
    APP_NAME,
    APP_VERSION,
    BACKGROUND_OPTIONS,
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

    files = sorted({str(path.resolve()): path for path in candidates}.items(), key=lambda item: Path(item[0]).name.lower())
    return {Path(path_str).name: path_str for path_str, _ in files}


def initialize_state(image_hash: str, gray_image_shape: tuple[int, ...]) -> None:
    if st.session_state.get("current_image_hash") != image_hash:
        suggested = suggest_grid_config(gray_image_shape, 4, 6)
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
        st.session_state.pop("linear_fit_selection", None)

    defaults = default_analysis_config()
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
            st.session_state["grid_anchor_x"] = float(clicked["x"])
            st.session_state["grid_anchor_y"] = float(clicked["y"])
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
                "3. Im Tab `Grid Setup` das Grid automatisch initialisieren und visuell kontrollieren.",
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
                    "Die App sucht im MVP nicht frei nach einzelnen Blobs, sondern initialisiert zuerst ein regulaeres Grid.",
                    "",
                    "Dafuer werden:",
                    "- spot-aehnliche Antwortbilder berechnet",
                    "- Spalten- und Zeilenprofile ausgewertet",
                    "- fehlende schwache Peaks auf das erwartete Raster regularisiert",
                    "- die gefundenen Zentren in einer zweiten lokalen Passung nachgeschaerft",
                    "",
                    "Das ist fuer Dot Blots meist stabiler als freie Blob-Erkennung, vor allem bei schwachen, unscharfen oder teilweise gesaettigten Spots.",
                ]
            )
        )

    with st.expander("Background-Methoden"):
        st.markdown(
            "\n".join(
                [
                    "- `local_annulus_median`: guter Default fuer viele Membranen",
                    "- `local_window_median`: nuetzlich, wenn benachbarte Spots den Annulus stoeren",
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
                    "- Grid automatisch initialisieren und danach visuell kontrollieren",
                    "- `Spot-Zentren lokal verfeinern` aktiviert lassen",
                    "- ROI-Radius etwas verkleinern",
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
            st.number_input("ROI-Radius", min_value=2.0, step=1.0, key="grid_roi_radius")
            st.selectbox("ROI-Form", ROI_SHAPES, key="grid_roi_shape")
        with right:
            render_anchor_picker(display_image_pil)

        current_grid = make_grid_config()
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

    with tabs[3]:
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

    metadata = ensure_metadata_shape(
        st.session_state.get("spot_metadata"),
        int(st.session_state["grid_rows"]),
        int(st.session_state["grid_cols"]),
        st.session_state["metadata_preset"],
    )
    sample_options = sorted([sample for sample in metadata["sample"].astype(str).unique() if sample])

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
        st.markdown("**Grid-Overlay Vorschau**")
        st.pyplot(qc_figure, use_container_width=True)

    with tabs[4]:
        formula_df = pd.DataFrame(
            {
                "Schritt": [
                    "Signalorientierung",
                    "Background-Korrektur",
                    "Normalisierung",
                    "Linearer Kennwert",
                ],
                "Beschreibung": [
                    "Intern bedeutet groessere Intensitaet immer staerkeres Signal.",
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
