# Dot Blot Quantifier User Guide

This guide is the practical companion to the repository README. It focuses on how to run the app, how to choose parameters, and how to interpret the outputs in a lab workflow.

## Purpose

The app quantifies dot-blot images from a regular grid layout. It is designed for:

- yield estimation from transfection samples
- dilution-series comparison across samples
- reproducible background correction
- QC-first review before exporting numbers

The current MVP is deliberately based on classical image analysis, not machine learning.

## Supported Input

- TIFF, TIF
- PNG
- JPG, JPEG
- grayscale images
- color images that can be converted to grayscale

After loading an image, the app shows:

- image shape
- data type
- intensity range
- whether the input was color or grayscale

## Recommended Workflow

1. Load a blot image or one of the bundled example images.
2. Check polarity in the `Preprocessing` tab.
3. Define the grid in `Grid Setup`.
4. Review and edit spot metadata.
5. Choose background correction and optional center refinement in `Quantification`.
6. Inspect the QC overlay and QC summary.
7. Review the linear dilution range in `Linear Range`.
8. Inspect full tables and plots in `Results`.
9. Export CSV, QC overlay, plots, and analysis JSON.

## Preprocessing and Polarity

The app internally converts the image into an oriented analysis image where larger values always mean stronger signal.

Available polarity modes:

- `Auto`
- `Spots hell`
- `Spots dunkel`
- `Bild invertieren`

Recommendation:

- Start with `Auto`.
- If the QC overlay or quantified values look implausible, switch manually.
- Keep the final polarity setting consistent across comparable experiments.

## Grid Setup

The app assumes a regular grid. For the MVP, this is the most reproducible strategy and is more robust than unrestricted blob detection on noisy membranes.

Main grid parameters:

- `Zeilen`
- `Spalten`
- `Anker X`, `Anker Y`
- `Abstand X`, `Abstand Y`
- `Rotation`
- `ROI-Radius`
- `ROI-Form`

Useful actions:

- `Grid automatisch initialisieren`
- image click to place the top-left anchor spot
- `Metadaten aus Preset neu erzeugen`

Current auto-initialization strategy:

- builds a spot-like response map
- estimates regular positions from image profiles
- regularizes missing weak peaks onto a grid
- performs a second local alignment pass so faint spots do not collapse the whole grid

Recommendation:

- Use auto-initialization as a starting point.
- Always visually confirm the ROI overlay before trusting the numbers.

## Spot Metadata

Each spot can carry:

- `spot_id`
- `sample`
- `dilution`
- `replicate`
- `exclude`

Presets:

- `Zeilen = Samples`
- `Spalten = Samples`
- `Flexibel / manuell`

Recommendation:

- Use numeric dilution labels when possible.
- Keep replicate naming consistent.
- Exclude bad spots instead of deleting them, so the audit trail stays intact.

## Quantification

For each spot, the app computes at least:

- raw integrated intensity
- raw mean intensity
- background value
- background statistics
- corrected integrated intensity
- corrected mean intensity

Main formula:

`corrected_integrated_intensity = sum(spot_roi) - area(spot_roi) * background_value`

Negative corrected values are allowed and flagged. They are often informative for weak or over-corrected spots.

## Background Correction Modes

### Local annulus

Uses a ring around the spot ROI.

Best when:

- the membrane background is locally stable
- neighboring spots are not too close

Recommended default:

- `local_annulus_median`

### Local window

Uses a square local neighborhood around the spot and excludes the ROI itself.

Best when:

- the annulus is contaminated by nearby spots
- the local background changes more gradually

Key parameter:

- `Lokales Fenster (Scale x ROI)`

### Background surface

Uses a smooth background image estimated by Gaussian filtering.

Best when:

- the membrane has a broad gradient
- illumination or chemiluminescence is uneven over the field

Key parameter:

- `Surface sigma`

### Global background

Uses one background value for the whole image.

Best when:

- the membrane background is very uniform

Caution:

- Usually less robust for real membranes than local background methods.

## Weak Spots and Automatic Detection

Weak spots are the main reason why unrestricted spot detection often becomes unstable. In this tool:

- the grid geometry is the primary model
- automatic initialization now regularizes missing weak peaks
- local center refinement uses a spot-like response map instead of relying only on the brightest raw pixel

If weak spots are still problematic:

- slightly reduce `Weak-Spot SNR threshold`
- use center refinement
- check polarity manually
- decrease ROI radius if the ROI captures too much background
- compare `local_annulus_median` against `local_window_median`

## Saturation

The app does not silently remove saturated spots. Instead, it flags them.

Saturation-related signals:

- large fraction of spot pixels near image maximum
- flat plateaus inside the spot
- unusually low variance inside a very bright spot

Recommendation:

- keep saturated spots visible in QC
- usually exclude them from linear dilution fitting

## Linear Range Analysis

The `Linear Range` tab proposes a contiguous signal range per sample for a linear fit. You can then adjust this manually.

Configurable options:

- X-axis transform for the fit
- minimum number of points
- whether saturated spots should be auto-excluded

Output per sample:

- slope
- intercept
- R squared
- number of included points
- selected spot IDs

Recommendation:

- use the automatic selection as a first pass
- review the fit plot and manually correct obvious non-linear regions

## Normalization

The default normalization mode is reference normalization per dilution level.

This is usually the most transparent first strategy because it avoids over-compressing the dilution series into a single metric too early.

Use normalization when:

- you want fold-change style comparison against a standard
- the same dilution levels are present across samples

## QC View

The QC overlay is the most important sanity check in the app.

It shows:

- spot ROIs
- background rings
- spot labels
- excluded spots
- weak spots
- saturated spots

Interpretation:

- cyan: regular spot
- yellow: weak spot
- orange: saturated spot
- red: excluded spot

Never rely on the output table without checking QC first.

## Results and Export

The app exports:

- full result CSV
- analysis parameter JSON
- QC overlay PNG
- plot PNGs
- linear range summary CSV

The JSON export is intended for reproducibility and includes:

- image metadata
- grid settings
- analysis settings
- spot metadata
- linear range summary
- manual fit selection

## Troubleshooting

### The grid is globally shifted

- re-run auto-initialization
- click the top-left spot again
- verify `Abstand X` and `Abstand Y`
- check whether polarity is correct

### Weak spots are missed

- keep the regular grid and use refinement instead of full manual placement
- reduce ROI radius slightly
- compare annulus vs local window background
- inspect whether the weak spot is actually below background noise

### Background looks too high

- switch from annulus to local window
- reduce annulus size if neighboring spots contaminate it
- try the background surface mode for strong gradients

### Corrected values become negative

- this can be real for extremely weak spots
- verify the ROI placement and local background
- review whether the chosen background mode is too aggressive

## Practical Defaults

For most blot images, a good starting point is:

- polarity: `Auto`
- background: `local_annulus_median`
- center refinement: enabled
- linear range: auto-select contiguous range
- normalization: reference per dilution

## Current Scope

The current version is optimized for:

- single-image analysis
- regular dot-blot grids
- transparent and reproducible measurements

Planned future directions include:

- better automatic initialization for rotated grids
- local fine-centering per spot with richer diagnostics
- Excel export
- batch processing
- more automated range selection and sample summaries
