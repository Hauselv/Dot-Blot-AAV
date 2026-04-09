# Dot Blot Quantifier

Dot Blot Quantifier is a local Streamlit app for reproducible, QC-first quantification of dot-blot images using classical image analysis.

The tool was designed for small lab workflows where dot blots are used to estimate relative sample yield after transfections, compare dilution series, and document the exact analysis settings used for each result.

## Goals

- robust quantification before clever automation
- transparent calculations instead of black-box scoring
- strong QC overlays so image geometry stays reviewable
- local execution without cloud dependency
- modular Python code that can be extended in small steps

## Current Feature Set

- image import for TIFF, TIF, PNG, JPG, and JPEG
- grayscale conversion for color and grayscale inputs
- automatic polarity detection with manual override
- regular-grid spot model with anchor, pitch, radius, and rotation controls
- automatic grid initialization from the image
- improved weak-spot handling during auto-initialization
- optional local center refinement for each spot
- editable spot metadata and exclude flags
- multiple background modes:
  - local annulus
  - local window
  - smooth background surface
  - global background
- support for median and mean background statistics
- spot-wise corrected integrated intensity and corrected mean intensity
- QC flags for weak, saturated, edge-clipped, and over-corrected spots
- reference normalization per dilution level
- automatic proposal of a linear dilution range per sample
- manual override of the linear-fit spot selection
- results tables, dilution plots, grid heatmap, and fit plots
- export of CSV, JSON, and PNG outputs
- bundled synthetic and external example images for testing

## Why Grid-Based Analysis

For dot blots, a regular grid is usually the most reliable starting point.

Compared with unrestricted blob detection, a grid-first strategy is typically more robust when:

- spots are weak
- some spots are saturated
- membranes have local background gradients
- images have been inverted during export
- one or more spots are blurred or partially missing

The current auto-initialization therefore treats automatic detection as a helper for grid placement, not as a replacement for grid geometry.

## Repository Structure

```text
streamlit_app.py          Streamlit UI and workflow orchestration
dotblot/
  analysis.py             Main quantification pipeline
  background.py           Background estimation helpers
  config.py               Defaults and UI options
  export.py               CSV, JSON, and figure export helpers
  grid.py                 Grid model, auto-init, center refinement
  image_io.py             Image loading and grayscale conversion
  linear_range.py         Linear dilution-range selection and fitting
  masks.py                Spot and background masks
  normalization.py        Reference normalization
  plotting.py             Matplotlib plots
  preprocessing.py        Polarity detection and image orientation
  qc.py                   QC overlay and QC summaries
  quantification.py       Per-spot metrics
  types.py                Dataclasses
tests/                    Unit and workflow-focused tests
sample_data/              Synthetic and external example images
docs/                     User-facing documentation
```

## Installation

Recommended with a local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the App

```powershell
.\.venv\Scripts\activate
streamlit run streamlit_app.py
```

The app runs locally in your browser and does not require a remote backend.

## Quick Start

1. Start the app.
2. Load your blot image or choose one of the bundled sample images.
3. Check polarity in `Preprocessing`.
4. Use `Grid automatisch initialisieren` in `Grid Setup`.
5. Review the ROI overlay and adjust anchor, spacing, radius, or rotation if needed.
6. Edit sample, dilution, replicate, and exclude fields.
7. Choose a background method in `Quantification`.
8. Inspect the QC overlay before trusting the table.
9. Review the linear range fit and summary.
10. Export CSV, PNG, and JSON outputs.

## Quantification Model

The app converts every image into an oriented analysis image where higher intensity always means stronger signal.

For each spot, the main corrected metric is:

`corrected_integrated_intensity = sum(spot_roi) - area(spot_roi) * background_value`

The app also reports:

- raw integrated intensity
- raw mean intensity
- corrected mean intensity
- background mean, median, and standard deviation
- SNR estimate
- QC flags

Negative corrected values are preserved and flagged instead of being clipped away.

## Background Correction Modes

### Local annulus

Best default for many dot blots. Uses a ring around the ROI and works well when neighboring spots do not contaminate the background region.

### Local window

Useful when the annulus is unstable or too close to adjacent spots. Uses a wider local neighborhood around the spot while excluding the spot itself.

### Background surface

Useful for membranes with smooth spatial background gradients. A smoothed background map is estimated once and then sampled under each spot.

### Global background

Simple and transparent, but usually less robust for real blot images unless the membrane background is very uniform.

## Handling Weak Spots

Weak spots are one of the main failure modes in automatic image analysis. The app currently addresses this by:

- using a spot-like response map rather than plain intensity alone
- estimating grid positions from image profiles
- regularizing missing peaks back onto an expected grid
- running a second local alignment pass so faint spots do not drag the entire grid off target
- optionally refining each spot center locally during quantification

This keeps the workflow reproducible while still improving sensitivity for weak spots.

## Linear Range Evaluation

The app supports dilution-series analysis with:

- automatic contiguous linear-range proposal per sample
- optional exclusion of saturated spots from the proposal
- manual per-spot override in the UI
- fit summary including slope, intercept, R squared, number of points, and selected spot IDs

The current default summary metric is the slope of the fitted linear range.

## QC Philosophy

The QC overlay is a first-class output, not a cosmetic extra.

It highlights:

- spot ROIs
- background rings
- spot IDs
- excluded spots
- weak spots
- saturated spots

Recommended interpretation:

- review QC before exporting values
- keep problematic spots visible and flagged
- exclude only after visual confirmation

## Outputs

The app can export:

- full results table as CSV
- linear-range summary as CSV
- QC overlay as PNG
- corrected signal plot as PNG
- dilution-series plot as PNG
- analysis parameters and metadata as JSON

The JSON export is intended for reproducibility and stores:

- image metadata
- grid settings
- analysis settings
- spot metadata
- linear range summary
- manual fit selection

## Example Data

Bundled examples live in:

- [sample_data](sample_data)
- [sample_data/external](sample_data/external)

These include:

- synthetic dot blots for controlled testing
- inverted examples for polarity checks
- external public-domain or permissively available reference images collected in [sample_data/external_sources.md](sample_data/external_sources.md)

## Documentation

User-facing documentation:

- [README.md](README.md)
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md)

The app itself also contains a `Help` tab with workflow guidance, parameter explanations, and troubleshooting notes.

## Development

### Run tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

### Check syntax

```powershell
.\.venv\Scripts\python -m compileall streamlit_app.py dotblot tests
```

## Assumptions and Current Scope

- single-image workflow, not batch processing
- spots lie on an approximately regular rectangular grid
- users know row and column count, or can estimate them
- classical image analysis only, no machine learning
- local usage is preferred over remote services

## Roadmap

- more robust initialization for rotated grids
- additional spot-centering diagnostics
- Excel export
- batch analysis of multiple images
- richer sample-level metrics
- improved automation for line-range selection

## Status

The project is already useful for MVP-style local analysis and is being iteratively tuned against synthetic and real blot images. The focus remains on correctness, transparency, and practical lab usability.
