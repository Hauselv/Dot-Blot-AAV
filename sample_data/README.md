# Sample Data

Dieses Verzeichnis enthaelt bewusst keine binaeren Testbilder im Repository.

Mit `generate_synthetic_dotblot.py` lassen sich reproduzierbare Testbilder erzeugen, um:

- Polarity-Autoerkennung
- Grid-Overlay
- lokale Background-Korrektur
- leichte Rotation
- Hintergrundgradient
- Saettigung im oberen Bereich

zu pruefen.

Erzeugt werden aktuell:

- `synthetic_dotblot_bright.png`
- `synthetic_dotblot_dark.png`
- `synthetic_dotblot_bright_16bit.tiff`
- `synthetic_dotblot_dark_16bit.tiff`

Unter `sample_data/external/` liegen zusaetzlich zwei oeffentlich verlinkte Wikimedia-Dot-Blot-Beispiele.
Die Quellen dazu stehen in `external_sources.md`.
