# Dot Blot Quantifier

Eine lokale Streamlit-App zur quantitativen Auswertung von Dot-Blot-Bildern mit klassischer Bildanalyse.

## MVP-Funktionen

- Bildimport fuer TIFF, PNG, JPG und JPEG
- robuste Graustufenkonvertierung fuer Farb- und Graustufenbilder
- automatische oder manuelle Polarity-Behandlung
- halbautomatische Grid-Definition mit Bildklick fuer den Ankerpunkt
- automatische Grid-Initialisierung aus dem Bild als Startpunkt
- optionale lokale Feinzentrierung der Spot-Zentren
- lokale Background-Korrektur per Annulus
- korrigierte Spot-Intensitaeten und QC-Flags
- Referenz-Normalisierung pro Verduenungsstufe
- Auto-Vorschlag fuer lineare Verdunnungsbereiche plus manuelle Uebersteuerung
- Sample-Zusammenfassung ueber linearen Fit im gewaehlten Bereich
- Tabellen, QC-Overlay und Basisplots
- CSV-, JSON- und PNG-Export

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Start

```bash
streamlit run streamlit_app.py
```

## Tests

```bash
pytest
```

## Hinweise

- Die App analysiert zunaechst Einzelbilder.
- Die Quantifizierung arbeitet intern immer mit einem orientierten Bild, in dem groessere Werte staerkeres Signal bedeuten.
- Sehr schwache oder gesaettigte Spots werden nicht entfernt, sondern explizit in den Ergebnissen markiert.
