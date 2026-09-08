# PsyFit

A Streamlit toolkit for single-molecule microscopy and transmission electron microscope
analysis. It takes raw acquisitions straight off the scope (Andor `.sif`, confocal `.dat`,
Abberior `.msr`, TEM `.dm3`/`.emd`) and returns fitted, plottable numbers: per-particle
brightness, emission spectra, colocalization fractions, size distributions, and synthesis
tables.

<!-- Add the deployed Streamlit URL here. -->

## What it does

Every tool has the same shape: upload raw files, set fitting parameters in the sidebar,
get back a figure and a CSV.

**Convert and export**

| Tool | What it does |
| --- | --- |
| Batch Convert | Bulk `.sif` to SVG/TIFF/PNG/JPEG plus a combined table, with optional fit overlays |
| Process Movie | `.sif` movie to MP4/MOV/TIFF with colormap, log scaling, labels, and a colorbar |

**Brightness and intensity**

| Tool | What it does |
| --- | --- |
| Brightness (WF) | 2D Gaussian PSF fits on widefield `.sif`, per-particle brightness distributions |
| Brightness (Conf) | Same for confocal `.dat`/`.tif`, normalized by dwell time and line accumulation |
| Movie Brightness | Per-frame PSF refitting plus Kalafut-Visscher step detection for single-dye brightness |
| Saturation Series | Brightness against excitation power density, by quadrant |

**Visualization**

| Tool | What it does |
| --- | --- |
| Confocal Visualization | Merged multi-channel panels with per-channel colormaps and contrast |
| MSR Analysis | Reads Abberior Imspector OBF containers: stack previews, saved metadata, FOV overlays |

**Spectral and colocalization**

| Tool | What it does |
| --- | --- |
| Dye Colocalization | Pairs dye and UCNP detections within a distance cutoff, reports colocalized fraction |
| Get Spectra | Per-particle emission spectra from spectrally dispersed images, wavelength-calibrated |
| Process Spectra | Curate, baseline-correct, normalize, average, and Gaussian-fit spectra CSVs |
| Spectra Saturation Series | Spectral evolution against excitation power, pooled by condition and laser current |

**Quantification and synthesis**

| Tool | What it does |
| --- | --- |
| Monomer Estimation | Aggregation state from brightness histograms, with optional concentration estimate |
| Shelling Injection Table | Shell-growth injection volumes and timing from core/shell geometry |
| Reaction Planner | Functionalization stoichiometry and mmol-to-mg weigh-outs for lanthanide precursors |
| TEM Size Analysis | Watershed segmentation and shape fitting for calibrated size distributions |
| FFT Analysis | Lattice spacing from HRTEM via FFT spot indexing, scored against candidate phases |

## Running locally

```bash
git clone https://github.com/htoll/psyfit.git
cd psyfit
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11 is what this is developed and tested against.

Movie Brightness is gated off on the hosted deployment (per-frame PSF fitting across a
full movie is memory-heavy enough to take down a shared instance) and needs a local run.
TEM Size Analysis works hosted but should be given one or two images at a time.

## Layout

```
app.py                 Navigation, tool registry, error handling, diagnostics
utils.py               Shared SIF I/O, PSF fitting, plotting, step detection
utilsJFS.py            Confocal .dat parsing helpers
tools/                 One module per tool, each exposing run()
tools/crystallography.py   Space-group / d-spacing engine backing FFT Analysis
tools/roi.py           Shared interactive rectangle-ROI widget
.streamlit/config.toml Upload limits and theme
```

Adding a tool means dropping a module in `tools/` with a `run()` entry point and adding
one row to the `TOOLS` registry in `app.py`. Everything else (navigation, the header
blurb, import-failure handling, the diagnostics check) comes from that registry.

## Notes on the numbers

- Brightness is reported in photons per second: raw counts are normalized by EM gain,
  exposure, and accumulation, a 2D Gaussian is fit to each detection, and the brightness is
  the summed intensity over the fit subregion minus the fitted background offset.
- Confocal brightness divides amplitude by dwell time and line accumulation so scans taken
  under different acquisition settings are directly comparable.
- Monomer Estimation's concentration figure assumes particles are uniformly distributed
  through a 3 mm PDMS well loaded with 5 uL of 1x PBS and allowed to settle for more than
  five minutes. Treat it as an estimate, not a measurement.
- Spectral tools need the paired `saving_info` and `fits` calibration pickles from the same
  acquisition date; a mismatched pair is the usual cause of failed spectrum extraction.
