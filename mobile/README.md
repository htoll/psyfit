# PsyFit Mobile

Andor Solis `.sif` viewer and single-molecule analysis for the phone. It builds
into one self-contained web page (no server, nothing fetched at run time) that
runs as a Claude artifact or in any browser.

What it does: opens `.sif` images and movies (from the Files picker, or from
Google Drive through the Drive connector when it runs as an artifact), fits
PSF brightness in pps the way PsyFit does, shows brightness histograms and
aggregation classes, restricts analysis to channels (MCL: four quadrants;
Nikon: see below), overlays channels and files with registration and
colocalization, plays movies, tracks particles and plots intensity traces with
step detection. Pan, pinch to zoom, double-tap a spot to fit it, hold for pixel
values, tap the colour bar to set its min and max.

## Build and test

```bash
cd mobile
npm install
npm run build     # dist/psyfit-mobile.html (open in a browser) and dist/artifact.html
npm test          # core checks in Node
```

`dist/artifact.html` is the page body to publish as a Claude artifact. It needs
two capabilities: `downloads` (saving images and CSVs) and `mcp` for the
`Google Drive` connector with the tools `search_files`, `download_file_content`,
`get_file_metadata` and `list_shared_drives`.

## Layout

```
src/main.js        Entry point
src/store.js       All state and every action that changes it
src/platform.js    Drive connector, saving files, on-device file cache, preferences
src/core/          No DOM: sif.js (reader/writer), numeric.js (detection, PSF fits),
                   stats.js, geometry.js (registration, colocalization), tracking.js,
                   layouts.js (microscopes and channels), render.js (colormaps,
                   compositing), demo.js (simulated files)
src/ui/            Preact components; viewer.js is the image canvas and gestures
test/              Node tests on the core modules and the simulated data
```

## Conventions

- Pixels are x = column, y = row with row 0 first, as PsyFit's `x_pix`/`y_pix`;
  images are drawn with row 0 at the bottom (matplotlib `origin='lower'`).
- Intensities are pps: counts × (5 / EM gain) / exposure / accumulations.
- Detection and the single-PSF fit follow `utils.integrate_sif` step for step.
  On the simulated demo image the fits match PsyFit's in every MCL quadrant
  (same peaks, brightness to a few parts per million).
- MCL channels are the quadrants PsyFit calls regions 1 to 4 (Blue, Green, Red,
  NIR). Pixel size 0.1011 µm.
- Nikon frames can be divided three ways, chosen under Display: left/right
  halves (the default), top/bottom halves, or the 280 px "Mr Beam" region that
  PsyFit's Nikon tools analyse. Pixel size 0.107 µm at 100×, scaled by a
  magnification in the file name.
