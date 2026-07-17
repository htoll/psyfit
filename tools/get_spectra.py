
import os
import re
import io
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

import streamlit as st

import sys

import utils
from tools import roi as roi_tool


import sif_parser

import pickle as pkl
import textwrap

PIX_SIZE_UM = 0.107 # from Mr Beam

# --- Spectrum extraction window (in spectral-detector pixels) ---
# A particle's spectrum is sampled from (xs - SPEC_LEFT) to (xs + SPEC_RIGHT)
# along x, averaged over rows (ys - SPEC_HALF_ROWS) .. (ys + SPEC_HALF_ROWS).
# Keep these in one place so the overlap/edge filters stay tied to the geometry
# actually used by get_spectrum().
SPEC_LEFT = 90      # pixels the dispersed spectrum extends left of xs
SPEC_RIGHT = 9      # pixels it extends right of xs
SPEC_HALF_ROWS = 2  # rows averaged above/below ys
SPEC_NPTS = 100     # samples across the window

# try:
#     from tools.process_files import process_files as _process_files_external  # type: ignore
# except Exception:
#     _process_files_external = None

# --- Helpers ---
def _process_files(uploaded_files, region="Mr Beam", threshold=1, signal="UCNP", pix_size_um=0.107, sig_threshold=0.3):
    processed_data: Dict[str, Dict[str, object]] = {}
    all_dfs = []
    temp_dir = Path(tempfile.gettempdir()) / "spec_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for uf in uploaded_files:
        file_path = temp_dir / uf.name # equivalent to file_path = os.path.join(temp_dir, uf.name)
        with open(file_path, "wb") as f:
            f.write(uf.getbuffer())
        try:
            df, image_data_cps, _ = utils.integrate_sif(
                str(file_path),
                region=region,
                threshold=threshold,
                signal=signal,
                pix_size_um=pix_size_um,
                sig_threshold=sig_threshold,
            )
            processed_data[uf.name] = {"df": df, "image": image_data_cps}
            all_dfs.append(df)
        except Exception as e:
            st.error(f"Error processing {uf.name}: {e}")
    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return processed_data, combined_df

def just_read_in(uploaded_files):
    full_frames = {}
    temp_dir = Path(tempfile.gettempdir()) / "spec_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for uf in uploaded_files:
        file_path = temp_dir / uf.name # equivalent to file_path = os.path.join(temp_dir, uf.name)
        with open(file_path, "wb") as f:
            f.write(uf.getbuffer())
        try:
            image_data, metadata = sif_parser.np_open(str(file_path), ignore_corrupt=True)
            image_data = image_data[0]  # (H, W)

            gainDAC = metadata['GainDAC']
            if gainDAC == 0:
                gainDAC =1 #account for gain turned off
            exposure_time = metadata['ExposureTime']
            accumulate_cycles = metadata['AccumulatedCycles']

            # Normalize counts → photons
            image_data_cps = image_data * (5.0 / gainDAC) / exposure_time / accumulate_cycles
            image_data_cps = np.flipud(image_data_cps)

            full_frames[uf.name] = image_data_cps
        except Exception as e:
            st.error(f"Error reading in {uf.name}: {e}")

    return full_frames

# def read_in_calibration(date, folder = "G:/Shared drives/SamPengLab/Alev_Studenikina/Multicolor/Heterogeneity/Calibration data/"):
#     import pickle as pkl

#     with open(f"{folder}saving_info_{date}_no_tracking.pkl", "rb") as f:
#         calibration = pkl.load(f)

#     with open(f"{folder}{date}_fits.pkl", "rb") as f:
#         calib_fits = pkl.load(f)

#     return calibration, calib_fits

def _classify_calibration_uploads(uploads):
    """Sort the combined calibration upload into (cal_file, fit_file) by filename.

    A name containing ``saving_info`` is the illumination-grid data; one containing
    ``fits`` is the wavelength fits. Returns ``(cal_file, fit_file, error)`` where
    ``error`` is a message (or ``None``) describing why classification failed.
    """
    cal_file = fit_file = None
    unknown = []
    for uf in uploads:
        name = uf.name.lower()
        if "saving_info" in name:
            cal_file = uf
        elif "fits" in name:
            fit_file = uf
        else:
            unknown.append(uf.name)

    err = None
    if len(uploads) > 2:
        err = ("Upload at most two calibration files — one 'saving_info…' and one "
               "'…fits…' .pkl.")
    elif unknown:
        err = ("Couldn't classify " + ", ".join(unknown) +
               " — filenames must contain 'saving_info' or 'fits'.")
    return cal_file, fit_file, err


def read_in_calibration(uploaded_files):
    temp_dir = Path(tempfile.gettempdir()) / "spec_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for uf in uploaded_files:
        file_path = temp_dir / uf.name # equivalent to file_path = os.path.join(temp_dir, uf.name)
        with open(file_path, "wb") as f:
            f.write(uf.getbuffer())

        with open(file_path, "rb") as f:
            file = pkl.load(f)
        
        files.append(file)

    return files
    
def get_spectrum(coord, frame, calibration, calib_fits, background="na"):
    object, image = utils.get_image(coord, np.array([val[1] for val in calibration.values()]), np.array([val[3] for val in calibration.values()]))

    xs, ys = image
    if round(ys) < 2:
        raise ValueError("y coordinate too close to edge of image to extract spectrum")
    if round(ys) < 510:
        rows_to_average_over = [-2, -1, 0, 1, 2]
    elif round(ys) == 510:
        rows_to_average_over = [-2, -1, 0, 1]
    elif round(ys) == 511:
        rows_to_average_over = [-2, -1, 0]
    else:
        raise ValueError("y coordinate too close to edge of image to extract spectrum")

    intensity = np.zeros(SPEC_NPTS)

    for j in rows_to_average_over:
        intensity += np.interp(np.linspace(xs - SPEC_LEFT, xs + SPEC_RIGHT, SPEC_NPTS), np.arange(512), frame[round(ys)+j, :])
    intensity /= len(rows_to_average_over)

    # if rightmost side cropped
    if len(intensity) < SPEC_NPTS:
        intensity = np.hstack((intensity, np.ones(SPEC_NPTS - len(intensity))*intensity[-1]))

    k=0
    distances = [utils.distance(coord, val[1]) for val in calibration.values()]
    sorted_ind = np.argsort(distances)
    ids_sorted_by_distance = np.array(list(calibration.keys()))[sorted_ind]
    closest = ids_sorted_by_distance[0]
    while closest not in calib_fits:
        k += 1
        closest = ids_sorted_by_distance[k]

    custom_pixel_to_wvl = utils.exp(np.arange(100), *calib_fits[closest])

    nms = utils.exp(np.arange(101)-0.5, *calib_fits[closest])
    nms_per_pixel = [nms[i+1]-nms[i] for i in range(100)]

    # if type(background) != str:
    #     background_subtract = np.interp(np.linspace(xs - 90, xs+9, 100), np.arange(512), background[round(ys)+j, :])
    #     if len(background_subtract) < 100:
    #         background_subtract = np.hstack((background_subtract, np.ones(100 - len(background_subtract))*background_subtract[-1]))
        
    #     return custom_pixel_to_wvl, intensity-background_subtract
    
    # else:
    return custom_pixel_to_wvl, intensity, nms_per_pixel
    

def fit_template_linear(y, template, return_params=False):
    """
    Fit y ~ a*template + b using least squares.
    Returns fitted background (a*template + b) and (a,b) if requested.
    """
    y = np.asarray(y).ravel()
    t = np.asarray(template).ravel()
    if y.shape != t.shape:
        raise ValueError("y and template must have same shape")

    # design matrix [template, 1]
    A = np.vstack([t, np.ones_like(t)]).T
    # solve least squares
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    bg = a * t + b
    if return_params:
        return bg, (a, b)
    return bg

def fit_template_sigma_clip(y, template, niter=10, sigma_thresh=1):
    y = np.asarray(y).ravel()
    t = np.asarray(template).ravel()
    mask = np.ones_like(y, dtype=bool)
    a = b = 0.0
    for _ in range(niter):
        if mask.sum() < 3:
            break
        bg, (a, b) = fit_template_linear(y[mask], t[mask], return_params=True)
        # build full bg for residuals
        full_bg = a*t + b
        resid = y - full_bg
        std = resid[mask].std(ddof=1) if mask.sum() > 1 else resid.std()
        # keep points that are not strong positive outliers (peaks)
        new_mask = resid < sigma_thresh * std
        if new_mask.sum() < 3 or np.array_equal(new_mask, mask):
            mask = new_mask
            break
        mask = new_mask
    bg = a*t + b
    return bg, (a, b), mask



# --- Illumination tiers (by fraction of the brightest grid point) ---
# Reference cutoffs (Methods.py): Very low <40%, Low <60%, Medium <80%, High ≥80%.
ILLUM_TIER_LABELS = ["Very low", "Low", "Medium", "High"]
ILLUM_TIER_COLORS = {
    "Very low": "#3b4cc0", "Low": "#7b9ff9", "Medium": "#f2a385", "High": "#b40426",
}
# Matplotlib line widths for the spectra plot — thicker = brighter illumination.
ILLUM_TIER_LW = {"Very low": 0.8, "Low": 1.5, "Medium": 2.3, "High": 3.2}


def _tier_for_relative(rel):
    """Classify relative illumination (raw / max over the grid) into a tier,
    using the reference cutoffs: Very low <40%, Low <60%, Medium <80%, High ≥80%.
    Non-finite (outside the calibrated region) is treated as 0 → Very low."""
    if not np.isfinite(rel):
        return "Very low"
    if rel >= 0.80:
        return "High"
    if rel >= 0.60:
        return "Medium"
    if rel >= 0.40:
        return "Low"
    return "Very low"


def _grid_brightness(v):
    """Illumination at a calibration grid point: the fitted spot amplitude
    (``val[2][0]``), matching the reference analysis (Methods.py)."""
    return float(v[2][0])


def _illumination_field(calibration):
    """Build a linear interpolator of grid-point brightness over object-plane
    position (val[1]), so a particle *between* calibration points gets an
    interpolated value. Returns ``(interp, max_raw)``; the interpolator yields
    NaN outside the calibrated convex hull (callers treat that as 0)."""
    from scipy.interpolate import LinearNDInterpolator
    pts = np.array([v[1] for v in calibration.values()], dtype=float)
    vals = np.array([_grid_brightness(v) for v in calibration.values()], dtype=float)
    interp = LinearNDInterpolator(pts, vals)
    max_raw = float(np.nanmax(vals)) if vals.size else 1.0
    return interp, (max_raw or 1.0)


def _interp_tier(interp, max_raw, x, y):
    """Interpolated (raw, relative, tier) illumination at object position (x, y)."""
    raw = float(interp(x, y))
    if not np.isfinite(raw):
        raw = 0.0
    rel = raw / max_raw
    return raw, rel, _tier_for_relative(rel)


def _build_filtered_coords(df, calibration, no_dim, remove_overlapping, left_edge_cutoff, roi=None):
    """Map localized particles to plotting coords and apply quality filters.

    Returns ``{particle_id: [x_pix, y_pix]}`` for the particles that survive,
    using the SAME geometry get_spectrum() uses so the filters actually reflect
    where each spectrum lands on the detector. Filters, in order:

      1. Brightness: drop particles with ``brightness_fit < no_dim``.
      2. Mappable: drop particles utils.get_image() can't place (out of bounds).
      3. Left edge: drop particles whose spectral sampling window runs off the
         left side of the detector. The window spans ``xs - SPEC_LEFT`` to
         ``xs + SPEC_RIGHT``; if its left end ``xs - SPEC_LEFT`` falls below
         ``left_edge_cutoff`` the dispersed channel bleeds into the baseline.
      4. Overlap (optional): two spectra collide when their windows overlap in
         both x and y. Both members of every colliding pair are removed.
    """
    coords = {}
    for i, (_, row) in enumerate(df.iterrows()):
        x = row["x_um"] / PIX_SIZE_UM
        y = row["y_um"] / PIX_SIZE_UM
        coords[i] = [x, 280 - y]

    # 0. Custom ROI: keep only particles inside the drawn rectangle. coords are
    #    already in the same (col, row) space as the displayed frame the ROI was
    #    drawn on, so this is a direct box test.
    if roi is not None:
        row0, row1, col0, col1 = roi
        for i in list(coords.keys()):
            sx, sy = coords[i]
            if not (col0 <= sx <= col1 and row0 <= sy <= row1):
                coords.pop(i, None)

    # 1. Brightness
    for i, (_, row) in enumerate(df.iterrows()):
        if row["brightness_fit"] < no_dim:
            coords.pop(i, None)

    # 1b. Illumination-tier filter (set by the calibration viewer). Each particle's
    #     illumination is interpolated from the calibration grid; only particles
    #     in the selected tiers survive. Disabled when all (or no) tiers selected.
    tf = st.session_state.get("calib_tier_filter")
    if tf and 0 < len(tf.get("include", ())) < len(ILLUM_TIER_LABELS):
        include = tf["include"]
        interp, max_raw = _illumination_field(calibration)
        for i in list(coords.keys()):
            x, y = coords[i]
            _raw, _rel, tier = _interp_tier(interp, max_raw, x, y)
            if tier not in include:
                coords.pop(i, None)

    # 2. Map each surviving particle to its spectral-detector position (xs, ys).
    #    NOTE: get_image returns (object, image); we want IMAGE (the spectral
    #    position), which is what the window math below operates on.
    grid = [c_val[1] for c_val in calibration.values()]
    grid_images = [c_val[3] for c_val in calibration.values()]
    spectral_coords = {}
    for i in list(coords.keys()):
        try:
            _, img_coord = utils.get_image(coords[i], grid, grid_images)
            spectral_coords[i] = img_coord
        except Exception:
            # Can't place it -> can't trust its spectrum.
            coords.pop(i, None)

    # 3. Left-edge bleed filter (always on; relax by lowering left_edge_cutoff).
    for i in list(coords.keys()):
        xs, _ys = spectral_coords[i]
        if xs - SPEC_LEFT < left_edge_cutoff:
            coords.pop(i, None)
            spectral_coords.pop(i, None)

    # 4. Overlap filter
    if remove_overlapping:
        x_thresh = SPEC_LEFT + SPEC_RIGHT   # windows touch in x within this span
        y_thresh = 2 * SPEC_HALF_ROWS       # share a row within this span
        to_remove = set()
        valid_keys = list(spectral_coords.keys())
        for a in range(len(valid_keys)):
            i = valid_keys[a]
            x1, y1 = spectral_coords[i]
            for b in range(a + 1, len(valid_keys)):
                j = valid_keys[b]
                x2, y2 = spectral_coords[j]
                if abs(x1 - x2) < x_thresh and abs(y1 - y2) <= y_thresh:
                    to_remove.add(i)
                    to_remove.add(j)
        for pid in to_remove:
            coords.pop(pid, None)

    return coords


def render_calibration_viewer(cal_file, fit_file):
    """Display the calibration illumination grid and let the user choose which
    illumination tiers (High/Medium/Low) feed spectra extraction."""
    import re
    import plotly.graph_objects as go

    st.subheader("Calibration data")
    try:
        calibration, _calib_fits = read_in_calibration([cal_file, fit_file])
    except Exception as e:
        st.error(f"Could not read calibration: {e}")
        return

    st.caption("Illumination = fitted spot amplitude (val[2][0]) at each grid point.")

    keys, gx, gy, raw = [], [], [], []
    for k, v in calibration.items():
        m = re.match(r"X(\d+)Y(\d+)", str(k))
        if not m:
            continue
        keys.append(k)
        gx.append(int(m.group(1)))
        gy.append(int(m.group(2)))
        raw.append(_grid_brightness(v))

    if not keys:
        st.warning("Calibration keys aren't in the expected 'X#Y#' grid format.")
        return

    gx = np.array(gx)
    gy = np.array(gy)
    raw = np.array(raw, dtype=float)
    max_raw = float(np.nanmax(raw)) or 1.0
    rel = raw / max_raw
    tiers = [_tier_for_relative(r) for r in rel]

    st.caption("Tiers by fraction of brightest point: Very low <40% · Low <60% "
               "· Medium <80% · High ≥80%.")

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        fig = go.Figure(go.Scatter(
            x=gx, y=gy, mode="markers",
            marker=dict(size=11, color=raw, colorscale="Viridis",
                        colorbar=dict(title="Illum."), line=dict(width=0)),
            customdata=np.stack([rel * 100.0, raw], axis=1),
            hovertemplate="X%{x} Y%{y}<br>%{customdata[0]:.0f}% (raw %{customdata[1]:.3g})<extra></extra>",
        ))
        fig.update_layout(title="Illumination", xaxis_title="Grid X",
                          yaxis_title="Grid Y", height=460,
                          margin=dict(l=40, r=10, t=40, b=40))
        fig.update_yaxes(scaleanchor="x", scaleratio=1)  # equal axes
        st.plotly_chart(fig, use_container_width=True, key="calib_heat")
    with pcol2:
        tfig = go.Figure()
        for label in ILLUM_TIER_LABELS:
            mask = np.array([t == label for t in tiers], dtype=bool)
            if mask.any():
                tfig.add_trace(go.Scatter(
                    x=gx[mask], y=gy[mask], mode="markers", name=label,
                    marker=dict(size=11, color=ILLUM_TIER_COLORS[label], line=dict(width=0)),
                    hovertemplate=f"{label}<br>X%{{x}} Y%{{y}}<extra></extra>",
                ))
        tfig.update_layout(title="Tier", xaxis_title="Grid X",
                           yaxis_title="Grid Y", height=460,
                           margin=dict(l=40, r=10, t=40, b=40),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        tfig.update_yaxes(scaleanchor="x", scaleratio=1)  # equal axes
        st.plotly_chart(tfig, use_container_width=True, key="calib_tier_heat")

    # Per-tier summary.
    summary = []
    for label in ILLUM_TIER_LABELS:
        sub = raw[np.array([t == label for t in tiers], dtype=bool)]
        if sub.size:
            summary.append({"Tier": label, "Points": int(sub.size),
                            "Min": f"{sub.min():.3g}", "Mean": f"{sub.mean():.3g}",
                            "Max": f"{sub.max():.3g}"})
    st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)

    include = st.multiselect(
        "Include these illumination tiers in spectra extraction",
        ILLUM_TIER_LABELS, default=ILLUM_TIER_LABELS,
        help="Each particle's illumination is interpolated from the calibration "
             "grid; only particles in the selected tiers are kept. Select all to "
             "disable the filter.",
    )
    st.session_state.calib_tier_filter = {"include": set(include)}
    if 0 < len(include) < len(ILLUM_TIER_LABELS):
        st.caption(f"Spectra extraction restricted to: {', '.join(include)} illumination.")
    st.divider()


# --- App ---
def run():
    with st.sidebar:
        st.header("Inputs v0.1")
        sif_files = utils.file_uploader_with_clear("SIF files", key="spectra_sif_uploads", type=["sif"], accept_multiple_files=True)

        background = st.file_uploader("Blank (optional)", type=["sif"], help='''
                    Upload image of an empty FOV under the same imaging conditions 
                    for background substraction. If absent, linear background estimation is performed.
        ''')
        # calibration_date = st.text_input("Calibration date", help="Please enter the date in the format like 250920")

        calibration_uploads = utils.file_uploader_with_clear(
            "Calibration files (saving_info + fits)", key="spectra_cal_uploads",
            accept_multiple_files=True,
            type=["pkl"], help=r'''
                Upload the two calibration .pkl files together — they're sorted
                automatically by filename:
                  • saving_info_YYMMDD.pkl (or ..._no_tracking.pkl) — the grid data
                  • YYMMDD_fits.pkl (or ..._fits_456_474_548_667_803.pkl) — the fits
                Both live in
                G:\Shared drives\SamPengLab\Alev_Studenikina\Multicolor\Heterogeneity\Calibration data
            ''')
        calibration_uploads = calibration_uploads or []
        cal_file, fit_file, cal_err = _classify_calibration_uploads(calibration_uploads)
        if cal_err:
            st.warning(cal_err)

        show_calibration = st.checkbox(
            "Show calibration data", value=False,
            help="Display the calibration illumination grid and choose which "
                 "illumination tiers (High/Medium/Low) feed spectra extraction.",
        )

        st.divider()
        st.header("Fitting")
        threshold = st.number_input("Threshold", min_value=0, value=2)
        radius_px = st.number_input("Radius (pixels)", min_value=1, value=2)
        use_custom_roi = st.checkbox(
            "Custom region (draw ROI)", value=False,
            help="Keep only particles inside a rectangle you draw on the first "
                 "image; the ROI coordinates are saved into the exported CSV.",
        )

        st.header("Display")
        # cmap = st.selectbox("Colormap", options=["gray","magma","viridis","plasma","hot","hsv"], index=0)
        # use_lognorm = st.checkbox("Log image scaling", value=True)
        vmax = st.number_input("vmax", value = 1000)
        show_colorbar = st.checkbox("Show colorbar", value=False)  

        st.header("Spectra")
        remove_overlapping = st.checkbox("Remove overlapping spectra", value=False)

        left_edge_cutoff = st.number_input(
            "Left-edge cutoff (drop spectra running off the left of the detector)",
            value=0,
            help=textwrap.dedent('''
                A particle's spectrum is sampled from (xs - 90) to (xs + 9) on the
                detector. If the left end (xs - 90) falls below this column the
                dispersed channel bleeds into the baseline, so the particle is
                dropped. 0 keeps only spectra that stay on-detector; raise it to
                add a safety margin, or set it negative to relax the filter.
            '''),
        )

        normalize = st.checkbox("Normalize spectra", value=False)

        no_dim = st.number_input("Exclude particles below a certain brightness threshold", value=0.0)

        scale_per_nm = st.checkbox("Scale intensity based on pixel bin size\n(can mess with background)", value=False)

        if st.button("Analyze"):
            st.session_state.analyze_clicked = True

    # Calibration viewer — available independent of the SIF analysis below.
    if show_calibration and cal_file is not None and fit_file is not None:
        render_calibration_viewer(cal_file, fit_file)
    elif show_calibration:
        st.info("Upload both calibration files (saving_info + fits) to view the grid.")

    if not sif_files:
        st.info("Upload SIF files to begin.")
        return

    if not cal_file or not fit_file:
        st.info("Please provide both calibration files (a 'saving_info' and a "
                "'fits' .pkl).")
        return

    # Prepare Matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import matplotlib.cm as cm

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if "analyze_clicked" not in st.session_state:
        st.session_state.analyze_clicked = False

    if st.session_state.analyze_clicked:
        # Localize
        u_data, _ = _process_files(sif_files, region="Mr Beam", threshold=threshold, signal="UCNP")
        full_images = just_read_in(sif_files)

        # Custom ROI: draw on the first displayed frame; one ROI filters every
        # file. Particles are kept only if they fall inside the rectangle.
        custom_roi = None
        if use_custom_roi:
            st.subheader("Custom region")
            first_frame = next(iter(full_images.values()), None)
            if first_frame is None:
                st.warning("Could not read an image to draw an ROI.")
            else:
                custom_roi = roi_tool.draw_roi(
                    first_frame, key="spectra_roi", cmap="gray",
                    flip_display=False,
                )
                if custom_roi is None:
                    st.info("Draw a rectangle above to restrict the analysis.")
                    return
        roi_cols = roi_tool.roi_columns(custom_roi)

        # Record the calibration source names before the loaded dicts shadow the
        # uploader objects (appended to the CSV for provenance).
        calib_file_name = getattr(cal_file, "name", "")
        calib_fits_name = getattr(fit_file, "name", "")
        calibration, calib_fits = read_in_calibration([cal_file, fit_file])

        background_image = None
        if background is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".sif") as tmp:
                tmp.write(background.getbuffer())
                tmp_path = tmp.name
            blank, metadata = sif_parser.np_open(tmp_path)
            blank = blank[0]  # (H, W)
            gainDAC = metadata['GainDAC'] or 1
            exposure_time = metadata['ExposureTime']
            accumulate_cycles = metadata['AccumulatedCycles']
            blank_cps = blank * (5.0 / gainDAC) / exposure_time / accumulate_cycles
            blank_cps = np.flipud(blank_cps)
            if blank_cps.shape[0] == 1:
                background_image = blank_cps[0]
            else:
                background_image = np.zeros((512, 512))
                for img in blank_cps:
                    background_image += img
                background_image /= blank_cps.shape[0]

        # Interpolated illumination field (grid-point spot amplitude), shared by
        # every particle for the CSV columns and trace thickness.
        illum_interp, illum_max = _illumination_field(calibration)

        all_spectra_data = []

        # One row per file: widefield image (left) beside its spectra (right).
        # Rendering each pair in the same st.columns row keeps them aligned no
        # matter how tall either figure is.
        for key, val in u_data.items():
            full_frame = full_images[key]
            df = val["df"]
            coords = _build_filtered_coords(
                df, calibration, no_dim, remove_overlapping, left_edge_cutoff,
                roi=custom_roi
            )
            colors = cm.rainbow(np.linspace(0, 1, len(coords.keys())))

            c_img, c_spec = st.columns(2)

            # --- Left: widefield image with localizations ---
            with c_img:
                fig, ax = plt.subplots(figsize=(5, 5))
                # Set vmin explicitly so a bright frame can't make matplotlib's
                # auto vmin exceed vmax ("minvalue must be <= maxvalue").
                vmin_eff = min(0.0, float(np.nanmin(full_frame)))
                vmax_eff = float(vmax) if float(vmax) > vmin_eff else vmin_eff + 1.0
                im_u = ax.imshow(full_frame, cmap="gray", vmin=vmin_eff,
                                 vmax=vmax_eff, origin="upper")
                ax.axis("off")
                if show_colorbar:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="10%", pad=0.2)
                    fig.colorbar(im_u, cax=cax)
                ax.set_title("\n".join(textwrap.wrap(key, width=25)))
                for i, id in enumerate(coords):
                    x, y = coords[id]
                    ax.scatter(x, y, s=2, c=colors[i])
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            # --- Right: extracted spectra (line thickness = illumination tier) ---
            with c_spec:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.set_title("Emission intensity per nm" if scale_per_nm
                             else "Emission intensity per pixel")
                for i, id in enumerate(coords):
                    x, y = coords[id]
                    raw_illum, rel_illum, tier_illum = _interp_tier(
                        illum_interp, illum_max, x, y
                    )
                    line_w = ILLUM_TIER_LW.get(tier_illum, 0.6)  # thicker = brighter
                    try:
                        wvl, spec, nms_per_pixel = get_spectrum(
                            np.array([x, y]), full_frame, calibration, calib_fits
                        )
                        # Background subtraction
                        if background is not None:
                            _bw, bkg_spec, _ = get_spectrum(
                                np.array([x, y]), background_image, calibration, calib_fits
                            )
                            res = fit_template_sigma_clip(spec, bkg_spec, 10, 1)
                            intensity_sans_bkg = spec - res[0]
                        else:
                            intensity_sans_bkg = spec - np.min(spec)

                        # Scaling + normalization
                        final_intensity = intensity_sans_bkg
                        if scale_per_nm:
                            final_intensity = final_intensity / nms_per_pixel
                        if normalize:
                            final_intensity = utils.normalize(final_intensity)

                        ax.plot(wvl, final_intensity, c=colors[i], linewidth=line_w)

                        # Extra columns are appended after the four Process Spectra
                        # requires, so the file stays compatible with it.
                        for w, inten in zip(wvl, final_intensity):
                            all_spectra_data.append({
                                "File": key,
                                "Particle_ID": id,
                                "Wavelength_nm": w,
                                "Intensity": inten,
                                "Particle_X": x,
                                "Particle_Y": y,
                                "Raw_Illumination": raw_illum,
                                "Relative_Illumination": rel_illum,
                                "Illumination_Tier": tier_illum,
                                "Calibration_File": calib_file_name,
                                "Calibration_Fits": calib_fits_name,
                                **roi_cols,
                            })
                    except Exception:
                        st.error(f"Error extracting spectrum for particle {id}")
                        continue

                ax.set_xlabel("Wavelength (nm)")
                from matplotlib.lines import Line2D
                legend_handles = [
                    Line2D([0], [0], color="black", lw=ILLUM_TIER_LW[t], label=t)
                    for t in ILLUM_TIER_LABELS
                ]
                ax.legend(handles=legend_handles, title="Illumination",
                          fontsize=7, title_fontsize=8, loc="upper right")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # --- CSV download (all files) ---
        if all_spectra_data:
            csv = pd.DataFrame(all_spectra_data).to_csv(index=False).encode("utf-8")
        else:
            csv = b""
        stems = [Path(name).stem for name in u_data.keys()]
        if not stems:
            csv_name = "all_extracted_spectra.csv"
        elif len(stems) <= 3:
            csv_name = "spectra_" + "_".join(stems) + ".csv"
        else:
            csv_name = f"spectra_{stems[0]}_and_{len(stems) - 1}_more.csv"
        st.download_button(
            label="Download All Spectra (CSV)",
            data=csv, file_name=csv_name, mime="text/csv",
        )
                            

                  
if __name__ == "__main__":
    run()