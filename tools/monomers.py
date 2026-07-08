# tools/monomers.py
import streamlit as st
import os, io, tempfile, hashlib, re
import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
from matplotlib.lines import Line2D

from utils import plot_brightness, plot_histogram, HWT_aesthetic, file_uploader_with_clear
from tools.process_files import process_files
from tools import roi as roi_tool

CATEGORY_ORDER = ["Monomers", "Dimers", "Trimers", "Multimers"]
CATEGORY_COLORS = {
    "Monomers":  "#029E73",  # green
    "Dimers":    "#0173B2",  # blue
    "Trimers":   "#DE8F05",  # orange
    "Multimers": "#D55E00",  # red
}

def thresholds_from_single_brightness(single_ucnp_brightness: float):
    """
    Return brightness thresholds [t1, t2, t3] in pps
    that split Monomers < 2x, 2x<=Dimers<3x, 3x<=Trimers<4x, >=4x Multimers.
    """
    t1 = 2.0 * single_ucnp_brightness
    t2 = 3.0 * single_ucnp_brightness
    t3 = 4.0 * single_ucnp_brightness
    return [t1, t2, t3]

def _hash_file(uploaded_file):
    uploaded_file.seek(0)
    h = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return h


def _normalize_saved_values(values_iterable):
    """
    Accept values that might be either:
      - (display_name, temp_path) tuples (new format), OR
      - plain path strings from older runs (legacy)
    Return a list of (display_name, temp_path) tuples.
    """
    normalized = []
    for v in values_iterable:
        if isinstance(v, (tuple, list)) and len(v) == 2:
            name, path = v
            normalized.append((str(name), str(path)))
        elif isinstance(v, str):
            name = os.path.basename(v)
            normalized.append((name, v))
    return normalized

from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

# --- Concentration-estimation constants -----------------------------------
AVOGADRO = 6.02214076e23                 # particles / mol
PLANE_THICKNESS_UM = 0.200               # 200 nm optical section of a PSF


def _format_molarity(molar: float) -> str:
    """Format a molar concentration with a common SI prefix (M, mM, µM, nM…)."""
    if not np.isfinite(molar) or molar <= 0:
        return "0 M"
    prefixes = [
        (1.0, "M"), (1e-3, "mM"), (1e-6, "µM"), (1e-9, "nM"),
        (1e-12, "pM"), (1e-15, "fM"), (1e-18, "aM"),
    ]
    for scale, unit in prefixes:
        if molar >= scale:
            return f"{molar / scale:.3g} {unit}"
    return f"{molar / 1e-18:.3g} aM"


def estimate_concentration(ppv, fov_area_um2, dilution,
                           plane_thickness_um=PLANE_THICKNESS_UM):
    """
    Estimate stock molarity from the average particles-per-view (ppv).

    Concentration is fundamentally (particles observed) / (volume observed),
    where the observed volume is the imaged FOV area times the 200 nm optical
    section thickness. All the well-geometry / droplet / plane-count terms
    cancel algebraically, so they are deliberately omitted here:

        M_stock = ppv * dilution / (A_fov * t * N_A)

    Molarity is intensive, so no bulk (stock/prep) volume enters the formula.
    """
    obs_volume_um3 = fov_area_um2 * plane_thickness_um
    obs_volume_l = obs_volume_um3 * 1e-15                   # 1 µm³ = 1e-15 L

    conc_diluted = ppv / obs_volume_l                      # particles / L (imaged sample)
    conc_stock = conc_diluted * dilution                   # particles / L (stock)
    molarity = conc_stock / AVOGADRO

    return {
        "ppv": ppv,
        "fov_area_um2": fov_area_um2,
        "obs_volume_um3": obs_volume_um3,
        "conc_diluted_M": conc_diluted / AVOGADRO,
        "molarity": molarity,
    }


def _fit_circle_lsq(x, y):
    """Algebraic (Kåsa) least-squares circle fit. Returns (cx, cy, r)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = sol
    cx = -D / 2.0
    cy = -E / 2.0
    val = cx * cx + cy * cy - F
    r = float(np.sqrt(val)) if val > 0 else float("nan")
    return float(cx), float(cy), r


def fit_aperture_circle(image, pix_size_um=None, min_diameter_um=20.0):
    """
    Fit the circular illuminated aperture (field stop) in a 2D image. Returns
    {cx, cy, r, area_px, fit} in pixels, or None if no plausible disk is found.

    The aperture may be partially occluded (e.g. clipped on the left), so a
    plain area-equivalent radius (√(area/π)) and centroid are biased. Instead we
    fit a circle by least squares to the *arc* of the illuminated boundary, with
    iterative outlier rejection to shed the straight occlusion edge — giving an
    accurate center/radius for the overlay. `area_px` remains the actual
    illuminated pixel count (the true counting area for concentration).

    Robustness for very bright signal on a flat background (where Otsu can latch
    onto a single bright particle and fit a tiny circle):
      - the component containing the ROI center (image center) is preferred over
        the largest one, so the fit is seeded on the aperture, not a stray blob;
      - fits smaller than `min_diameter_um` (needs `pix_size_um`) are rejected
        (return None) so the caller falls back to the full crop area.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2 or img.size == 0:
        return None
    try:
        from scipy import ndimage
        from skimage.filters import threshold_otsu
    except Exception:
        return None

    sm = ndimage.gaussian_filter(img, sigma=2)
    finite = sm[np.isfinite(sm)]
    if finite.size == 0 or float(np.ptp(finite)) == 0:
        return None
    try:
        thr = float(threshold_otsu(sm))
    except Exception:
        return None

    mask = sm > thr
    if mask.sum() < 0.02 * mask.size:  # need a plausibly disk-sized bright region
        return None

    lbl, n = ndimage.label(mask)
    if n == 0:
        return None

    # Prefer the component covering the ROI center; else the largest one.
    h, w = mask.shape
    center_label = int(lbl[h // 2, w // 2])
    if center_label != 0:
        chosen = center_label
    else:
        sizes = ndimage.sum(np.ones_like(lbl, dtype=float), lbl, index=np.arange(1, n + 1))
        chosen = 1 + int(np.argmax(sizes))
    disk = ndimage.binary_fill_holes(lbl == chosen)
    area_px = float(disk.sum())

    # Minimum radius (px) implied by the minimum FOV diameter.
    min_r_px = 0.0
    if pix_size_um and min_diameter_um:
        min_r_px = (float(min_diameter_um) / 2.0) / float(pix_size_um)

    def _valid(res):
        return res is not None and res["r"] >= min_r_px

    cy0, cx0 = ndimage.center_of_mass(disk)
    fallback = {"cx": float(cx0), "cy": float(cy0),
                "r": float(np.sqrt(area_px / np.pi)), "area_px": area_px, "fit": "area"}

    # Boundary pixels of the illuminated region.
    boundary = disk & ~ndimage.binary_erosion(disk)
    ys, xs = np.nonzero(boundary)
    if xs.size < 8:
        return fallback if _valid(fallback) else None

    # Drop boundary pixels on the image frame (crop edges are not real arc).
    on_frame = (xs <= 0) | (ys <= 0) | (xs >= w - 1) | (ys >= h - 1)
    xf, yf = xs[~on_frame].astype(float), ys[~on_frame].astype(float)
    if xf.size < 8:
        xf, yf = xs.astype(float), ys.astype(float)

    cx, cy, r = _fit_circle_lsq(xf, yf)
    # Iteratively reject the straight occlusion edge (large-residual points).
    for _ in range(8):
        if not np.isfinite(r):
            break
        res = np.abs(np.sqrt((xf - cx) ** 2 + (yf - cy) ** 2) - r)
        med = np.median(res)
        mad = np.median(np.abs(res - med)) + 1e-9
        keep = res < med + 2.0 * 1.4826 * mad
        if keep.sum() < 8 or keep.all():
            break
        xf, yf = xf[keep], yf[keep]
        cx, cy, r = _fit_circle_lsq(xf, yf)

    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(r)) or r <= 0:
        return fallback if _valid(fallback) else None
    result = {"cx": cx, "cy": cy, "r": r, "area_px": area_px, "fit": "circle"}
    return result if _valid(result) else None


def plot_monomer_brightness(
    image_data_cps,
    df,
    show_fits=True,
    plot_brightness_histogram=False,
    normalization=False,
    pix_size_um=0.1,
    cmap='plasma',
    single_ucnp_brightness=None,
    *,
    interactive=False,
    dragmode='zoom'
):
    """
    Plot brightness map and overlay Gaussian-fit circles colored by brightness category.
    If interactive=True returns a Plotly figure; otherwise Matplotlib.
    """
    if not interactive:
        fig_width, fig_height = 5, 5
        scale = fig_width / 5
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        norm = LogNorm() if normalization else None
        im = ax.imshow(image_data_cps + 1, cmap=cmap, norm=norm, origin='lower')
        ax.tick_params(axis='both', length=0,
                       labelleft=False, labelright=False,
                       labeltop=False, labelbottom=False)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10 * scale)
        cbar.set_label('pps', fontsize=10 * scale)

        if single_ucnp_brightness is None:
            single_ucnp_brightness = float(np.mean(image_data_cps))

        t1, t2, t3 = thresholds_from_single_brightness(single_ucnp_brightness)

        if show_fits:
            for _, row in df.iterrows():
                x_px = row['x_pix']
                y_px = row['y_pix']
                brightness_pps = row['brightness_integrated']
                brightness_kpps = brightness_pps / 1000.0

                radius_px = 3 * max(row['sigx_fit'], row['sigy_fit']) / pix_size_um

                if brightness_pps < t1:
                    cat = "Monomers"
                elif brightness_pps < t2:
                    cat = "Dimers"
                elif brightness_pps < t3:
                    cat = "Trimers"
                else:
                    cat = "Multimers"

                circle_color = CATEGORY_COLORS[cat]
                circle = Circle((x_px, y_px), radius_px,
                                color=circle_color, fill=False,
                                linewidth=1.25 * scale, alpha=0.95)
                ax.add_patch(circle)

                ax.text(x_px + 7.5, y_px + 7.5,
                        f"{brightness_kpps:.1f} kpps",
                        color='white', fontsize=7 * scale,
                        ha='center', va='center')

            legend_elements = [
                Line2D([0], [0], color=CATEGORY_COLORS["Monomers"], lw=2, label="Monomers"),
                Line2D([0], [0], color=CATEGORY_COLORS["Dimers"],   lw=2, label="Dimers"),
                Line2D([0], [0], color=CATEGORY_COLORS["Trimers"],  lw=2, label="Trimers"),
                Line2D([0], [0], color=CATEGORY_COLORS["Multimers"],lw=2, label="Multimers"),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=8, frameon=False, labelcolor='white')

        plt.tight_layout()
        HWT_aesthetic()
        return fig

    # --- Interactive Plotly path ---
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np

    if single_ucnp_brightness is None:
        single_ucnp_brightness = float(np.mean(image_data_cps))
    t1, t2, t3 = thresholds_from_single_brightness(single_ucnp_brightness)

    cmap_map = {
        "magma": "Magma", "viridis": "Viridis", "plasma": "Plasma",
        "hot": "Hot", "gray": "Gray", "hsv": "HSV", "cividis": "Cividis", "inferno": "Inferno"
    }
    plotly_scale = cmap_map.get(cmap, "plasma")

    img = image_data_cps.astype(float)
    if normalization:
        eps = max(float(np.percentile(img, 0.01)), 1e-9)
        img_display = np.log10(np.clip(img + 1.0, eps, None))
    else:
        img_display = img

    fig = px.imshow(img_display, origin="lower", aspect="equal", color_continuous_scale=plotly_scale)
    img_custom = np.expand_dims(img, axis=-1)
    fig.data[0].customdata = img_custom
    fig.data[0].hovertemplate = (
        "x=%{x:.0f}px<br>y=%{y:.0f}px<br>pps=%{customdata[0]:.1f}<extra></extra>"
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        dragmode=dragmode,
        coloraxis_colorbar=dict(
            title="pps" if not normalization else "log10(pps)",
            yanchor="middle",
            y=0.5,
            lenmode="fraction",
            len=0.8,
            thickness=20,
        ),
        xaxis_title="X (px)",
        yaxis_title="Y (px)"
    )

    if df is not None and not df.empty:
        xs = df['x_pix'].to_numpy()
        ys = df['y_pix'].to_numpy()
        rs = (3 * np.maximum(df['sigx_fit'].to_numpy(), df['sigy_fit'].to_numpy()) / pix_size_um).astype(float)
        br = df['brightness_integrated'].to_numpy()
        br_k = (br / 1000.0).astype(float)
        cats = np.where(br < t1, 'Monomers', np.where(br < t2, 'Dimers', np.where(br < t3, 'Trimers', 'Multimers')))
        colors = [CATEGORY_COLORS[c] for c in cats]

        custom = np.stack([br_k, cats], axis=1)
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='markers',
            marker=dict(size=1, opacity=0),
            name='Fits',
            customdata=custom,
            hovertemplate="x=%{x:.2f}px<br>y=%{y:.2f}px<br>brightness=%{customdata[0]:.1f} kpps<br>%{customdata[1]}<extra></extra>",
            showlegend=False,
        ))

        if show_fits:
            shapes=[]
            for x,y,r,c in zip(xs,ys,rs,colors):
                shapes.append(dict(type='circle', xref='x', yref='y', x0=x-r, x1=x+r, y0=y-r, y1=y+r,
                                   line=dict(width=1.5, color=c), fillcolor='rgba(0,0,0,0)', layer='above'))
            fig.update_layout(shapes=shapes)

            fig.add_trace(go.Scatter(
                x=xs + 7.5,
                y=ys + 7.5,
                mode='text',
                text=[f"{v:.1f} kpps" for v in br_k],
                textfont=dict(color='white', size=10),
                textposition='middle center',
                showlegend=False,
                hoverinfo='skip',
            ))

            for cat in CATEGORY_ORDER:
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                         marker=dict(color=CATEGORY_COLORS[cat], size=10),
                                         name=cat))

    h, w = img.shape
    fig.update_xaxes(range=[-0.5, w - 0.5], constrain='domain', showgrid=False, zeroline=False)
    fig.update_yaxes(range=[-0.5, h - 0.5], scaleanchor='x', scaleratio=1, showgrid=False, zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def _process_files_cached(saved_records, region, threshold, signal, pix_size_um=0.1, sig_threshold=0.3, roi=None):
    class _FakeUpload:
        def __init__(self, name, path):
            self.name = name
            self._path = path
        def getbuffer(self):
            with open(self._path, "rb") as f:
                return memoryview(f.read())

    uploads = [_FakeUpload(name, path) for (name, path) in saved_records]
    pf = getattr(process_files, "__wrapped__", None)
    if pf is None:
        raise RuntimeError("process_files.__wrapped__ not found; cannot bypass Streamlit cache.")

    # FORWARD the parameters to process_files.__wrapped__
    return pf(
        uploads,
        region=region,
        threshold=threshold,
        signal=signal,
        pix_size_um=pix_size_um,
        sig_threshold=sig_threshold,
        roi=roi,
    )


def run():

    # Persistent state
    if "saved_files" not in st.session_state:
        # key -> (display_name, temp_path)  (legacy may be plain path str)
        st.session_state.saved_files = {}
    if "processed" not in st.session_state:
        st.session_state.processed = None  # (processed_data, combined_df)
    if "selected_file_name" not in st.session_state:
        st.session_state.selected_file_name = None
    normalized_records = []
    selected_file_name = st.session_state.get("selected_file_name")
    threshold = None
    signal = None
    region = None

    with st.sidebar:
        uploaded_files = file_uploader_with_clear(
            "Upload .sif file", key="monomers_uploads", type=["sif"], accept_multiple_files=True
        )

        # --- SYNC PHASE: make session match current uploader selection ---
        prev_keys = set(st.session_state.saved_files.keys())
        changed = False

        # 1) Add new uploads
        current_keys = set()
        if uploaded_files:
            for f in uploaded_files:
                key = f"{f.name}:{_hash_file(f)}"
                current_keys.add(key)
                if key not in st.session_state.saved_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".sif") as tmp:
                        tmp.write(f.getbuffer())
                        st.session_state.saved_files[key] = (f.name, tmp.name)
                    changed = True

        # 2) Remove files no longer present in the uploader
        #    (also clean up their temp files)
        stale_keys = [k for k in st.session_state.saved_files.keys() if k not in current_keys]
        for k in stale_keys:
            val = st.session_state.saved_files[k]
            name, path = (val if isinstance(val, (tuple, list)) and len(val) == 2 else (os.path.basename(val), val))
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
            del st.session_state.saved_files[k]
            changed = True

        # 3) If the set of saved files changed (added/removed), invalidate results
        if changed or (set(st.session_state.saved_files.keys()) != prev_keys):
            st.session_state.processed = None
            # If selected file no longer exists, clear selection
            current_names = [v[0] if isinstance(v, (tuple, list)) else os.path.basename(v)
                             for v in st.session_state.saved_files.values()]
            if st.session_state.selected_file_name not in current_names:
                st.session_state.selected_file_name = None


        # --- UI to select file & params (based on synced saved_files) ---
        current_values = list(st.session_state.saved_files.values())
        normalized_records = _normalize_saved_values(current_values)
        file_options = [display for (display, _) in normalized_records]

        if file_options:
            default_index = 0
            if st.session_state.selected_file_name in file_options:
                default_index = file_options.index(st.session_state.selected_file_name)
            selected_file_name = st.selectbox(
                "Select sif to display:", options=file_options, index=default_index
            )
            st.session_state.selected_file_name = selected_file_name

            microscope = st.selectbox(
                "Microscope",
                options=["MCL", "Nikon / Mr Beam"],
                help="Microscope used to acquire the images.",
                key="mono_microscope",
            )

            # Parameters (kept to preserve existing UI)
            threshold = st.number_input(
                                        "Threshold", min_value=1, value=1,
                                        help=("Stringency of fit, higher value is more selective:\n"
                                              "- UCNP signal sets absolute peak cut off\n"
                                              "- Dye signal sets sensitivity of blob detection"),
                                        key="mono_threshold",
                                        )

            signal = st.selectbox(
                                    "Signal", options=["UCNP", "dye"],
                                    help=("Changes detection method:\n"
                                          "- UCNP for high SNR (sklearn peakfinder)\n"
                                          "- dye for low SNR (sklearn blob detection)"),
                                    key="mono_signal",
                                    )
            if microscope == "MCL":
                diagram = """ Splits sif into quadrants (256x256 px):
                                ┌─┬─┐
                                │ 1 │ 2 │
                                ├─┼─┤
                                │ 3 │ 4 │
                                └─┴─┘
                                Blue=1, Green=2, Red=3, NIR=4
                                """
                # Display channel names; map to underlying quadrant regions.
                mcl_region_map = {"Blue": "1", "Green": "2", "Red": "3", "NIR": "4", "all": "all"}
                region_label = st.selectbox(
                    "Region", options=["Blue", "Green", "Red", "NIR", "all"], help=diagram,
                    key="mono_region_label",
                )
                region = mcl_region_map[region_label]
            else:
                # Nikon / Mr Beam: no quadrant selection; fixed Mr Beam crop.
                region = "Mr Beam"

            use_custom_roi = st.checkbox(
                "Custom region (draw ROI)", value=False,
                help="Restrict detection to a rectangle you draw on the selected "
                     "file (shown in the main panel); overrides the Region above. "
                     "The ROI coordinates are saved into the exported CSV.",
                key="mono_use_roi",
            )
            st.session_state["monomers_use_roi"] = use_custom_roi

            # --- Acquisition & sample metadata (used for concentration estimation) ---
            objective_mag = st.number_input(
                "Objective magnification",
                min_value=1, value=60, step=1,
                help="Objective magnification used during acquisition.",
                key="mono_objective_mag",
            )
            # Pixel size is derived from the microscope (+ magnification), not
            # entered manually.
            if microscope == "MCL":
                pix_size_um = 0.1011
                pix_help = "Based on LJ measurement in Blue Channel (Spring 2026)."
            else:  # Nikon / Mr Beam
                pix_size_um = 0.107 * 100.0 / float(objective_mag)
                pix_help = (
                    "Nikon: 0.107 µm at 100×; scales as 0.107 × 100 / magnification "
                    f"= {pix_size_um:.4f} µm at {objective_mag}×."
                )
            st.number_input(
                "Pixel size (µm)",
                value=float(pix_size_um), format="%.4f", disabled=True,
                help=pix_help,
            )
            dilution_str = st.text_input(
                "Dilution",
                value="1E3",
                help="Sample dilution factor (scientific notation accepted, e.g. 1E3).",
                key="mono_dilution",
            )
            try:
                dilution = float(dilution_str)
            except ValueError:
                st.warning("Enter a valid dilution (e.g. 1E3).")
                dilution = None

            estimate_conc = st.checkbox(
                "Estimate concentration",
                value=False,
                help="Estimate stock molarity from the average particles per field of view.",
                key="mono_estimate_conc",
            )
            if estimate_conc:
                axial_depth_um = st.number_input(
                    "Axial detection depth (µm)",
                    min_value=0.001, value=0.5, step=0.05, format="%.3f",
                    help=("Thickness of the imaged slab used as the counting volume "
                          "(A_fov × depth) the "
                          "depth over which a particle is still detected/fit, including "
                          "out-of-focus ones. Concentration scales as 1/depth."),
                    key="mono_axial_depth",
                )
            else:
                axial_depth_um = 0.5

            st.session_state["objective_mag"] = objective_mag
            st.session_state["microscope"] = microscope
            st.session_state["dilution"] = dilution
            st.session_state["pix_size_um"] = pix_size_um
            st.session_state["estimate_conc"] = estimate_conc
            st.session_state["axial_depth_um"] = axial_depth_um

            cmap_options = ["gray", "viridis", "magma", "hot",  "hsv"]
            current_cmap = st.session_state.get("monomers_cmap", "plasma")
            try:
                default_cmap_index = cmap_options.index(current_cmap)
            except ValueError:
                default_cmap_index = cmap_options.index("plasma")
            cmap = st.selectbox("Colormap", options=cmap_options, index=default_cmap_index)
            st.session_state["monomers_cmap"] = cmap
            show_fits = st.checkbox("Show fits", value=True, key="mono_show_fits")
            normalization = st.checkbox("Log Image Scaling", value=True, key="mono_normalization")
            save_format = st.selectbox(
                "Download format", ["svg", "png", "jpeg"], key="mono_save_format"
            ).lower()

            # Process automatically. _process_files_cached is @st.cache_data keyed on
            # (saved_records, region, threshold, signal, pix_size_um), so tuning any of
            # these re-runs analysis, while display-only params (cmap, bins, brightness
            # range…) hit the cache and re-render instantly — no "Process" button needed.
            saved_records = tuple(normalized_records)
            # Custom ROI (drawn in the main panel on a prior rerun) overrides the
            # region. If enabled but not yet drawn, hold off until it exists.
            custom_roi = (st.session_state.get("monomers_roi")
                          if use_custom_roi else None)
            if use_custom_roi and custom_roi is None:
                st.session_state.processed = None
            else:
                with st.spinner("Processing…"):
                    processed_data, combined_df = _process_files_cached(
                        saved_records,
                        region=region,
                        threshold=threshold,
                        signal=signal,
                        pix_size_um=pix_size_um,
                        roi=custom_roi,
                    )
                st.session_state.processed = (processed_data, combined_df)

        else:
            st.session_state.selected_file_name = None
            st.session_state.processed = None

    # CUSTOM ROI: draw on the selected file (main panel) and stash for the next
    # rerun's processing. One ROI is applied to every file.
    active_roi = None
    if st.session_state.get("monomers_use_roi") and file_options:
        st.subheader("Custom region")
        sel_name = st.session_state.get("selected_file_name") or file_options[0]
        sel_path = next((p for (d, p) in normalized_records if d == sel_name), None)
        ref_img = roi_tool.read_sif_raw_from_path(sel_path) if sel_path else None
        if ref_img is None:
            st.warning("Could not read the selected file to draw an ROI.")
        else:
            active_roi = roi_tool.draw_roi(
                ref_img, key="monomers_roi_canvas",
                cmap=st.session_state.get("monomers_cmap", "plasma"), log=True,
            )
            st.session_state["monomers_roi"] = active_roi
            if active_roi is None:
                st.info("Draw a rectangle above to run the analysis inside it.")

    # DISPLAY
    if st.session_state.get("processed"):
        processed_data, combined_df = st.session_state.processed
        roi_tool.stamp_roi(combined_df, active_roi)
        if combined_df is not None and not combined_df.empty:
            st.download_button(
                "Download data (CSV)",
                combined_df.to_csv(index=False).encode("utf-8"),
                file_name="monomers_compiled.csv",
                mime="text/csv",
                key="monomers_csv",
            )
        selected_file_name = st.session_state.get("selected_file_name")
        if not selected_file_name and processed_data:
            selected_file_name = next(iter(processed_data.keys()))
            st.session_state.selected_file_name = selected_file_name

        data_to_plot = processed_data.get(selected_file_name) if selected_file_name else None
        df_selected = data_to_plot.get("df") if data_to_plot else None
        image_data_cps = data_to_plot.get("image") if data_to_plot else None

        microscope = st.session_state.get("microscope", "MCL")
        pix = float(st.session_state.get("pix_size_um", 0.107))

        # Both MCL and Nikon image through a circular aperture (field stop), so
        # fit it in either case: used to overlay on the preview and to get an
        # accurate FOV area for concentration. Seeded on the ROI center with a
        # 20 µm minimum diameter. If the fit fails, fall back to a circle
        # inscribed in the region (not the full square crop). Skipped for the
        # 'custom' region (not a full circular aperture).
        aperture = None
        if image_data_cps is not None and str(region) != "custom":
            aperture = fit_aperture_circle(image_data_cps, pix_size_um=pix,
                                           min_diameter_um=20.0)
            if aperture is None and getattr(image_data_cps, "ndim", 0) >= 2:
                h_px, w_px = image_data_cps.shape[:2]
                r_px = min(h_px, w_px) / 2.0
                aperture = {"cx": w_px / 2.0, "cy": h_px / 2.0, "r": r_px,
                            "area_px": float(np.pi * r_px ** 2), "fit": "inscribed"}

        top_left, top_right = st.columns(2)
        fig_pie = None
        thresholds = None
        single_ucnp_brightness_value = st.session_state.get("single_ucnp_brightness")

        with top_right:
            if not combined_df.empty:
                brightness_vals = combined_df['brightness_integrated'].values
                default_min_val = float(np.min(brightness_vals))
                default_max_val = float(np.max(brightness_vals))

                mc1, mc2 = st.columns(2)
                user_min_val_str = mc1.text_input("Min Brightness (pps)", value=f"{default_min_val:.2e}")
                user_max_val_str = mc2.text_input("Max Brightness (pps)", value=f"{default_max_val:.2e}")

                try:
                    user_min_val = float(user_min_val_str)
                    user_max_val = float(user_max_val_str)
                except ValueError:
                    st.warning("Please enter valid numbers (you can use scientific notation like 1e6).")
                    st.stop()

                if user_min_val >= user_max_val:
                    st.warning("Min brightness must be less than max brightness.")
                else:
                    if single_ucnp_brightness_value is None:
                        default_spb = float(np.mean(brightness_vals))
                    else:
                        default_spb = float(single_ucnp_brightness_value)
                    default_spb = float(np.clip(default_spb, user_min_val, user_max_val))

                    bc1, bc2 = st.columns(2)
                    num_bins = bc1.number_input("# Bins:", value=50, key="mono_num_bins")
                    single_ucnp_brightness_value = bc2.number_input(
                        "Single Particle Brightness (pps)",
                        min_value=user_min_val,
                        max_value=user_max_val,
                        value=default_spb,
                    )
                    st.session_state["single_ucnp_brightness"] = float(single_ucnp_brightness_value)

                    thresholds = thresholds_from_single_brightness(single_ucnp_brightness_value)

                    fig_hist_final, _, _ = plot_histogram(
                        combined_df,
                        min_val=user_min_val,
                        max_val=user_max_val,
                        num_bins=num_bins,
                        thresholds=thresholds,
                    )
                    fig_hist_final.set_size_inches(5, 3)
                    fig_hist_final.tight_layout()
                    st.pyplot(fig_hist_final)
                    plt.close(fig_hist_final)

                    bins_for_pie = [user_min_val] + [t for t in thresholds if user_min_val < t < user_max_val] + [user_max_val]
                    bins_for_pie = sorted(bins_for_pie)
                    num_bins_pie = len(bins_for_pie) - 1
                    labels_for_pie = CATEGORY_ORDER[:num_bins_pie]

                    if len(labels_for_pie) != num_bins_pie:
                        st.warning(f"Label/bin mismatch: {len(labels_for_pie)} labels for {num_bins_pie} bins.")
                    else:
                        categories = pd.cut(
                            combined_df['brightness_integrated'],
                            bins=bins_for_pie,
                            right=False,
                            include_lowest=True,
                            labels=labels_for_pie,
                        )
                        category_counts = categories.value_counts().reset_index()
                        category_counts.columns = ['Category', 'Count']

                        fig_pie = px.pie(
                            category_counts,
                            values='Count',
                            names='Category',
                            color='Category',
                            color_discrete_map=CATEGORY_COLORS,
                        )
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=11)
                        fig_pie.update_layout(
                            font=dict(size=11),
                            showlegend=False,
                            height=260,
                            margin=dict(l=0, r=0, t=0, b=0),
                        )

        single_ucnp_brightness_value = st.session_state.get(
            "single_ucnp_brightness", single_ucnp_brightness_value
        )


        with top_left:
            if not selected_file_name:
                st.info("Upload .sif files to view results.")
            elif data_to_plot is None:
                st.error(f"Data for file '{selected_file_name}' not found.")
            else:
                fig_image = plot_monomer_brightness(
                    image_data_cps,
                    df_selected,
                    show_fits=show_fits,
                    normalization=normalization,
                    pix_size_um=pix,
                    cmap=st.session_state.get("monomers_cmap", "plasma"),
                    single_ucnp_brightness=single_ucnp_brightness_value,
                    interactive=True,
                )
                if aperture is not None:
                    a = aperture
                    fig_image.add_shape(
                        type='circle', xref='x', yref='y',
                        x0=a['cx'] - a['r'], x1=a['cx'] + a['r'],
                        y0=a['cy'] - a['r'], y1=a['cy'] + a['r'],
                        line=dict(color='cyan', width=2, dash='dot'),
                    )
                fig_image.update_layout(height=430, margin=dict(l=0, r=0, t=20, b=0))
                fmt = save_format.lower()
                if fmt not in {"png", "jpeg", "jpg", "svg", "webp"}:
                    fmt = "png"
                st.plotly_chart(
                    fig_image,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d", "toggleSpikelines"],
                        "toImageButtonOptions": {"format": fmt},
                    },
                )
                html_bytes = fig_image.to_html().encode("utf-8")
                st.download_button(
                    label="Download PSFs (HTML)",
                    data=html_bytes,
                    file_name=f"{selected_file_name}.html",
                    mime="text/html",
                )

        processed = st.session_state.processed[0]
        if processed:
            psf_counts = {os.path.basename(name): len(processed[name]["df"]) for name in processed.keys()}

            def extract_sif_number(filename):
                m = re.search(r'_([0-9]+)\.sif$', filename)
                return m.group(1) if m else filename

            file_names = [extract_sif_number(n) for n in psf_counts.keys()]
            counts = list(psf_counts.values())
            mean_count = np.mean(counts) if counts else 0

            # ---------------- Row 2: pie · PSF counts · concentration ------
            r2_pie, r2_counts, r2_conc = st.columns(3)

            with r2_pie:
                st.caption("Population breakdown")
                if fig_pie is not None:
                    st.plotly_chart(fig_pie, use_container_width=True)

            with r2_counts:
                st.caption("PSF counts per file")
                fig_count, ax_count = plt.subplots(figsize=(4.5, 3))
                ax_count.bar(file_names, counts)
                ax_count.axhline(
                    mean_count,
                    color=CATEGORY_COLORS["Multimers"],
                    linestyle='--',
                    label=f'Avg = {mean_count:.1f}',
                    linewidth=0.8,
                )
                ax_count.set_ylabel("# Fit PSFs", fontsize=10)
                ax_count.set_xlabel("SIF #", fontsize=10)
                ax_count.legend(fontsize=9)
                ax_count.tick_params(axis='x', labelsize=8)
                ax_count.tick_params(axis='y', labelsize=8)
                HWT_aesthetic()
                fig_count.tight_layout()
                st.pyplot(fig_count)
                plt.close(fig_count)

            with r2_conc:
                st.caption("Concentration estimate")
                if not st.session_state.get("estimate_conc"):
                    st.caption("Enable *Estimate concentration* in the sidebar.")
                else:
                    dilution = st.session_state.get("dilution")
                    axial_depth_um = float(st.session_state.get("axial_depth_um", 0.5))

                    # FOV area: fitted aperture, else an inscribed-circle fallback
                    # (set above), else the full crop for the 'custom' region.
                    if aperture is not None:
                        fov_area = aperture["area_px"] * (pix ** 2)
                        d_um = 2 * aperture["r"] * pix
                        if aperture.get("fit") == "inscribed":
                            fov_desc = (f"inscribed circle Ø{2 * aperture['r']:.0f} px "
                                        f"({d_um:.1f} µm) — aperture fit failed")
                        else:
                            # Counting area = actual illuminated pixels (excludes any
                            # occluded part). Fitted r/Ø describe the full circle.
                            fov_desc = (f"illuminated aperture: {aperture['area_px']:.0f} px "
                                        f"(fit r={aperture['r']:.0f} px, Ø{d_um:.1f} µm)")
                    elif image_data_cps is not None and getattr(image_data_cps, "ndim", 0) >= 2:
                        fh, fw = image_data_cps.shape[:2]
                        fov_area = fh * fw * (pix ** 2)
                        fov_desc = f"{fw}×{fh} px full crop (custom region)"
                    else:
                        fov_area = 0.0
                        fov_desc = "unknown"

                    if dilution is None:
                        st.warning("Enter a valid dilution.")
                    elif not counts:
                        st.info("No particles detected.")
                    elif fov_area <= 0:
                        st.warning("Could not determine FOV area.")
                    else:
                        est = estimate_concentration(
                            ppv=mean_count, fov_area_um2=fov_area,
                            dilution=dilution,
                            plane_thickness_um=axial_depth_um,
                        )

                        # --- Error analysis -------------------------------
                        # Concentration is linear in ppv (all other terms treated
                        # as exact), so the fractional error carries straight
                        # through: σ_M/M = σ_ppv/ppv. We use the field-to-field
                        # SD of the PSF counts and report the standard error of
                        # the mean (SD/√N) as the uncertainty on the mean-derived
                        # concentration.
                        n_fields = len(counts)
                        counts_sd = float(np.std(counts, ddof=1)) if n_fields > 1 else 0.0
                        counts_sem = counts_sd / np.sqrt(n_fields) if n_fields > 1 else 0.0
                        rel_err = counts_sem / mean_count if mean_count else 0.0
                        cv = counts_sd / mean_count if mean_count else 0.0
                        molarity_err = est["molarity"] * rel_err
                        diluted_err = est["conc_diluted_M"] * rel_err

                        st.metric("Stock concentration", _format_molarity(est["molarity"]))
                        if n_fields > 1:
                            st.caption(
                                f"± {_format_molarity(molarity_err)}  "
                                f"(SEM, n={n_fields} fields, CV={cv:.0%})"
                            )
                        else:
                            st.caption("Single field — no error estimate (need ≥2 files).")
                        st.metric("Avg particles / view", f"{est['ppv']:.1f} ppv")

                        with st.expander("Show calculation", expanded=False):
                            obs_l = est["obs_volume_um3"] * 1e-15
                            density_l = est["ppv"] / obs_l if obs_l else 0.0
                            calc = "\n".join([
                                f"ppv (avg particles/view) = {est['ppv']:.4g}",
                                f"FOV area  A_fov          = {fov_area:,.1f} µm²",
                                f"   ({fov_desc}, pixel {pix:g} µm)",
                                f"axial depth  t           = {axial_depth_um * 1000:.0f} nm  (detection range)",
                                f"obs volume V = A_fov·t    = {est['obs_volume_um3']:.4g} µm³",
                                f"                         = {obs_l:.3e} L",
                                "",
                                f"density  = ppv / V        = {density_l:.3e} /L",
                                f"diluted molarity          = {_format_molarity(est['conc_diluted_M'])}"
                                + (f" ± {_format_molarity(diluted_err)}" if n_fields > 1 else ""),
                                f"× dilution {dilution:g}",
                                f"stock molarity            = {_format_molarity(est['molarity'])}"
                                + (f" ± {_format_molarity(molarity_err)}" if n_fields > 1 else ""),
                                "",
                                "-- error analysis (from PSF-count spread) --",
                                f"n fields                 = {n_fields}",
                                f"PSF count SD             = {counts_sd:.3g}",
                                f"SEM = SD/√n              = {counts_sem:.3g}",
                                f"relative error SEM/mean  = {rel_err:.1%}",
                                f"CV = SD/mean             = {cv:.1%}",
                                "σ_M/M = σ_ppv/ppv  (M is linear in ppv)",
                                "",
                                "M_stock = ppv · dilution / (A_fov · t · N_A)",
                            ])
                            st.code(calc, language="text")
