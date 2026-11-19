import os
import re
import io
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import hashlib
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

# Ensure utils is accessible if needed
import sys
if "/mnt/data" not in sys.path:
    sys.path.append("/mnt/data")
import utils

PIX_SIZE_UM = 0.1  # fixed

# --- Caching and Processing Helpers ---

def _hash_file(uf) -> str:
    try:
        b = uf.getbuffer()
    except Exception:
        pos = uf.tell()
        b = uf.read()
        uf.seek(pos)
    return hashlib.md5(b).hexdigest()

def _build_proc_key(sif_files, region_ucnp, region_dye, threshold_ucnp, threshold_dye, ucnp_id, dye_id, min_distance, split_mode, first_acquired, crowded):
    names_hashes = tuple(sorted((f.name, _hash_file(f)) for f in (sif_files or [])))
    return (names_hashes, str(region_ucnp), str(region_dye), float(threshold_ucnp), float(threshold_dye), str(ucnp_id), str(dye_id), int(min_distance), str(split_mode), str(first_acquired), bool(crowded))

def _extract_common_stem(uploaded_files):
    if not uploaded_files:
        return "results"
    name = Path(uploaded_files[0].name).stem
    if "_" in name:
        return name.rsplit("_", 1)[0]
    return name 

def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

# --- Processing Functions (Unchanged) ---

import warnings
try:
    from scipy.optimize import OptimizeWarning  # type: ignore
except Exception:
    class OptimizeWarning(Warning):
        pass
warnings.filterwarnings("ignore", message="Adding colorbar to a different Figure")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", category=OptimizeWarning)

try:
    from tools.process_files import process_files as _process_files_external  # type: ignore
except Exception:
    _process_files_external = None

def _process_files_fallback(uploaded_files, region, threshold=1, signal="UCNP", pix_size_um=PIX_SIZE_UM, sig_threshold=0.3, min_distance=5, crowded=False):
    processed_data: Dict[str, Dict[str, object]] = {}
    all_dfs = []
    temp_dir = Path(tempfile.gettempdir()) / "coloc_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for uf in uploaded_files:
        file_path = temp_dir / uf.name
        with open(file_path, "wb") as f:
            f.write(uf.getbuffer())
        try:
            df, image_data_cps, *_ = utils.integrate_sif(
                str(file_path),
                region=region,
                threshold=threshold,
                signal=signal,
                pix_size_um=pix_size_um,
                sig_threshold=sig_threshold,
                min_distance=min_distance,
                crowded=crowded,
            )
            processed_data[uf.name] = {"df": df, "image": image_data_cps}
            all_dfs.append(df)
        except Exception as e:
            st.error(f"Error processing {uf.name}: {e}")
    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return processed_data, combined_df

def _process_files(uploaded_files, region, threshold, signal, min_distance=5, crowded=False):
    if _process_files_external is not None:
        return _process_files_external(uploaded_files, region, threshold=threshold, signal=signal,
                                       pix_size_um=PIX_SIZE_UM, min_distance=min_distance,
                                       crowded=crowded)
    return _process_files_fallback(uploaded_files, region, threshold=threshold, signal=signal, pix_size_um=PIX_SIZE_UM, min_distance=min_distance, crowded=crowded)

# --- Single-emitter brightness helpers ---

def _isolated_psfs(df, min_radius_px):
    """Keep only PSFs whose nearest neighbor (same image) is >= min_radius_px away.

    Used for the monomer assumption: PSFs sitting closer than ``min_radius_px``
    to another PSF are likely aggregates/overlapping and are dropped so the
    brightness histogram reflects isolated single UCNPs.
    """
    if min_radius_px <= 0 or not isinstance(df, pd.DataFrame) or len(df) < 2:
        return df
    if not {"x_pix", "y_pix"}.issubset(df.columns):
        return df
    xs = df["x_pix"].to_numpy(dtype=float)
    ys = df["y_pix"].to_numpy(dtype=float)
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        dist = np.hypot(xs - xs[i], ys - ys[i])
        dist[i] = np.inf
        if np.nanmin(dist) < min_radius_px:
            keep[i] = False
    return df[keep]


def _single_ucnp_brightness_ui(u_data, assume_monomers, default_manual=1e5):
    """Return the single-UCNP brightness (pps) for the num_ucnps calculation.

    If ``assume_monomers`` is True, pool every UCNP detection across all
    processed UCNP images/regions and use the primary GMM component mean (mu) of
    the brightness histogram as the single-UCNP brightness. Otherwise fall back
    to a manual number input.
    """
    if not assume_monomers:
        return st.number_input(
            "Single UCNP brightness (pps)", min_value=0.0, value=default_manual,
            format="%.2e",
        )

    # Pool all UCNP detections across every processed UCNP image / region.
    dfs = [b.get("df") for b in (u_data or {}).values()]
    dfs = [d for d in dfs if isinstance(d, pd.DataFrame) and not d.empty
           and "brightness_integrated" in d.columns]
    if not dfs:
        st.warning("No UCNP detections available to fit. Enter the brightness manually.")
        return st.number_input(
            "Single UCNP brightness (pps)", min_value=0.0, value=default_manual,
            format="%.2e", key="coloc_single_ucnp_manual_fallback",
        )

    st.markdown("**Single-UCNP brightness (monomer assumption)**: Gaussian fit of isolated UCNP detections.")
    ctrl_c1, ctrl_c2 = st.columns(2)
    with ctrl_c1:
        min_radius_px = st.number_input(
            "Minimum PSF separation (px)", min_value=0.0, value=10.0, step=1.0,
            key="coloc_ucnp_min_radius",
            help="Drop UCNPs with another UCNP closer than this (likely aggregates), "
                 "so only isolated monomers contribute to the brightness fit.",
        )
    with ctrl_c2:
        wf_components = st.number_input(
            "GMM components (UCNP)", min_value=1, max_value=4, value=2,
            key="coloc_ucnp_components",
        )

    # Filter to isolated PSFs per image before pooling.
    iso_dfs = [_isolated_psfs(d, min_radius_px) for d in dfs]
    iso_dfs = [d for d in iso_dfs if isinstance(d, pd.DataFrame) and not d.empty]
    n_total = int(sum(len(d) for d in dfs))
    if not iso_dfs:
        st.warning(
            f"No isolated UCNPs remain after the {min_radius_px:.0f} px separation "
            "filter. Lower the minimum separation, or enter the brightness manually."
        )
        return st.number_input(
            "Single UCNP brightness (pps)", min_value=0.0, value=default_manual,
            format="%.2e", key="coloc_single_ucnp_manual_fallback",
        )

    combined_ucnp = pd.concat(iso_dfs, ignore_index=True)
    st.caption(
        f"Isolated UCNPs used: {len(combined_ucnp)} / {n_total} "
        f"(≥ {min_radius_px:.0f} px from any neighbor)."
    )

    hist_out = utils.plot_histogram(combined_ucnp, n_components=int(wf_components))
    # plot_histogram returns (fig, mu, sigma) with data, or just fig if empty.
    if isinstance(hist_out, tuple):
        fig_u, mu_u, sigma_u = hist_out
    else:
        fig_u, mu_u, sigma_u = hist_out, None, None

    hcol, mcol = st.columns([2, 1])
    with hcol:
        st.pyplot(fig_u, use_container_width=True)
        plt.close(fig_u)
    with mcol:
        st.metric("UCNP PSFs fit", f"{len(combined_ucnp)}")
        if mu_u is not None:
            st.success(f"Single-UCNP brightness ≈ {mu_u:.3g} ± {sigma_u:.2g} pps")

    if mu_u is not None and mu_u > 0:
        st.caption(f"Using single-UCNP brightness = {float(mu_u):.3g} pps (monomer assumption).")
        return float(mu_u)

    st.warning("Could not fit a single-UCNP brightness. Enter it manually.")
    return st.number_input(
        "Single UCNP brightness (pps)", min_value=0.0, value=default_manual,
        format="%.2e", key="coloc_single_ucnp_manual_fallback",
    )


def _single_dye_brightness_ui(dye_sif_files, default_manual=5e2):
    """Return the single-dye brightness (pps) for the num_dyes calculation.

    If ``dye_sif_files`` are provided, run widefield (WF) brightness analysis on
    them (signal='dye') and use the primary GMM component mean (mu) of the
    brightness histogram. Region / threshold / min-distance are user-tunable so
    the fit can be dialed in. If no files are uploaded, fall back to a manual
    number input (same behavior as before).
    """
    if not dye_sif_files:
        return st.number_input(
            "Single Dye brightness (pps)", min_value=0.0, value=default_manual,
            format="%.2e",
        )

    st.markdown("**Single-dye brightness (WF analysis)**: tune region/threshold for good fits.")
    wf1, wf2, wf3, wf4 = st.columns(4)
    with wf1:
        wf_region = st.selectbox("Region", options=["1", "2", "3", "4", "all"],
                                 index=4, key="coloc_wf_region")
    with wf2:
        wf_threshold = st.number_input("Threshold", min_value=0.0, value=10.0,
                                       step=0.5, key="coloc_wf_threshold")
    with wf3:
        wf_min_distance = st.number_input("Min distance (px)", min_value=1,
                                          value=5, key="coloc_wf_min_distance")
    with wf4:
        wf_components = st.number_input("GMM components", min_value=1, max_value=4,
                                        value=2, key="coloc_wf_components")

    _, dye_combined = _process_files(
        dye_sif_files, region=wf_region, threshold=wf_threshold,
        signal="dye", min_distance=wf_min_distance,
    )

    if (dye_combined is None or dye_combined.empty
            or "brightness_integrated" not in dye_combined.columns):
        st.warning(
            "No dye PSFs detected in the uploaded images. Adjust region / "
            "threshold / min distance, or enter the brightness manually below."
        )
        return st.number_input(
            "Single Dye brightness (pps)", min_value=0.0, value=default_manual,
            format="%.2e", key="coloc_single_dye_manual_fallback",
        )

    hist_out = utils.plot_histogram(dye_combined, n_components=int(wf_components))
    # plot_histogram returns (fig, mu, sigma) with data, or just fig if empty.
    if isinstance(hist_out, tuple):
        fig_dye, mu_dye, sigma_dye = hist_out
    else:
        fig_dye, mu_dye, sigma_dye = hist_out, None, None

    hcol, mcol = st.columns([2, 1])
    with hcol:
        st.pyplot(fig_dye, use_container_width=True)
        plt.close(fig_dye)
    with mcol:
        st.metric("Dye PSFs fit", f"{len(dye_combined)}")
        if mu_dye is not None:
            st.success(f"Single-dye brightness ≈ {mu_dye:.3g} ± {sigma_dye:.2g} pps")

    if mu_dye is not None and mu_dye > 0:
        st.caption(f"Using single-dye brightness = {float(mu_dye):.3g} pps (from WF analysis).")
        return float(mu_dye)

    st.warning("Could not fit a single-dye brightness. Enter it manually.")
    return st.number_input(
        "Single Dye brightness (pps)", min_value=0.0, value=default_manual,
        format="%.2e", key="coloc_single_dye_manual_fallback",
    )


# --- Plotting Helpers ---

# Per-region colormaps: each fades from black up to a region-specific color.
_REGION_COLORS = {
    "1": (0.0, 1.0, 1.0),                    # cyan
    "2": (0.0, 1.0, 0.0),                    # green
    "3": (1.0, 0.2, 0.28),                   # neon red (bright, high contrast on black)
    "4": (204 / 255, 121 / 255, 167 / 255),  # reddish purple
}

def _region_cmap(region):
    """Return a black->region-color colormap, or None if the region has no mapping."""
    color = _REGION_COLORS.get(str(region))
    if color is None:
        return None
    return LinearSegmentedColormap.from_list(f"region_{region}", [(0.0, 0.0, 0.0), color])

def _resolve_cmap(cmap_choice, region):
    """Resolve the selected colormap for a given region.

    When 'by region' is selected, use the region-specific black->color map,
    falling back to 'magma' for regions without a mapping (e.g. 'all').
    """
    if cmap_choice == "by region":
        return _region_cmap(region) or "magma"
    return cmap_choice

def HWT_aesthetic():
    """Applies basic aesthetic settings to the current matplotlib axes."""
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

def _despine(ax, keep=("left", "bottom")):
    """Remove top/right spines (or any not in ``keep``) and turn off the grid."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
    ax.grid(False)

def _style_axes(ax, scale=1.25):
    """Make labels/ticks/title Arial bold and ``scale``x larger (default +25%)."""
    base = plt.rcParams.get("font.size", 10.0)
    size = base * scale
    for txt in (ax.title, ax.xaxis.label, ax.yaxis.label):
        txt.set_fontfamily("Arial")
        txt.set_fontweight("bold")
        txt.set_fontsize(size)
    for lbl in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        lbl.set_fontfamily("Arial")
        lbl.set_fontweight("bold")
        lbl.set_fontsize(size)

def _fit_gaussian(values):
    """Fit a Gaussian to 1D data and return (mu, sigma, cv_percent).

    Uses scipy's MLE fit when available, otherwise falls back to the sample
    mean/std. ``cv_percent`` is 100 * sigma / mu (the coefficient of variation).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None, None, None
    try:
        from scipy.stats import norm
        mu, sigma = norm.fit(values)
    except Exception:
        mu, sigma = float(np.mean(values)), float(np.std(values))
    cv = 100.0 * sigma / mu if mu else float("nan")
    return float(mu), float(sigma), float(cv)

def _robust_histogram_fit(values, fence=3.0, min_bins=5, max_bins=40):
    """Histogram + Gaussian fit that a few outliers can't wreck.

    * Range: Tukey fences at ``fence`` x IQR beyond the quartiles (3 = only far outliers,
      e.g. aggregates). Points outside are counted, not plotted or fitted.
    * Bins: Freedman-Diaconis width (2 IQR n^-1/3) over that range, clamped to
      [min_bins, max_bins]; Sturges when the IQR is zero.
    * Fit: with >= 50 points over >= 6 occupied bins, a Gaussian least-squares fitted to
      the bin counts, each weighted by its Poisson error, starting from the median and
      MAD. Smaller samples spread too thinly over the bins for a 3-parameter curve fit,
      so they (and any fit that fails or leaves the data range) get the maximum-likelihood
      Gaussian of the plotted values instead: their mean and SD. Either way the outliers
      beyond the fences are already gone, so neither is dragged by them.

    Returns a dict: values (in range), edges, counts, mu, sigma, mu_err, n, n_out,
    method, redchi2 (NaN when not fitted).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    out = {"values": v, "edges": None, "counts": None, "mu": None, "sigma": None,
           "mu_err": float("nan"), "n": int(v.size), "n_out": 0, "method": "",
           "redchi2": float("nan")}
    if v.size < 2:
        return out
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    if iqr > 0:
        keep = (v >= q1 - fence * iqr) & (v <= q3 + fence * iqr)
        out["n_out"] = int((~keep).sum())
        v = v[keep]
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    if iqr > 0:
        width = 2.0 * iqr / v.size ** (1.0 / 3.0)
        nb = int(np.clip(np.ceil((hi - lo) / width), min_bins, max_bins))
    else:
        nb = int(np.clip(np.ceil(np.log2(v.size) + 1), min_bins, max_bins))
    counts, edges = np.histogram(v, bins=nb, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw = float(edges[1] - edges[0])
    mad_sigma = 1.4826 * float(np.median(np.abs(v - np.median(v))))
    sigma0 = mad_sigma if mad_sigma > 0 else max(float(np.std(v)), bw)
    sd_ml = float(np.std(v, ddof=1)) if v.size > 1 else float("nan")
    out.update(values=v, edges=edges, counts=counts, n=int(v.size),
               mu=float(np.mean(v)), sigma=sd_ml,
               mu_err=sd_ml / np.sqrt(v.size), method="Gaussian MLE (mean, SD)")

    if v.size >= 50 and (counts > 0).sum() >= 6:
        try:
            from scipy.optimize import curve_fit
            g = lambda x, a, m, sd: a * np.exp(-0.5 * ((x - m) / sd) ** 2)
            p0 = [counts.max(), float(np.median(v)), sigma0]
            popt, pcov = curve_fit(g, centers, counts, p0=p0,
                                   sigma=np.sqrt(np.maximum(counts, 1.0)),
                                   bounds=([0, lo, bw / 4], [np.inf, hi, (hi - lo) * 2]),
                                   maxfev=10000)
            a, m, sd = popt
            if lo <= m <= hi:
                model = g(centers, *popt)
                dof = max(nb - 3, 1)
                out.update(mu=float(m), sigma=float(abs(sd)),
                           mu_err=float(np.sqrt(pcov[1, 1])) if np.isfinite(pcov[1, 1]) else float("nan"),
                           method="Gaussian fit to histogram",
                           redchi2=float(np.sum((counts - model) ** 2 / np.maximum(model, 1.0)) / dof))
        except (RuntimeError, ValueError):
            pass
    return out


def _dyes_per_ucnp_figure(y_subset, n_dye_psfs, n_ucnp_psfs):
    """Histogram of dyes per single UCNP with its Gaussian (see _robust_histogram_fit)
    and a small annotation box: dye:UCNP PSF ratio, outliers trimmed, mu and sigma.
    Returns (figure, fit dict)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    hf = _robust_histogram_fit(y_subset)
    if hf["edges"] is not None:
        ax.hist(hf["values"], bins=hf["edges"], color="0.75", edgecolor="black", linewidth=0.8)
        if hf["mu"] is not None and hf["sigma"]:
            # Gaussian density scaled to counts per bin.
            bw = float(hf["edges"][1] - hf["edges"][0])
            xs = np.linspace(hf["edges"][0], hf["edges"][-1], 300)
            pdf = np.exp(-0.5 * ((xs - hf["mu"]) / hf["sigma"]) ** 2)
            pdf *= hf["n"] * bw / (hf["sigma"] * np.sqrt(2 * np.pi))
            ax.plot(xs, pdf, color="crimson", lw=2)
        ax.set_title("Dyes per single UCNP")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)   # headroom so the box clears the bars
    else:
        ax.set_title("Dyes per single UCNP: no data")
        ax.set_ylim(0, 5)

    ratio = n_dye_psfs / n_ucnp_psfs if n_ucnp_psfs else float("nan")
    lines = []
    if hf["n_out"]:
        lines.append(f"Outliers not shown: {hf['n_out']}")
    if hf["mu"] is not None and hf["n"] >= 2:
        cv = 100.0 * hf["sigma"] / hf["mu"] if hf["mu"] else float("nan")
        lines.append(f"μ = {hf['mu']:.2f} ± {hf['mu_err']:.2f}" if np.isfinite(hf["mu_err"])
                     else f"μ = {hf['mu']:.2f}")
        lines.append(f"σ = {hf['sigma']:.2f}  (σ/μ = {cv:.0f}%)")
    # Last: the ratio over every fitted PSF in both channels, not just the plotted UCNPs.
    lines.append(f"All dyes / UCNPs: {ratio:.2f}")
    ax.set_xlabel("Number of Dyes")
    ax.set_ylabel("Count")
    _despine(ax)
    _style_axes(ax)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.text(0.97, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=plt.rcParams.get("font.size", 10.0) * 1.1, family="Arial",
            linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.6", alpha=0.92))
    return fig, hf


def _autoscale_xy(ax, x, y, pad=0.05):
    import numpy as _np
    x = _np.asarray(x); y = _np.asarray(y)
    m = _np.isfinite(x) & _np.isfinite(y)
    ax.set_xlim(0, 5)
    if not m.any():
        return
    x_in = (x >= 0) & (x <= 5) & m
    if not x_in.any(): x_in = m
    yv = y[x_in]
    if yv.size == 0: return
    ymin, ymax = float(_np.nanmin(yv)), float(_np.nanmax(yv))
    dy = ymax - ymin
    if not _np.isfinite(dy) or dy <= 0:
        dy = max(1.0, abs(ymax) if _np.isfinite(ymax) else 1.0)
    ax.set_ylim(ymin - dy * pad, ymax + dy * pad)

# Shared colorbar geometry so images and the reconstruction shrink equally.
_CBAR_FRACTION = 0.046
_CBAR_PAD = 0.04

# Extra right margin so the vertical colorbar label isn't clipped when the
# figure renders without a tight bounding box (bbox_inches=None).
_CBAR_RIGHT_MARGIN = 0.82

def _add_colorbar(fig, ax, mappable):
    """Add a colorbar outside the axes with a vertical 'pps' label."""
    fig.subplots_adjust(right=_CBAR_RIGHT_MARGIN)
    cb = fig.colorbar(mappable, ax=ax, fraction=_CBAR_FRACTION, pad=_CBAR_PAD)
    cb.set_label("pps")
    return cb

def _reserve_colorbar_space(fig, ax):
    """Shrink ``ax`` by a colorbar's worth of space without drawing a bar.

    Keeps the reconstruction panel the same size as the image panels when they
    have a colorbar: the same right margin and a hidden colorbar steal the same
    axes space, then the bar itself is hidden.
    """
    from matplotlib.cm import ScalarMappable
    fig.subplots_adjust(right=_CBAR_RIGHT_MARGIN)
    sm = ScalarMappable()
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=_CBAR_FRACTION, pad=_CBAR_PAD)
    cb.ax.set_visible(False)

def _corner_label(ax, text):
    """Top-left channel label: white bold text on a grey (alpha 0.8) background."""
    ax.text(
        0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top",
        color="white", fontsize=11, fontweight="bold", fontfamily="Arial",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="grey",
                  edgecolor="none", alpha=0.8),
    )

def _overlay_circles(ax, df: pd.DataFrame, color: str, alpha: float, label: bool = False):
    from matplotlib.patches import Circle
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    required = {"x_pix","y_pix","sigx_fit","sigy_fit","brightness_fit"}
    if not required.issubset(df.columns):
        return
    # On an image panel, pin the view to the image: otherwise a circle around a PSF near
    # the edge extends the data limits, the axes zoom out, and this panel's image ends
    # up smaller than its neighbours. Circles are clipped at the image edge instead.
    limits = (ax.get_xlim(), ax.get_ylim()) if ax.images else None
    for _, row in df.iterrows():
        try:
            rad_px = 4 * float(max(row["sigx_fit"], row["sigy_fit"])) / PIX_SIZE_UM
        except Exception:
            rad_px = 6.0
        ax.add_patch(Circle((row["x_pix"], row["y_pix"]), radius=rad_px, color=color, fill=False, linewidth=1.2, alpha=alpha))
        if label:
            ax.text(row["x_pix"] + 8, row["y_pix"] + 8, f"{row['brightness_fit']/1000:.1f} kpps",
                    color=color, fontsize=8, ha="center", va="center", clip_on=True)
    if limits is not None:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])

# --- Matching Logic ---

def _split_ucnp_dye(files: List[Any], ucnp_id="976", dye_id="638") -> Tuple[List[Any], List[Any]]:
    u_tok = str(ucnp_id).lower().strip()
    d_tok = str(dye_id).lower().strip()
    ucnp, dye = [], []
    for f in files:
        name = f.name if hasattr(f, "name") else str(f)
        lname = name.lower()
        has_ucnp = u_tok in lname if u_tok else False
        has_dye  = d_tok in lname if d_tok else False
        if has_ucnp and not has_dye:
            ucnp.append(f)
        elif has_dye and not has_ucnp:
            dye.append(f)
        elif has_ucnp and has_dye:
            st.warning(f"Filename matches both tokens, skipping: {name}")
        else:
            st.warning(f"Filename matches neither token, skipping: {name}")
    return ucnp, dye

def _sample_key(name: str, tokens=()) -> Tuple[str, int]:
    """``(sample key, sequence number)`` for an acquisition filename.

    The sequence number is the trailing number (``..._10.sif`` -> 10). The sample key is
    everything before it, split on ``_``/``-``/spaces, minus any segment containing one
    of ``tokens`` (the UCNP/dye ID tokens), lower-cased. So in token mode
    ``HWT07_011A_1to10_976nm1500mA_3`` and ``HWT07_011A_1to10_638nm90mA_4`` share the key
    ``hwt07_011a_1to10``, while ``HWT08_135A_1to100k_..._10`` and
    ``HWT08_135A_1to500_..._9`` stay apart.
    """
    stem = Path(str(name)).stem
    m = re.match(r"^(.*?)(\d+)$", stem)
    prefix, idx = (m.group(1), int(m.group(2))) if m else (stem, -1)
    toks = [str(t).lower().strip() for t in tokens if str(t).strip()]
    parts = [seg for seg in re.split(r"[_\-\s]+", prefix)
             if seg and not any(t in seg.lower() for t in toks)]
    return "_".join(parts).lower(), idx


def _split_ucnp_dye_alternating(files: List[Any], first: str = "UCNP") -> Tuple[List[Any], List[Any]]:
    """Split files assuming UCNP/Dye images were acquired in alternating order.

    Files are grouped by sample (filename minus its trailing number, see _sample_key)
    and, within each sample, ordered by that number and assigned alternately: the
    ``first``-acquired type gets the 1st, 3rd, 5th ... file and the other type the 2nd,
    4th ... Alternating per sample matters: across a whole upload, one sample with an
    odd number of files would swap UCNP and dye for every sample after it.
    """
    def _name(f):
        return f.name if hasattr(f, "name") else str(f)

    groups: Dict[str, List[Tuple[int, Any]]] = {}
    for f in files:
        key, idx = _sample_key(_name(f))
        groups.setdefault(key, []).append((idx, f))
    first_is_ucnp = str(first).lower().startswith("u")
    ucnp, dye = [], []
    for key in sorted(groups, key=natural_sort_key):
        ordered = sorted(groups[key], key=lambda t: (t[0], natural_sort_key(_name(t[1]))))
        for pos, (_, f) in enumerate(ordered):
            if (pos % 2 == 0) == first_is_ucnp:
                ucnp.append(f)
            else:
                dye.append(f)
    return ucnp, dye


def _match_ucnp_dye_files(ucnps: List[Any], dyes: List[Any], tokens=()) -> List[Tuple[Any, Any]]:
    """Pair each UCNP image with the dye image acquired next to it.

    Only files from the same sample (same _sample_key, ignoring the ID ``tokens``) are
    paired; within a sample, files are walked in sequence-number order and adjacent
    UCNP/dye files whose numbers differ by at most 1 become a pair.
    """
    def _name(f):
        return f.name if hasattr(f, "name") else str(f)

    groups: Dict[str, List[dict]] = {}
    for kind, files in (("u", ucnps), ("d", dyes)):
        for f in files:
            key, idx = _sample_key(_name(f), tokens)
            groups.setdefault(key, []).append({"file": f, "type": kind, "idx": idx})

    pairs: List[Tuple[Any, Any]] = []
    for key in sorted(groups, key=natural_sort_key):
        entries = sorted(groups[key], key=lambda x: (x["idx"], x["type"] == "d"))
        i = 0
        while i < len(entries) - 1:
            cur, nxt = entries[i], entries[i + 1]
            if cur["type"] != nxt["type"] and abs(cur["idx"] - nxt["idx"]) <= 1:
                u_file = cur["file"] if cur["type"] == "u" else nxt["file"]
                d_file = nxt["file"] if nxt["type"] == "d" else cur["file"]
                pairs.append((u_file, d_file))
                i += 2
            else:
                i += 1
    return pairs

def _compute_coloc_mask(df_u: pd.DataFrame, df_d: pd.DataFrame, radius_px: int):
    if df_u is None or df_d is None or df_u.empty or df_d.empty:
        return None, None, []
    if not {"x_pix","y_pix"}.issubset(df_u.columns) or not {"x_pix","y_pix"}.issubset(df_d.columns):
        return None, None, []
    u_mask = np.zeros(len(df_u), dtype=bool)
    d_mask = np.zeros(len(df_d), dtype=bool)
    pairs = []
    used_d = set()
    for i_u, row_u in df_u.iterrows():
        du = df_d.loc[~df_d.index.isin(used_d)]
        if du.empty:
            continue
        dx = du["x_pix"].values - row_u["x_pix"]
        dy = du["y_pix"].values - row_u["y_pix"]
        dist = np.hypot(dx, dy)
        j = dist.argmin()
        if dist[j] <= radius_px:
            d_idx = du.index[j]
            u_mask[df_u.index.get_loc(i_u)] = True
            d_mask[df_d.index.get_loc(d_idx)] = True
            pairs.append((i_u, d_idx, float(dist[j])))
            used_d.add(d_idx)
    return u_mask, d_mask, pairs

# --- Channel registration and overlay ---

# Single-hue black->colour maps for the overlay: unlike hot/afmhot they never run on to
# white, so a channel keeps one colour at every intensity and overlap reads as a mix.
_SINGLE_HUE = {"red": "#ff0000", "green": "#00ff00", "blue": "#0080ff",
               "magenta": "#ff00ff", "cyan": "#00ffff", "yellow": "#ffff00"}
for _name, _hex in _SINGLE_HUE.items():
    if _name not in plt.colormaps():
        plt.colormaps.register(LinearSegmentedColormap.from_list(_name, ["#000000", _hex]))

OVERLAY_CMAPS = ["gray", "red", "green", "blue", "magenta", "cyan", "yellow",
                 "hot", "afmhot", "magma", "inferno", "viridis", "cividis"]
# The UCNP channel tops out at this brightness in the merge, so that full UCNP + full
# dye stays a visible mix (gray + red = pink) instead of saturating to white.
OVERLAY_UCNP_MAX = 0.7


def _xy(df):
    if not isinstance(df, pd.DataFrame) or df.empty or not {"x_pix", "y_pix"}.issubset(df.columns):
        return np.zeros((0, 2))
    return df[["x_pix", "y_pix"]].to_numpy(dtype=float)


def _fit_affine(src, dst):
    """Least-squares affine src→dst, or None if degenerate. ``from_estimate`` replaced
    ``estimate`` in scikit-image 0.26; keep working on either side of that change."""
    from skimage.transform import AffineTransform
    if hasattr(AffineTransform, "from_estimate"):
        t = AffineTransform.from_estimate(src, dst)
        return t if t else None
    t = AffineTransform()
    return t if t.estimate(src, dst) else None


MIN_REG_MATCHES = 5   # fewer common PSFs than this and any "registration" is noise


def _estimate_registration(pairs, u_data, d_data, max_shift=30, tol=1.5):
    """Affine map from dye to UCNP pixel coordinates, from fitted PSF positions.

    Pooled over every image pair, on the assumption that the optical path is the same
    throughout one batch. Steps: (1) every dye-to-UCNP displacement within ``max_shift``
    px votes in a 1-px histogram, whose peak is the coarse shift (true matches pile up at
    one offset, chance neighbours spread out); (2) dye points are paired to the nearest
    UCNP within 2*``tol`` px after that shift; (3) RANSAC fits an affine to those pairs,
    rejecting mismatches beyond ``tol`` px; (4) all pairs within ``tol`` of the RANSAC
    model are re-matched and the affine is refitted by least squares. Falls back to the
    pure shift when there are too few matches for a stable affine, and returns None when
    fewer than MIN_REG_MATCHES PSFs are common to both channels (with little true
    colocalization the shift vote just picks a chance alignment).
    """
    from scipy.ndimage import gaussian_filter as _gf
    from scipy.spatial import cKDTree
    from skimage.measure import ransac
    from skimage.transform import AffineTransform

    sets = [(_xy(u_data.get(u, {}).get("df")), _xy(d_data.get(d, {}).get("df"))) for u, d in pairs]
    sets = [(u, d) for u, d in sets if len(u) >= 3 and len(d) >= 3]
    if not sets:
        return None

    votes = []
    for u, d in sets:
        tree = cKDTree(u)
        for pt, idx in zip(d, tree.query_ball_point(d, max_shift)):
            if idx:
                votes.append(u[idx] - pt)
    if not votes:
        return None
    votes = np.vstack(votes)
    hist, xe, ye = np.histogram2d(votes[:, 0], votes[:, 1], bins=2 * max_shift,
                                  range=[[-max_shift, max_shift], [-max_shift, max_shift]])
    ix, iy = np.unravel_index(np.argmax(_gf(hist, 1.0)), hist.shape)
    coarse = np.array([(xe[ix] + xe[ix + 1]) / 2, (ye[iy] + ye[iy + 1]) / 2])
    near = np.hypot(*(votes - coarse).T) <= tol
    shift = votes[near].mean(axis=0) if near.any() else coarse

    def match(transform, radius):
        src, dst = [], []
        for u, d in sets:
            dist, j = cKDTree(u).query(transform(d), distance_upper_bound=radius)
            ok = np.isfinite(dist)
            src.append(d[ok]); dst.append(u[j[ok]])
        return np.vstack(src), np.vstack(dst)

    translate = AffineTransform(translation=shift)
    src, dst = match(translate, 2 * tol)
    if len(src) < MIN_REG_MATCHES:
        return None
    result = {"tform": translate, "kind": "shift only"}
    if len(src) >= 6:
        model, inliers = ransac((src, dst), AffineTransform, min_samples=3,
                                residual_threshold=tol, max_trials=2000, rng=0)
        if model is not None and inliers is not None and inliers.sum() >= 6:
            src, dst = match(model, tol)
            refined = _fit_affine(src, dst) if len(src) >= 6 else None
            if refined is not None:
                result = {"tform": refined, "kind": "affine"}
    src, dst = match(result["tform"], tol)
    if len(src) < MIN_REG_MATCHES:
        return None
    resid = np.hypot(*(result["tform"](src) - dst).T)
    result.update(n_matches=int(len(src)), rms=float(np.sqrt(np.mean(resid ** 2))))
    return result


def _describe_registration(reg) -> str:
    t = reg["tform"]
    tx, ty = t.translation
    sx, sy = t.scale if np.ndim(t.scale) else (t.scale, t.scale)
    return (f"Dye → UCNP registration ({reg['kind']}, {reg['n_matches']} matched PSFs, "
            f"RMS {reg['rms']:.2f} px): shift ({tx:+.2f}, {ty:+.2f}) px, "
            f"scale {sx:.4f} × {sy:.4f}, rotation {np.degrees(t.rotation):+.2f}°, "
            f"shear {np.degrees(getattr(t, 'shear', 0.0)):+.2f}°.")


def _registered(df, reg):
    """``df`` with its PSF positions mapped into the UCNP frame."""
    if reg is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    xy = reg["tform"](_xy(df))
    out["x_pix"], out["y_pix"] = xy[:, 0], xy[:, 1]
    return out


def _channel_rgb(img, cmap, vmax, log):
    """One channel mapped through its colormap to RGB for the overlay. The floor is the
    image median (≈ the background, PSFs cover a small fraction of pixels) rather than
    the minimum: with an illuminated disc on a dark surround the minimum sits far below
    the background, which would paint the whole disc in the channel's colour and hide
    the other channel. The ceiling is the panel's display max."""
    a = np.asarray(img, dtype=float) + 1.0
    lo = float(np.nanmedian(a))
    hi = (vmax + 1.0) if vmax else float(np.nanmax(a))
    if hi <= lo:
        hi = float(np.nanmax(a))
    if log:
        lo = max(lo, 1.0)
        hi = max(hi, lo * 1.0001)
        vals = (np.log(np.clip(a, lo, hi)) - np.log(lo)) / (np.log(hi) - np.log(lo))
    else:
        vals = np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1)
    return plt.get_cmap(cmap)(vals)[..., :3]


def _overlay_rgb(u_img, d_img, reg, u_cmap, d_cmap, u_vmax, d_vmax, log):
    """Dye image warped into the UCNP frame, each channel in its own colormap, combined
    with a screen blend (1 - (1-a)(1-b)): black stays black and each channel shows in
    its own colour. The UCNP channel is capped at OVERLAY_UCNP_MAX so overlap mixes
    rather than whiting out: with the defaults, UCNP alone is light gray, dye alone
    red, and both together pink."""
    from skimage.transform import warp
    warped = warp(np.asarray(d_img, dtype=float), reg["tform"].inverse,
                  output_shape=np.shape(u_img), order=1, preserve_range=True,
                  cval=float(np.nanmin(d_img)))
    u_rgb = _channel_rgb(u_img, u_cmap, u_vmax, log) * OVERLAY_UCNP_MAX
    d_rgb = _channel_rgb(warped, d_cmap, d_vmax, log)
    return 1.0 - (1.0 - u_rgb) * (1.0 - d_rgb)


def _add_scale_bar(ax, img_shape, pix_size_um=PIX_SIZE_UM, color="white"):
    """A 'nice' (1/2/5 x 10^k um) scale bar about a fifth of the image width, lower right."""
    h, w = img_shape[:2]
    target = 0.2 * w * pix_size_um
    k = 10 ** np.floor(np.log10(target))
    length_um = max(m * k for m in (1, 2, 5) if m * k <= target) if target >= k else k
    length_px = length_um / pix_size_um
    x1, y = w * 0.95, h * 0.06
    ax.plot([x1 - length_px, x1], [y, y], color=color, lw=3, solid_capstyle="butt")
    ax.text(x1 - length_px / 2, y + h * 0.03, f"{length_um:g} µm", color=color,
            ha="center", va="bottom", fontsize=10, fontweight="bold")


def _representative_fov_figure(u_name, d_name, u_bundle, d_bundle, reg, u_cmap, d_cmap,
                               panel_cmap_u, panel_cmap_d, u_vmax, d_vmax, log,
                               u_mask_df=None):
    """UCNP | dye (registered into the UCNP frame) | merged, as one figure-ready panel."""
    from skimage.transform import warp
    u_img, d_img = u_bundle["image"], d_bundle["image"]
    d_reg = warp(np.asarray(d_img, dtype=float), reg["tform"].inverse,
                 output_shape=np.shape(u_img), order=1, preserve_range=True,
                 cval=float(np.nanmin(d_img)))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    norm_u = LogNorm() if log else None
    norm_d = LogNorm() if log else None
    axes[0].imshow(u_img + 1, cmap=panel_cmap_u, norm=norm_u, origin="lower",
                   vmax=(u_vmax + 1) if (u_vmax and not log) else None)
    # Dye drawn as in the merged panel (its colormap, median background floor).
    axes[1].imshow(d_reg + 1, cmap=d_cmap, norm=norm_d, origin="lower",
                   vmin=(float(np.nanmedian(d_img)) + 1) if not log else None,
                   vmax=(d_vmax + 1) if (d_vmax and not log) else None)
    axes[2].imshow(_overlay_rgb(u_img, d_img, reg, u_cmap, d_cmap, u_vmax, d_vmax, log),
                   origin="lower")
    for ax, label in zip(axes, ("UCNP", "Dye", "Merged")):
        ax.set_xticks([]); ax.set_yticks([])
        _corner_label(ax, label)
        _add_scale_bar(ax, np.shape(u_img))
    if u_mask_df is not None and not u_mask_df.empty:
        _overlay_circles(axes[2], u_mask_df, color="lime", alpha=0.9, label=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.03)
    return fig


def _seeded_vmax_input(label, key, auto):
    """Monomers-style display max: re-seeded whenever the auto value changes (new data
    or percentage); a typed value sticks until then."""
    if st.session_state.get(f"_{key}_auto") != auto or key not in st.session_state:
        st.session_state[f"_{key}_auto"] = auto
        st.session_state[key] = auto
    return st.number_input(label, min_value=0.0, format="%.4g", key=key,
                           help=f"Auto: {auto:.4g} pps. Edit to override; resets when the "
                                "data or the percentage change.")


# --- Main App ---

def run():
    # Note: st.set_page_config is owned by app.py (called once, first); not called here.
    with st.sidebar:
        st.header("Inputs")
        sif_files = utils.file_uploader_with_clear("SIF files (UCNP + Dye)", key="coloc_uploads", type=["sif"], accept_multiple_files=True)
        stem = _extract_common_stem(sif_files)

        st.header("IDs")
        split_mode = st.radio(
            "File identification",
            options=["By filename token", "Alternating acquisition"],
            help="Choose how UCNP vs Dye files are identified. 'By filename token' "
                 "matches substrings in the filename. 'Alternating acquisition' "
                 "assumes UCNP and Dye images were taken one after another and "
                 "sorts by file numbering.",
        )
        if split_mode == "By filename token":
            ucnp_id = st.text_input("UCNP ID token", value="976", help="Substring used to identify UCNP files (matched in filename).")
            dye_id  = st.text_input("Dye ID token",  value="638", help="Substring used to identify Dye files (matched in filename).")
            first_acquired = "UCNP"
        else:
            first_acquired = st.selectbox(
                "First acquired", options=["UCNP", "Dye"], index=0,
                help="Which type was imaged first. Files are sorted by their "
                     "filename numbering and assigned alternately.",
            )
            ucnp_id = dye_id = ""

        st.divider()
        st.header("Fitting")
        th_u, th_d = st.columns(2)
        threshold_ucnp = th_u.number_input(
            "Threshold (UCNP)", min_value=0.0, value=2.0, step=0.05,
            help="UCNP peaks must exceed the image mean by this many standard deviations "
                 "(after light smoothing). Lower finds dimmer particles.")
        threshold_dye = th_d.number_input(
            "Threshold (Dye)", min_value=0.0, value=2.0, step=0.05,
            help="Sensitivity of the dye blob detector (its threshold is 5x this value). "
                 "Lower finds dimmer dye spots.")
        crowded = st.checkbox(
            "Crowded-field fitting", value=True,
            help="Fits each PSF together with its overlapping neighbours instead of as a "
                 "lone Gaussian, then subtracts all fits and looks again for PSFs hidden "
                 "under brighter neighbours. Recovers many more particles in dense regions "
                 "and gives cleaner brightnesses; about 3x slower per image.",
        )
        min_distance = st.number_input(
            "Minimum distance (px)", min_value=1, value=5,
            help="Minimum separation between detected PSFs. With crowded-field fitting, "
                 "3 resolves close pairs; without it, increase to avoid fitting PSFs that "
                 "are too close together.",
        )
        region_ucnp = st.selectbox("Region (UCNP)", options=["1","2","3","4","all"], index=0)
        region_dye  = st.selectbox("Region (Dye)",  options=["1","2","3","4","all"], index=0)
        radius_px = st.number_input("Colocalization radius (pixels)", min_value=1, value=2)

        st.header("Overlays")
        show_all_fits = st.checkbox("Show all fits", value=False)
        show_coloc_fits = st.checkbox("Show colocalized fits", value=True)
        show_recon = st.checkbox(
            "Show reconstruction", value=False,
            help="Adds a panel plotting the fitted UCNP and dye positions, with "
                 "colocalized dye PSFs circled.")

        st.header("Display")
        cmap = st.selectbox("Colormap", options=["by region","magma","viridis","plasma","hot","gray","hsv"], index=5)
        use_lognorm = st.checkbox("Log image scaling", value=False)
        show_colorbars = st.checkbox("Show colorbars on images", value=True)
        scale_box = st.container()   # display max, filled once the data are processed

        st.header("Channel overlay")
        show_overlay = st.checkbox(
            "Show registered overlay", value=True,
            help="Warps each dye image onto its UCNP image with an affine transform "
                 "estimated from the fitted PSF positions, and shows both in one image.",
        )
        u_ov_cmap, d_ov_cmap = "gray", "red"
        if show_overlay:
            u_ov_cmap = st.selectbox("UCNP colormap", OVERLAY_CMAPS, index=OVERLAY_CMAPS.index("gray"))
            d_ov_cmap = st.selectbox("Dye colormap", OVERLAY_CMAPS, index=OVERLAY_CMAPS.index("red"))
        use_reg_match = st.checkbox(
            "Use registration for colocalization", value=False,
            help="Map dye positions into the UCNP frame before matching, correcting any "
                 "shift, scale or rotation between the channels. Off = match raw pixel "
                 "positions (previous behaviour).",
        )

    if not sif_files:
        st.info("Upload SIF files to begin.")
        return

    # --- Processing ---
    # Splitting and pairing are cheap and are redone every run; only the PSF fits are
    # cached. The cache key includes the actual UCNP/dye file assignment, so a change in
    # how files are split (settings or code) can never be masked by stale fits.
    if split_mode == "Alternating acquisition":
        ucnp_files, dye_files = _split_ucnp_dye_alternating(sif_files, first=first_acquired)
    else:
        ucnp_files, dye_files = _split_ucnp_dye(sif_files, ucnp_id=ucnp_id, dye_id=dye_id)
    proc_key = _build_proc_key(sif_files, region_ucnp, region_dye, threshold_ucnp, threshold_dye, ucnp_id, dye_id, min_distance, split_mode, first_acquired, crowded) + (
        tuple(f.name for f in ucnp_files), tuple(f.name for f in dye_files))
    if "proc_key" not in st.session_state or st.session_state["proc_key"] != proc_key:
        with st.spinner("Fitting PSFs..."):
            u_data, _ = _process_files(ucnp_files, region=region_ucnp, threshold=threshold_ucnp, signal="UCNP", min_distance=min_distance, crowded=crowded) if ucnp_files else ({}, pd.DataFrame())
            d_data, _ = _process_files(dye_files,  region=region_dye,  threshold=threshold_dye, signal="dye",  min_distance=min_distance, crowded=crowded)  if dye_files  else ({}, pd.DataFrame())
        st.session_state.update({"proc_key": proc_key, "u_data": u_data, "d_data": d_data})
    else:
        u_data = st.session_state["u_data"]
        d_data = st.session_state["d_data"]

    u_names = sorted(list(u_data.keys()), key=natural_sort_key)
    d_names = sorted(list(d_data.keys()), key=natural_sort_key)
    pairs = _match_ucnp_dye_files(u_names, d_names, tokens=(ucnp_id, dye_id))
    reg_key = (proc_key, tuple(pairs))
    if st.session_state.get("coloc_reg_key") != reg_key:
        st.session_state["coloc_reg"] = _estimate_registration(pairs, u_data, d_data)
        st.session_state["coloc_reg_key"] = reg_key
    reg = st.session_state.get("coloc_reg")

    st.caption(f"Matched {len(pairs)} UCNP↔Dye pairs.")
    paired = {n for pr in pairs for n in pr}
    unpaired = sorted((set(u_data) | set(d_data)) - paired, key=natural_sort_key)
    if unpaired:
        st.warning(f"{len(unpaired)} file(s) had no partner from the same sample with an "
                   "adjacent number and were left out: " + ", ".join(unpaired))
    if show_overlay or use_reg_match:
        if reg is None:
            st.warning(f"Couldn't register the channels: fewer than {MIN_REG_MATCHES} PSFs "
                       "appear in both. Colocalization uses raw pixel positions and the "
                       "overlay is shown unregistered.")
        else:
            st.caption(_describe_registration(reg))
    if show_overlay:
        st.caption(f"Overlay: UCNP in **{u_ov_cmap}**, dye in **{d_ov_cmap}**; where both are "
                   "bright the colours mix"
                   + (" (gray + red = pink)" if (u_ov_cmap, d_ov_cmap) == ("gray", "red") else "")
                   + (", green circles mark colocalized UCNPs." if show_coloc_fits else "."))
    match_reg = reg if use_reg_match else None
    # Identity when registration failed, so the overlay still shows the raw alignment.
    from skimage.transform import AffineTransform
    overlay_reg = reg if reg is not None else {"tform": AffineTransform()}

    # Linear display max per channel (as in Monomer Estimation): the brightness that
    # keeps the chosen share of fitted particles in range, so a few bright aggregates
    # don't wash everything else out.
    u_vmax = d_vmax = None
    if not use_lognorm:
        with scale_box:
            vis_pct = st.number_input(
                "Particles in display range (%)", min_value=50.0, max_value=100.0,
                value=90.0, step=5.0, format="%.0f", key="coloc_vis_pct",
                help="Sets each channel's linear display max to the brightness of this "
                     "percentile of its fitted particles (peak pixel, pooled over all "
                     "files). The brightest remainder saturate; 100% = no clipping.",
            )
            u_auto = utils.auto_display_vmax(u_data, vis_pct)
            d_auto = utils.auto_display_vmax(d_data, vis_pct)
            if u_auto is not None:
                u_vmax = _seeded_vmax_input("UCNP display max (pps)", "coloc_u_vmax", u_auto)
            if d_auto is not None:
                d_vmax = _seeded_vmax_input("Dye display max (pps)", "coloc_d_vmax", d_auto)

    # --- Pre-calculate Statistics to avoid layout jumping ---
    overall_ucnp_hits = overall_ucnp_total = 0
    overall_dye_hits  = overall_dye_total  = 0
    
    ucnp_pct_list = []
    dye_pct_list = []
    pair_stats = []   # (u_name, d_name, ucnp coloc %, dye coloc %) for the representative FOV
    
    # We iterate once purely for stats (computation is fast, plotting is slow)
    for u_name, d_name in pairs:
        u_df = u_data.get(u_name, {}).get("df", pd.DataFrame())
        d_df = d_data.get(d_name, {}).get("df", pd.DataFrame())
        
        u_mask, d_mask, _ = _compute_coloc_mask(u_df, _registered(d_df, match_reg), radius_px=radius_px)

        u_h = int(u_mask.sum()) if (u_mask is not None) else 0
        u_t = len(u_df) if isinstance(u_df, pd.DataFrame) else 0
        
        d_h = int(d_mask.sum()) if (d_mask is not None) else 0
        d_t = len(d_df) if isinstance(d_df, pd.DataFrame) else 0
        
        overall_ucnp_hits += u_h
        overall_ucnp_total += u_t
        overall_dye_hits += d_h
        overall_dye_total += d_t

        if u_t > 0: ucnp_pct_list.append(100.0 * u_h / u_t)
        if d_t > 0: dye_pct_list.append(100.0 * d_h / d_t)
        if u_t > 0 and d_t > 0:
            pair_stats.append((u_name, d_name, 100.0 * u_h / u_t, 100.0 * d_h / d_t))

    # Calculate overall aggregate %
    ov_u_pct = 100.0 * overall_ucnp_hits / max(overall_ucnp_total, 1)
    ov_d_pct = 100.0 * overall_dye_hits / max(overall_dye_total, 1)
    
    # Calculate SD across images
    sd_u = np.std(ucnp_pct_list) if ucnp_pct_list else 0.0
    sd_d = np.std(dye_pct_list) if dye_pct_list else 0.0

    # Display Overall Stats at the TOP
    st.markdown(
        f"### Overall Colocalized\n"
        f"**UCNP:** {overall_ucnp_hits}/{overall_ucnp_total} "
        f"({ov_u_pct:.1f}% ± {sd_u:.1f}%) "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Dye:** {overall_dye_hits}/{overall_dye_total} "
        f"({ov_d_pct:.1f}% ± {sd_d:.1f}%)"
    )
    
    st.divider()

    tab_pairs = st.container()
    tab_plots = st.container()

    matched_rows = []

    with tab_pairs:
        # --- Iterate Pairs for Display ---
        for u_name, d_name in pairs:
            u_bundle = u_data.get(u_name, {})
            d_bundle = d_data.get(d_name, {})
            u_df = u_bundle.get("df", pd.DataFrame())
            d_df = d_bundle.get("df", pd.DataFrame())
            u_img = u_bundle.get("image", None)
            d_img = d_bundle.get("image", None)

            # Re-compute mask for this specific pair to get display data. Matching runs in
            # the UCNP frame when registration is on; exported dye positions stay raw.
            d_df_m = _registered(d_df, match_reg)
            u_mask, d_mask, pair_idx = _compute_coloc_mask(u_df, d_df_m, radius_px=radius_px)

            # Collect matched data rows
            if pair_idx:
                for i_u, i_d, dist in pair_idx:
                    row_u = u_df.loc[i_u] if i_u in u_df.index else None
                    row_d = d_df.loc[i_d] if i_d in d_df.index else None
                    if row_u is not None and row_d is not None:
                        matched_rows.append({
                            "ucnp_sif": u_name,
                            "dye_sif": d_name,
                            "ucnp_x_pix": row_u.get("x_pix", np.nan),
                            "ucnp_y_pix": row_u.get("y_pix", np.nan),
                            "ucnp_brightness": row_u.get("brightness_fit", np.nan),
                            "dye_x_pix": row_d.get("x_pix", np.nan),
                            "dye_y_pix": row_d.get("y_pix", np.nan),
                            "dye_brightness": row_d.get("brightness_fit", np.nan),
                            "distance_px": dist,
                        })

            # Stats for this specific image
            u_h = int(u_mask.sum()) if (u_mask is not None) else 0
            u_t = len(u_df) if isinstance(u_df, pd.DataFrame) else 0
            d_h = int(d_mask.sum()) if (d_mask is not None) else 0
            d_t = len(d_df) if isinstance(d_df, pd.DataFrame) else 0
            
            p_u = 100.0 * u_h / max(u_t, 1)
            p_d = 100.0 * d_h / max(d_t, 1)

            # 2. Setup Columns (UCNP | Dye | Recon [| Overlay])
            want_overlay = (show_overlay and isinstance(u_img, np.ndarray)
                            and isinstance(d_img, np.ndarray))
            cols = iter(st.columns(2 + int(show_recon) + int(want_overlay)))
            colL, colM = next(cols), next(cols)
            colR = next(cols) if show_recon else None
            colO = next(cols) if want_overlay else None
            text_style = "min-height: 50px; line-height: 1.2; word-wrap: break-word; margin-bottom: 5px;"

            # -- Left: UCNP Image --
            with colL:
                st.markdown(
                f"<div style='{text_style}'><b>UCNP:</b><br>{u_name}</div>", 
                unsafe_allow_html=True
                            )
                if isinstance(u_img, np.ndarray):
                    fig_u, ax_u = plt.subplots(figsize=(5,5))
                    ax_u.set_xticks([]); ax_u.set_yticks([])
                    norm = LogNorm() if use_lognorm else None
                    im_u = ax_u.imshow(u_img + 1, cmap=_resolve_cmap(cmap, region_ucnp), norm=norm, origin="lower",
                                       vmax=(u_vmax + 1) if (u_vmax and not use_lognorm) else None)
                    _corner_label(ax_u, "UCNP")
                    if show_colorbars:
                        _add_colorbar(fig_u, ax_u, im_u)
                else:
                    fig_u, ax_u = plt.subplots(figsize=(5,5))
                    ax_u.text(0.5,0.5,"No image", ha="center", va="center"); ax_u.axis("off")

            # -- Middle: Dye Image --
            with colM:
                st.markdown(
                f"<div style='{text_style}'><b>Dye:</b><br>{d_name}</div>", 
                unsafe_allow_html=True
                            )
                if isinstance(d_img, np.ndarray):
                    fig_d, ax_d = plt.subplots(figsize=(5,5))
                    ax_d.set_xticks([]); ax_d.set_yticks([])
                    norm = LogNorm() if use_lognorm else None
                    # With the overlay on, draw the dye exactly as it appears there (same
                    # colormap, background floor at the median) so the two compare directly.
                    im_d = ax_d.imshow(d_img + 1, cmap=d_ov_cmap if show_overlay else _resolve_cmap(cmap, region_dye),
                                       norm=norm, origin="lower",
                                       vmin=(float(np.nanmedian(d_img)) + 1) if (show_overlay and not use_lognorm) else None,
                                       vmax=(d_vmax + 1) if (d_vmax and not use_lognorm) else None)
                    _corner_label(ax_d, "Dye")
                    if show_colorbars:
                        _add_colorbar(fig_d, ax_d, im_d)
                else:
                    fig_d, ax_d = plt.subplots(figsize=(5,5))
                    ax_d.text(0.5,0.5,"No image", ha="center", va="center"); ax_d.axis("off")

            # -- Optional: Reconstruction Plot --
            if show_recon:
                with colR:
                    st.markdown(
                    f"<div style='{text_style}'><b>Reconstruction</b></div>", 
                    unsafe_allow_html=True
                                )
                    fig_r, ax_r = plt.subplots(figsize=(5,5))
                
                    # Set bounds
                    if isinstance(u_img, np.ndarray):
                        h, w = u_img.shape
                        ax_r.set_xlim(0, w); ax_r.set_ylim(0, h)
                    elif isinstance(d_img, np.ndarray):
                        h, w = d_img.shape
                        ax_r.set_xlim(0, w); ax_r.set_ylim(0, h)
                    else:
                        ax_r.axis("equal")
                
                    # Plot UCNP (Grey)
                    if not u_df.empty:
                        ax_r.scatter(u_df["x_pix"], u_df["y_pix"], c="dodgerblue", s=25, alpha=0.65, label="UCNP", edgecolors='none')


                    # Plot Dye (Red)
                    if not d_df_m.empty:
                        ax_r.scatter(d_df_m["x_pix"], d_df_m["y_pix"], c="firebrick", s=25, alpha=0.65, label="Dye", edgecolors='none')
                        # Circle the colocalized dye PSFs
                        if d_mask is not None and d_mask.any():
                            coloc_d = d_df_m[d_mask]
                            ax_r.scatter(coloc_d["x_pix"], coloc_d["y_pix"], marker="o", facecolors='none',  edgecolors='black', s=150, linewidths=1)

                    ax_r.set_aspect('equal')
                    ax_r.set_xticks([]); ax_r.set_yticks([])
                    ax_r.legend(loc='upper right', fontsize='x-small', framealpha=0.8)
                    # Reserve matching colorbar space so the reconstruction shrinks
                    # by the same amount as the image panels.
                    if show_colorbars:
                        _reserve_colorbar_space(fig_r, ax_r)
                    # bbox_inches=None disables Streamlit's tight crop so the full
                    # fixed-size canvas renders; with the reserved colorbar space
                    # all three panels keep the same image size.
                    st.pyplot(fig_r, bbox_inches=None)
                    plt.close(fig_r)

            # -- Optional: registered two-colour overlay --
            if want_overlay:
                with colO:
                    title = "Overlay" if reg is not None else "Overlay (unregistered)"
                    st.markdown(f"<div style='{text_style}'><b>{title}</b></div>",
                                unsafe_allow_html=True)
                    fig_o, ax_o = plt.subplots(figsize=(5, 5))
                    ax_o.imshow(_overlay_rgb(u_img, d_img, overlay_reg, u_ov_cmap, d_ov_cmap,
                                             u_vmax, d_vmax, use_lognorm), origin="lower")
                    ax_o.set_xticks([]); ax_o.set_yticks([])
                    _corner_label(ax_o, "Overlay")
                    if show_coloc_fits and u_mask is not None:
                        _overlay_circles(ax_o, u_df[u_mask], color="lime", alpha=0.9, label=False)
                    if show_colorbars:
                        _reserve_colorbar_space(fig_o, ax_o)
                    st.pyplot(fig_o, bbox_inches=None)
                    plt.close(fig_o)

            # Apply Overlays to Images
            if show_all_fits:
                _overlay_circles(ax_u, u_df, color="white", alpha=0.7, label=False)
                _overlay_circles(ax_d, d_df, color="white", alpha=0.7, label=False)
            if show_coloc_fits and u_mask is not None and d_mask is not None:
                _overlay_circles(ax_u, u_df[u_mask], color="lime", alpha=0.9, label=False)
                _overlay_circles(ax_d, d_df[d_mask], color="lime", alpha=0.9, label=False)

            # Display Stats text
            st.markdown(f"**Colocalized:** UCNP {u_h}/{u_t} ({p_u:.1f}%) · Dye {d_h}/{d_t} ({p_d:.1f}%)")

            # Render the image plots (bbox_inches=None keeps the full canvas so
            # the colorbar doesn't shrink the image relative to the recon panel).
            with colL: st.pyplot(fig_u, bbox_inches=None)
            with colM: st.pyplot(fig_d, bbox_inches=None)
            plt.close(fig_u); plt.close(fig_d)

            
            st.divider()

    # --- CSV Download & Analysis Plots ---
    if matched_rows:
        matched_df = pd.DataFrame(matched_rows)
        st.session_state['coloc_matched_df'] = matched_df

    with tab_plots:
        matched_df = st.session_state.get("coloc_matched_df", pd.DataFrame())
        if matched_df is None or matched_df.empty:
            st.info("No matched peaks yet.")
        else:
            st.markdown("#### Single-emitter calibration")
            cal_c1, cal_c2 = st.columns(2)
            with cal_c1:
                assume_ucnp_monomers = st.checkbox(
                    "Assume imaged UCNPs are monomers", value=True,
                    help="Use the Gaussian (GMM) fit of the pooled UCNP brightness across all "
                         "UCNP images/regions as the single-UCNP brightness. Uncheck to enter it manually.",
                )
            with cal_c2:
                dye_sif_files = utils.file_uploader_with_clear(
                    "Single-dye images (.sif), optional",
                    key="coloc_single_dye_uploads", type=["sif"], accept_multiple_files=True,
                    help="Upload single-dye .sif images to measure the single-dye brightness "
                         "via widefield (WF) brightness analysis. Leave empty to enter it manually.",
                )

            single_ucnp_brightness = _single_ucnp_brightness_ui(u_data, assume_ucnp_monomers)
            single_dye_brightness = _single_dye_brightness_ui(dye_sif_files)

            md = matched_df.copy()
            md["num_ucnps"] = md["ucnp_brightness"].astype(float) / max(single_ucnp_brightness, 1e-12)
            md["num_dyes"]  = md["dye_brightness"].astype(float)  / max(single_dye_brightness,  1e-12)
            
            # CSV Download Logic
            meta = io.StringIO()
            meta.write("Single Emitter,Brightness(pps)\n")
            meta.write(f"single_ucnp_brightness_pps,{single_ucnp_brightness}\n")
            meta.write(f"single_dye_brightness_pps,{single_dye_brightness}\n")
            meta.write("\n")
            md_csv = io.StringIO()
            md.to_csv(md_csv, index=False)
            md_csv.seek(0)
            payload = (meta.getvalue() + md_csv.getvalue()).encode("utf-8")
            
            st.download_button(
                "Download results (CSV)",
                data=payload,
                file_name=f"{stem}_colocalized_results.csv",
                mime="text/csv",
            )

            thresh_factor = st.number_input(
                "UCNP quality cutoff (× single UCNP brightness)",
                min_value=0.0, max_value=1.0, value=0.3, step=0.05,
                help="Exclude points with UCNP brightness below factor × single-UCNP brightness."
            )
            thresholded_df = md[md["ucnp_brightness"] >= thresh_factor * single_ucnp_brightness].copy()

            fig_sc2, ax_sc2 = plt.subplots(figsize=(6,5))
            xvals = md["num_ucnps"].to_numpy(dtype=float)
            yvals = md["num_dyes"].to_numpy(dtype=float)
            ax_sc2.scatter(md["num_ucnps"].to_numpy(), md["num_dyes"].to_numpy(), alpha=0.6, clip_on=False)
            ax_sc2.set_xlabel("Number of UCNPs per PSF")
            ax_sc2.set_ylabel("Number of Dyes per PSF")
            ax_sc2.set_title("Matched UCNPs")
            _autoscale_xy(ax_sc2, xvals, yvals)
            HWT_aesthetic()
            ax_sc2.grid(True, which="both", linestyle=":", color="lightgrey", alpha=0.7)

            colA, colB = st.columns(2)
            with colA:
                st.pyplot(fig_sc2)
                plt.close(fig_sc2)

            msk = (thresholded_df["num_ucnps"] >= 0) & (thresholded_df["num_ucnps"] <= 2)
            y_subset = thresholded_df.loc[msk, "num_dyes"].dropna().to_numpy()
            y_subset = y_subset[np.isfinite(y_subset)]
            fig_h2, hf = _dyes_per_ucnp_figure(y_subset, overall_dye_total, overall_ucnp_total)

            with colB:
                st.pyplot(fig_h2)
                plt.close(fig_h2)
                if hf["mu"] is not None and hf["n"] >= 2:
                    st.caption(
                        f"All dyes / UCNPs = {overall_dye_total} dye PSFs / {overall_ucnp_total} "
                        f"UCNP PSFs fitted; n = {hf['n']} single UCNPs plotted. "
                        f"μ, σ: {hf['method'].lower()}"
                        + (f" (χ²ᵣ = {hf['redchi2']:.2f})" if np.isfinite(hf["redchi2"]) else "")
                        + f"; bins by Freedman–Diaconis"
                        + (f"; {hf['n_out']} far outliers (> 3 IQR) excluded." if hf["n_out"] else "."))

    # --- Representative field of view ---
    # The image pair whose UCNP colocalization % is closest to the pooled value, so the
    # example shown matches the headline number rather than the best-looking field.
    if pair_stats:
        st.divider()
        st.markdown("#### Representative field of view")
        rep_u, rep_d, rep_pu, rep_pd = min(pair_stats, key=lambda t: abs(t[2] - ov_u_pct))
        u_bundle, d_bundle = u_data.get(rep_u, {}), d_data.get(rep_d, {})
        if isinstance(u_bundle.get("image"), np.ndarray) and isinstance(d_bundle.get("image"), np.ndarray):
            rep_u_df = u_bundle.get("df", pd.DataFrame())
            rep_u_mask, _, _ = _compute_coloc_mask(
                rep_u_df, _registered(d_bundle.get("df", pd.DataFrame()), match_reg), radius_px=radius_px)
            fig_fov = _representative_fov_figure(
                rep_u, rep_d, u_bundle, d_bundle, overlay_reg, u_ov_cmap, d_ov_cmap,
                _resolve_cmap(cmap, region_ucnp), _resolve_cmap(cmap, region_dye),
                u_vmax, d_vmax, use_lognorm,
                u_mask_df=rep_u_df[rep_u_mask] if (show_coloc_fits and rep_u_mask is not None) else None)
            st.caption(
                f"**{rep_u}** + **{rep_d}**: UCNP colocalization {rep_pu:.1f}% (pooled "
                f"{ov_u_pct:.1f}%), dye {rep_pd:.1f}% (pooled {ov_d_pct:.1f}%). Dye shown "
                + ("registered to the UCNP frame." if reg is not None else "unregistered.")
                + (" Green circles: colocalized UCNPs." if show_coloc_fits else ""))
            st.pyplot(fig_fov, bbox_inches=None)
            png, svg = io.BytesIO(), io.BytesIO()
            fig_fov.savefig(png, format="png", dpi=300, bbox_inches="tight")
            fig_fov.savefig(svg, format="svg", bbox_inches="tight")
            plt.close(fig_fov)
            d1, d2, _ = st.columns([1, 1, 4])
            d1.download_button("Download PNG", png.getvalue(), file_name=f"{stem}_representative_fov.png",
                               mime="image/png", key="coloc_fov_png")
            d2.download_button("Download SVG", svg.getvalue(), file_name=f"{stem}_representative_fov.svg",
                               mime="image/svg+xml", key="coloc_fov_svg")

if __name__ == "__main__":
    run()