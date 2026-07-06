"""Process and compare emission spectra exported by the Get Spectra tool.

Consumes the CSVs produced by ``tools/get_spectra.py`` (columns:
``File``, ``Particle_ID``, ``Wavelength_nm``, ``Intensity``) and lets the user:

  * plot every uploaded CSV as its own interactive figure,
  * exclude spectra by clicking a trace or dragging a box over a region,
  * optionally subtract a spline (pybaselines) baseline from each spectrum,
  * normalize by max pixel, max-in-range, or total area,
  * volume-normalize by a per-file effective radius (r_eff),
  * pick each file's color (true-color picker; individual traces are shades),
  * combine every CSV's average into one legended figure, and
  * drag two regions on the combined plot to read per-file area ratios.

All spectra are cropped at ``CROP_NM`` (artifacts dominate past it). The heavy
per-spectrum processing (baseline + normalization) is cached so toggling
exclusions only re-renders — it doesn't recompute.
"""

import io
import colorsys

import numpy as np
import pandas as pd
import streamlit as st

from pybaselines import Baseline

import plotly.graph_objects as go

# Columns expected from a Get Spectra export.
REQUIRED_COLS = ["File", "Particle_ID", "Wavelength_nm", "Intensity"]

# Spectra are cropped here — past this there tend to be detector artifacts.
CROP_NM = 875.0

# Normalization choices (single-select).
NORM_NONE = "None"
NORM_MAX = "Max pixel value"
NORM_MAX_RANGE = "Max value in range"
NORM_AREA = "Total area"
NORM_AREA_RANGE = "Area in range"
NORM_VOLUME = "Volume (r_eff)"
NORM_METHODS = [NORM_NONE, NORM_MAX, NORM_MAX_RANGE, NORM_AREA, NORM_AREA_RANGE,
                NORM_VOLUME]

# Methods that need a user-supplied wavelength range.
NORM_RANGE_METHODS = {NORM_MAX_RANGE, NORM_AREA_RANGE}

# Baseline choices (single-select).
BASELINE_OFF = "Off"
BASELINE_MEAN = "Mean (600–630 nm)"
BASELINE_SPLINE = "Spline (pybaselines)"
BASELINE_METHODS = [BASELINE_OFF, BASELINE_MEAN, BASELINE_SPLINE]

# Default window for the mean-subtraction baseline.
BASELINE_MEAN_LO = 600.0
BASELINE_MEAN_HI = 630.0

# Per-trace line width is mapped from relative illumination (when present).
WIDTH_MIN = 1.5
WIDTH_MAX = 7.0
WIDTH_DEFAULT = 4.0

# Illumination tiers (fraction of the brightest calibration point): Low / Medium
# / High, split into thirds. Spectra with no illumination column are never filtered.
ILLUM_LOW_MAX = 1.0 / 3.0
ILLUM_MED_MAX = 2.0 / 3.0
ILLUM_TIERS = ["Low", "Medium", "High"]

# Spectral regions for the integrated-area bar plot: (name, lo_nm, hi_nm, color).
REGIONS = [
    ("Blue", 400.0, 505.0, "#4589ff"),
    ("Green", 505.0, 605.0, "#6fdc8c"),
    ("Red", 605.0, 705.0, "#fa4d56"),
    ("NIR", 705.0, 900.0, "#d12771"),
]

# Shading + labels for the two peak-ratio regions.
RATIO_FILL = ["rgba(31,119,180,0.20)", "rgba(214,39,40,0.20)"]
RATIO_LABELS = ["A", "B"]

# Combined-plot interactive tools (radio shown underneath the plot).
TOOL_NONE = "None"
TOOL_RATIO = "Peak ratio"
TOOL_POP = "Population estimation"
COMBINED_TOOLS = [TOOL_NONE, TOOL_RATIO, TOOL_POP]

# Fill opacity for the population Gaussian shadings.
POP_FILL_ALPHA = 0.18


# --- Signal processing ------------------------------------------------------
def _spline_baseline(wvl, y, lam=1e3, p=0.01, num_knots=100, niter=10):
    """Penalized-spline asymmetric baseline (pybaselines ``pspline_asls``).

    The spline analogue of ALS: fits a smooth B-spline that follows the baseline
    while ignoring peaks. ``lam`` sets smoothness, ``p`` the asymmetry, and
    ``num_knots`` the number of spline knots. Returns the baseline array.
    """
    y = np.asarray(y, dtype=float)
    wvl = np.asarray(wvl, dtype=float)
    n = y.size
    if n < 5:
        return np.zeros_like(y)

    finite = np.isfinite(y)
    if not finite.all():
        if finite.sum() < 5:
            return np.zeros_like(y)
        y = np.interp(np.arange(n), np.flatnonzero(finite), y[finite])

    knots = int(min(num_knots, max(n // 2, 4)))
    fitter = Baseline(x_data=wvl)
    baseline, _ = fitter.pspline_asls(y, lam=lam, p=p, num_knots=knots, max_iter=niter)
    return np.asarray(baseline, dtype=float)


def _apply_baseline(wvl, y, baseline):
    """Subtract a baseline from ``y`` according to ``baseline`` (a dict with a
    ``method`` key) and return the corrected array.

    * ``"mean"``: subtract the mean value found in [lo, hi] (default
      500–750 nm) from every point — a simple, fast offset correction.
    * ``"spline"``: subtract a penalized-spline asymmetric baseline.
    """
    if not baseline:
        return y
    if baseline["method"] == "mean":
        lo, hi = baseline["lo"], baseline["hi"]
        m = (wvl >= lo) & (wvl <= hi) & np.isfinite(y)
        if not m.any():                       # window empty — fall back to global mean
            m = np.isfinite(y)
        if m.any():
            return y - float(np.nanmean(y[m]))
        return y
    if baseline["method"] == "spline":
        return y - _spline_baseline(
            wvl, y, lam=baseline["lam"], p=baseline["p"],
            num_knots=baseline.get("num_knots", 100), niter=baseline.get("niter", 10),
        )
    return y


def _width_for_illum(rel):
    """Map a relative illumination (0–1) to a line width. NaN/None → default."""
    if rel is None or not np.isfinite(rel):
        return WIDTH_DEFAULT
    r = min(max(float(rel), 0.0), 1.0)
    return WIDTH_MIN + r * (WIDTH_MAX - WIDTH_MIN)


def _illum_tier(rel):
    """Classify a relative illumination (0–1) into Low / Medium / High tiers, or
    ``None`` when there's no illumination data (NaN) — those are never filtered."""
    if rel is None or not np.isfinite(rel):
        return None
    if rel < ILLUM_LOW_MAX:
        return "Low"
    if rel < ILLUM_MED_MAX:
        return "Medium"
    return "High"


def _normalize_spectrum(wvl, inten, method, rng, volume, baseline=None):
    """Baseline-correct then normalize a single spectrum by the chosen method.

    Order: (1) optional baseline subtraction, (2) clip negatives, (3) the chosen
    normalization. The methods are mutually exclusive — ``NORM_VOLUME`` divides by
    the particle volume; the others do max/area scaling. ``baseline`` is ``None``
    or a dict with a ``method`` key (see :func:`_apply_baseline`).
    """
    wvl = np.asarray(wvl, dtype=float)
    y = np.asarray(inten, dtype=float)

    y = _apply_baseline(wvl, y, baseline)

    # Clip negatives to 0 — baseline subtraction can push noise below zero;
    # those values are non-physical and would skew area/peak normalization.
    y = np.where(np.isfinite(y) & (y < 0), 0.0, y)

    if method == NORM_VOLUME:
        if volume and volume > 0:
            y = y / volume
    elif method == NORM_MAX:
        m = np.nanmax(y) if y.size else 0.0
        if m:
            y = y / m
    elif method == NORM_MAX_RANGE:
        lo, hi = rng
        mask = (wvl >= lo) & (wvl <= hi)
        if mask.any():
            m = np.nanmax(y[mask])
            if m:
                y = y / m
    elif method == NORM_AREA:
        order = np.argsort(wvl)
        area = np.trapz(y[order], wvl[order])
        if area:
            y = y / area
    elif method == NORM_AREA_RANGE:
        lo, hi = rng
        order = np.argsort(wvl)
        ww, yy = wvl[order], y[order]
        mask = (ww >= lo) & (ww <= hi) & np.isfinite(yy)
        if mask.sum() >= 2:
            area = np.trapz(yy[mask], ww[mask])
            if area:
                y = y / area

    return wvl, y


def _average_spectra(specs, npts=400):
    """Average ``(wvl, y)`` spectra on a shared grid (union range, NaN-padded)."""
    if not specs:
        return np.array([]), np.array([])
    lo = min(np.min(w) for w, _ in specs)
    hi = max(np.max(w) for w, _ in specs)
    grid = np.linspace(lo, hi, npts)
    stack = []
    for w, y in specs:
        order = np.argsort(w)
        stack.append(np.interp(grid, w[order], y[order], left=np.nan, right=np.nan))
    return grid, np.nanmean(np.vstack(stack), axis=0)


def _area_in_range(wvl, y, lo, hi):
    """Trapezoidal integral of ``y`` over [lo, hi] (finite samples only)."""
    wvl = np.asarray(wvl, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (wvl >= lo) & (wvl <= hi) & np.isfinite(y)
    return float(np.trapz(y[m], wvl[m])) if m.sum() >= 2 else 0.0


def _integrate_regions(wvl, y):
    """Trapezoidal integral of ``y`` over each band in REGIONS."""
    return [_area_in_range(wvl, y, lo, hi) for _n, lo, hi, _c in REGIONS]


def _gaussian(x, amp, mu, sigma):
    """A single Gaussian peak."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# How far (nm) a fitted peak center may drift from the user's clicked position.
POP_MAX_SHIFT = 25.0


def _fit_population_gaussians(wvl, y, centers, max_shift=POP_MAX_SHIFT):
    """Fit a sum of Gaussians (one per clicked center) to a single spectrum.

    Returns a list of component dicts ``{center, amp, mu, sigma, area}`` aligned
    to ``centers`` (sorted ascending), or ``None`` if the fit can't be done. Each
    peak's center is bounded to ``center ± max_shift`` so components stay tied to
    the clicks the user placed. Area is the analytic Gaussian integral
    ``amp·sigma·√(2π)`` — the basis for the relative-population percentages.
    """
    from scipy.optimize import curve_fit

    wvl = np.asarray(wvl, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(wvl) & np.isfinite(y)
    wvl, y = wvl[m], y[m]
    centers = sorted(float(c) for c in centers)
    if wvl.size < 5 or not centers:
        return None

    order = np.argsort(wvl)
    wvl, y = wvl[order], y[order]
    yspan = float(np.nanmax(y) - np.nanmin(y)) or 1.0
    n = len(centers)

    p0, lo, hi = [], [], []
    for c in centers:
        amp0 = max(float(np.interp(c, wvl, y)), yspan * 0.05)
        p0 += [amp0, c, 20.0]
        lo += [0.0, c - max_shift, 2.0]
        hi += [np.inf, c + max_shift, 150.0]

    def model(x, *p):
        out = np.zeros_like(x, dtype=float)
        for i in range(n):
            out += _gaussian(x, p[3 * i], p[3 * i + 1], p[3 * i + 2])
        return out

    try:
        popt, _ = curve_fit(model, wvl, y, p0=p0, bounds=(lo, hi), maxfev=20000)
    except Exception:
        return None

    comps = []
    for i in range(n):
        amp, mu, sigma = popt[3 * i:3 * i + 3]
        comps.append({
            "center": centers[i], "amp": float(amp), "mu": float(mu),
            "sigma": float(sigma), "area": float(amp * sigma * np.sqrt(2 * np.pi)),
        })
    return comps


def _rgba(color, alpha):
    """Convert a ``#rrggbb`` (or already-``rgb()``) color to an ``rgba()`` string."""
    if isinstance(color, str) and color.startswith("rgb(") and color.endswith(")"):
        inner = color[4:-1]
        return f"rgba({inner},{alpha})"
    r, g, b = _hex_to_rgb01(color)
    return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha})"


def _y_axis_label(method, volume):
    """Plotly y-axis title reflecting the active units / normalization."""
    if method == NORM_MAX:
        return "Intensity (peak-normalized, a.u.)"
    if method == NORM_MAX_RANGE:
        return "Intensity (range-peak-normalized, a.u.)"
    if method == NORM_AREA:
        return "Intensity (area-normalized, nm<sup>-1</sup>)"
    if method == NORM_AREA_RANGE:
        return "Intensity (range-area-normalized, nm<sup>-1</sup>)"
    if method == NORM_VOLUME:
        return "Intensity (photons / s / px / nm<sup>3</sup>)"
    return "Intensity (photons / s / px)"


def _processing_summary(method, baseline, volume_norm, rng):
    """One-line description of the active processing chain."""
    norm = {
        NORM_NONE: "none (raw photons/s/px)",
        NORM_MAX: "peak max → 1",
        NORM_MAX_RANGE: (f"max in {rng[0]:.0f}–{rng[1]:.0f} nm → 1"
                         if rng and rng[1] > rng[0] else "max in range → 1"),
        NORM_AREA: "total area → 1",
        NORM_AREA_RANGE: (f"area in {rng[0]:.0f}–{rng[1]:.0f} nm → 1"
                          if rng and rng[1] > rng[0] else "area in range → 1"),
        NORM_VOLUME: "÷ volume (per-file r_eff)",
    }[method]
    parts = [f"crop ≤ {CROP_NM:.0f} nm", f"Normalization: {norm}"]
    if not baseline:
        parts.append("baseline: off")
    elif baseline["method"] == "mean":
        parts.append(f"baseline: mean in {baseline['lo']:.0f}–{baseline['hi']:.0f} nm subtracted")
    elif baseline["method"] == "spline":
        parts.append(f"baseline: spline (λ={baseline['lam']:.0e}, p={baseline['p']:.3f})")
    return " · ".join(parts)


# --- Colors -----------------------------------------------------------------
def _hex_to_rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _shades(base_hex, n):
    """``n`` shades of ``base_hex`` (same hue/sat, varied lightness) as rgb()."""
    if n <= 0:
        return []
    r, g, b = _hex_to_rgb01(base_hex)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    lights = [l] if n == 1 else np.linspace(0.30, 0.78, n)
    out = []
    for li in lights:
        rr, gg, bb = colorsys.hls_to_rgb(h, float(li), s)
        out.append(f"rgb({int(round(rr * 255))},{int(round(gg * 255))},{int(round(bb * 255))})")
    return out


# Colormaps to sample the picker palette from, with how many colors to draw from
# each. Longer gradients (viridis/plasma 7, magma 6) give smooth same-family
# series; crest was dropped for being too close to mako.
PALETTE_COLORMAPS = [
    ("mako", 4),
    ("rocket", 5),
    ("flare", 5),
    ("viridis", 7),
    ("plasma", 7),
    ("magma", 6),
]


@st.cache_data(show_spinner=False)
def _curated_palette():
    """Graph-legible colors sampled evenly from perceptual colormaps (see
    PALETTE_COLORMAPS). Returns a list of ``(label, hex)``; near-white tints are
    dropped so every swatch reads on a white background."""
    import matplotlib
    try:
        import seaborn  # noqa: F401  (registers mako/rocket/flare)
    except Exception:
        pass

    pal, seen = [], set()
    for name, count in PALETTE_COLORMAPS:
        try:
            cmap = matplotlib.colormaps[name]
        except Exception:
            continue
        for j, x in enumerate(np.linspace(0.12, 0.88, count), start=1):
            h = matplotlib.colors.to_hex(cmap(float(x))).lower()
            r, g, b = _hex_to_rgb01(h)
            _hh, l, _s = colorsys.rgb_to_hls(r, g, b)
            if l > 0.85 or h in seen:       # too light, or duplicate
                continue
            seen.add(h)
            pal.append((f"{name} {j}", h))
    return pal or [("default", "#1f77b4")]


@st.cache_data(show_spinner=False)
def _default_color_indices():
    """Palette indices in the order newly-uploaded files should default to:
    plasma 7, 6, 5 … 1 first (a clean descending gradient), then the rest of the
    palette in its natural order."""
    pal = _curated_palette()
    plasma = [k for k, (lbl, _h) in enumerate(pal) if lbl.startswith("plasma")]
    order = list(reversed(plasma))
    order += [k for k in range(len(pal)) if k not in plasma]
    return order


def _color_select_ui(file_name, default_idx):
    """Compact color chooser: a swatch strip previewing every curated color (the
    selected one outlined), a dropdown to pick, and a wide swatch of the choice.
    Returns the chosen hex. This replaces the full-spectrum ``color_picker``."""
    pal = _curated_palette()
    n = len(pal)
    sel_key = f"colorsel_{file_name}"
    idx = st.session_state.get(sel_key, int(default_idx) % n)
    idx = idx if isinstance(idx, int) and 0 <= idx < n else int(default_idx) % n

    strip = "<div style='line-height:0'>" + "".join(
        f'<span title="{lbl} — {h}" style="display:inline-block;width:24px;height:24px;'
        f'background:{h};margin:2px;border-radius:4px;vertical-align:top;'
        f'box-shadow:{"0 0 0 3px #000 inset" if i == idx else "0 0 0 1px #bbb inset"};'
        f'"></span>'
        for i, (lbl, h) in enumerate(pal)
    ) + "</div>"
    st.markdown(strip, unsafe_allow_html=True)

    choice = st.selectbox(
        "Color", options=list(range(n)), index=idx,
        format_func=lambda i: f"{pal[i][0]}  ({pal[i][1]})", key=sel_key,
    )
    chosen = pal[choice][1]
    st.markdown(
        f'<div style="width:100%;height:20px;background:{chosen};'
        f'border:1px solid #666;border-radius:4px;margin-top:4px;"></div>',
        unsafe_allow_html=True,
    )
    return chosen


# --- Cached per-CSV processing ----------------------------------------------
@st.cache_data(show_spinner=False)
def _process_csv(raw_bytes, method, rng, volume, baseline_tuple):
    """Parse a CSV, crop at CROP_NM, and baseline/normalize every spectrum.

    Returns ``(specs, error)`` where ``specs`` is a list of
    ``(key, wvl, y, rel_illum)`` (key = "<source SIF>::<particle id>",
    ``rel_illum`` = per-particle Relative_Illumination or NaN when absent).
    Cached on the raw bytes + processing params, so exclusion toggles never
    recompute this.
    """
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        return None, f"Could not read file: {e}"

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, "missing column(s): " + ", ".join(missing)

    df = df[df["Wavelength_nm"] <= CROP_NM]
    has_illum = "Relative_Illumination" in df.columns

    bl = _decode_baseline_tuple(baseline_tuple)

    specs = []
    for (src, pid), g in df.groupby(["File", "Particle_ID"], sort=True):
        g = g.sort_values("Wavelength_nm")
        wvl, y = _normalize_spectrum(
            g["Wavelength_nm"].to_numpy(), g["Intensity"].to_numpy(),
            method, rng, volume, bl,
        )
        rel = np.nan
        if has_illum:
            vals = pd.to_numeric(g["Relative_Illumination"], errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size:
                rel = float(np.nanmedian(vals))
        specs.append((f"{src}::{pid}", wvl, y, rel))
    return specs, None


def _decode_baseline_tuple(baseline_tuple):
    """Turn the cache-friendly baseline tuple back into a dict (or None)."""
    if not baseline_tuple:
        return None
    kind = baseline_tuple[0]
    if kind == "mean":
        return {"method": "mean", "lo": baseline_tuple[1], "hi": baseline_tuple[2]}
    if kind == "spline":
        return {"method": "spline", "lam": baseline_tuple[1], "p": baseline_tuple[2],
                "num_knots": baseline_tuple[3], "niter": baseline_tuple[4]}
    return None


def _traces_in_box(specs, xr, yr):
    """Indices of spectra whose curve passes through the box [xr] × [yr].

    Interpolates each trace inside the x-range so traces are caught even when no
    raw sample falls in the box — fixes 'box misses the trace' flakiness."""
    lo, hi = sorted(float(v) for v in xr[:2])
    ylo, yhi = sorted(float(v) for v in yr[:2])
    xs = np.linspace(lo, hi, 80)
    hits = []
    for i, (_k, wvl, y, _rel) in enumerate(specs):
        yi = np.interp(xs, wvl, y, left=np.nan, right=np.nan)
        if np.any((yi >= ylo) & (yi <= yhi)):
            hits.append(i)
    return hits


# --- Per-file rendering -----------------------------------------------------
def _render_region_bar(grid, mean, file_key):
    """One horizontal bar spanning the spectrum's wavelength range, split into the
    Blue/Green/Red/NIR bands with each segment's width proportional to its share
    of the total integrated area. A compact composition readout under the trace.
    """
    areas = np.asarray(_integrate_regions(grid, mean), dtype=float)
    total = float(areas.sum())
    if total <= 0:
        st.caption("Integrated regions: no positive area to compose.")
        return
    fracs = areas / total

    xmin = float(np.nanmin(grid))
    xmax = float(np.nanmax(grid))
    span = xmax - xmin if xmax > xmin else 1.0

    bfig = go.Figure()
    cursor = xmin
    for i, (name, _lo, _hi, color) in enumerate(REGIONS):
        frac = float(fracs[i])
        w = frac * span
        bfig.add_trace(go.Bar(
            y=["Regions"], x=[w], base=cursor, orientation="h",
            marker_color=color, name=name,
            text=[f"{frac:.0%}"] if frac >= 0.06 else [""],
            textposition="inside", insidetextanchor="middle",
            hovertemplate=f"{name}: {frac:.1%} (area {areas[i]:.3g})<extra></extra>",
        ))
        cursor += w

    bfig.update_layout(
        barmode="overlay",
        xaxis=dict(title="Wavelength (nm)", range=[xmin, xmax]),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=6, b=40), height=120,
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
        bargap=0.0,
    )
    st.plotly_chart(bfig, use_container_width=True, key=f"bar_{file_key}")


def _render_file(file_key, specs, opts, base_color, volume):
    """Render one CSV's interactive figure; return its average ``(grid, mean)``.

    Individual spectra are shades of ``base_color``; the average is black.

    Selection is strictly **one-directional to avoid the sync bugs that plagued
    earlier versions**: a click or box *only ever excludes* the traces it touches.
    Re-inclusion is done with plain buttons (Clear, or per-trace). Every state
    change bumps this figure's own nonce, which remounts *only this* chart so its
    selection is consumed exactly once (no lingering box to re-fire, and no shared
    state — one figure's edit can never disturb another's exclusions).
    """
    fig_nonce = st.session_state.plot_nonce.setdefault(file_key, 0)

    # Keep exclusion bookkeeping against the FULL key set so the illumination
    # filter below never silently drops a manual exclusion.
    excluded = st.session_state.spectra_excluded.setdefault(file_key, set())
    excluded.intersection_update([k for k, _, _, _ in specs])

    # Illumination-tier filter: hide spectra whose tier isn't selected. Spectra
    # with no illumination data (tier None) are always shown.
    enabled_tiers = opts.get("illum_tiers")
    if enabled_tiers is not None:
        vis_specs = [s for s in specs
                     if _illum_tier(s[3]) is None or _illum_tier(s[3]) in enabled_tiers]
    else:
        vis_specs = list(specs)
    n_hidden_illum = len(specs) - len(vis_specs)

    spec_keys = [k for k, _, _, _ in vis_specs]  # indices match plotted traces
    shades = _shades(base_color, len(vis_specs))
    has_illum = any(np.isfinite(rel) for _, _, _, rel in vis_specs)
    fig = go.Figure()
    included = []  # (wvl, y) for the average

    for idx, (key, wvl, y, rel) in enumerate(vis_specs):
        is_excluded = key in excluded
        width = _width_for_illum(rel) if has_illum else WIDTH_DEFAULT
        illum_txt = f" · illum {rel:.0%}" if np.isfinite(rel) else ""
        fig.add_trace(go.Scatter(
            x=wvl, y=y, mode="lines+markers",
            line=dict(color="lightgrey" if is_excluded else shades[idx], width=width),
            marker=dict(size=5, opacity=0.01),
            opacity=0.5 if is_excluded else 1.0,
            name=key.split("::", 1)[-1],
            showlegend=False,
            hovertemplate=f"{key}{illum_txt}<br>%{{x:.1f}} nm, %{{y:.3g}}<extra></extra>",
        ))
        if not is_excluded:
            included.append((wvl, y))

    grid, mean = _average_spectra(included)
    if opts["show_average"] and grid.size:
        fig.add_trace(go.Scatter(
            x=grid, y=mean, mode="lines",
            line=dict(color="black", width=6),
            name="Average", showlegend=True, hoverinfo="skip",
        ))

    if has_illum:  # legend proxies explaining the line-thickness encoding
        for label, w in (("Low illumination", WIDTH_MIN), ("High illumination", WIDTH_MAX)):
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color="grey", width=w),
                name=label, showlegend=True, hoverinfo="skip",
            ))

    fig.update_layout(
        xaxis_title="Wavelength (nm)",
        yaxis_title=_y_axis_label(opts["method"], volume),
        margin=dict(l=60, r=10, t=10, b=40),
        height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
        dragmode="select",  # click-drag draws a selection box
    )
    if included:  # rescale y to the included spectra (excluded greys may clip)
        ys = np.concatenate([y for _, y in included])
        ys = ys[np.isfinite(ys)]
        if ys.size:
            lo, hi = float(ys.min()), float(ys.max())
            span = hi - lo
            pad = 0.05 * span if span > 0 else (abs(hi) * 0.05 or 1.0)
            fig.update_yaxes(range=[lo - pad, hi + pad])

    n_vis = len(vis_specs)
    n_incl = len(included)
    illum_note = (" · line thickness ∝ relative illumination"
                  if has_illum else " · no illumination column in CSV")
    if n_hidden_illum:
        illum_note += f" · {n_hidden_illum} hidden by illumination filter"
    c_cap, c_btn = st.columns([4, 1])
    with c_cap:
        st.caption(
            f"{n_vis} shown · {n_incl} included · {n_vis - n_incl} excluded "
            f"— click a trace or drag a box to **exclude** it (re-include with the "
            f"buttons below or *Clear*).{illum_note}"
        )
    with c_btn:
        if st.button("Clear exclusions", key=f"clear_{file_key}", disabled=not excluded):
            excluded.clear()
            st.session_state.plot_nonce[file_key] = fig_nonce + 1  # remount → wipe box
            st.rerun()

    event = st.plotly_chart(
        fig, use_container_width=True,
        on_select="rerun", selection_mode=["points", "box"],
        key=f"plot_{file_key}_{fig_nonce}",
    )
    try:
        sel = event["selection"]
        boxes = sel.get("box") or []
        pts = sel.get("points") or []
    except (TypeError, KeyError, IndexError):
        boxes, pts = [], []

    # Resolve the selection into target trace indices. Selection ONLY excludes,
    # so acting is idempotent: reprocessing another figure's lingering selection
    # on a shared rerun can add nothing new and is therefore harmless.
    targets = set()
    if boxes:
        xr = boxes[0].get("x") or []
        yr = boxes[0].get("y") or []
        if len(xr) >= 2 and len(yr) >= 2:
            targets = set(_traces_in_box(vis_specs, xr, yr))
    elif pts:
        for p in pts:
            cn = p.get("curve_number")
            if cn is not None and cn < len(spec_keys):
                targets.add(cn)

    newly = [spec_keys[i] for i in targets if spec_keys[i] not in excluded]
    if newly:
        excluded.update(newly)
        # Bump THIS figure's nonce only: remount to consume the selection so the
        # same box can't re-fire and nothing lingers to fight a later re-include.
        st.session_state.plot_nonce[file_key] = fig_nonce + 1
        st.rerun()

    # Integrated-region composition as a single horizontal bar (see helper).
    if opts["barplot"] and grid.size:
        _render_region_bar(grid, mean, file_key)

    # Re-include control. Plain buttons only — stateless and one-directional, so
    # unlike a multiselect they can't lag behind `excluded` or fire a stale
    # callback on another figure's rerun. Clicking one re-includes that trace and
    # bumps the nonce so the lingering selection box can't immediately re-exclude.
    with st.expander(f"Excluded spectra ({len(excluded)}) — click to re-include"):
        if excluded:
            for k in sorted(excluded):
                if st.button(f"↩ {k.split('::', 1)[-1]}", key=f"reinc_{file_key}_{k}"):
                    excluded.discard(k)
                    st.session_state.plot_nonce[file_key] = fig_nonce + 1
                    st.rerun()
        else:
            st.caption("No spectra excluded.")

    return grid, mean, n_incl


# --- App --------------------------------------------------------------------
def run():
    st.session_state.setdefault("spectra_excluded", {})  # file_key -> set(keys)
    st.session_state.setdefault("plot_nonce", {})        # file_key -> int (remount to clear)
    st.session_state.setdefault("show_combined", False)
    st.session_state.setdefault("ratio_ranges", {})  # key -> [(lo,hi)|None, ...]
    st.session_state.setdefault("ratio_ptr", {})     # key -> next slot (0/1)
    st.session_state.setdefault("ratio_last", {})    # key -> last box signature
    st.session_state.setdefault("pop_centers", {})   # key -> [peak x, ...]
    st.session_state.setdefault("pop_last", {})      # key -> last click signature
    st.session_state.setdefault("reff_store", {})    # file -> r_eff (survives toggling)

    with st.sidebar:
        st.header("Inputs")
        csv_files = st.file_uploader(
            "Spectra CSVs (from Get Spectra)",
            type=["csv"], accept_multiple_files=True,
        )

        st.divider()
        st.header("Processing")
        show_average = st.checkbox("Show average of all spectra", value=True)
        barplot = st.checkbox(
            "Barplot integrated regions", value=False,
            help="Integrate each CSV's average over Blue/Green/Red/NIR bands.",
        )

        st.caption("Illumination tiers to include "
                   f"(Low <{ILLUM_LOW_MAX:.0%} · Medium <{ILLUM_MED_MAX:.0%} · "
                   "High ≥ that, of the brightest calibration point):")
        ci1, ci2, ci3 = st.columns(3)
        illum_tiers = set()
        if ci1.checkbox("Low", value=True, key="illum_low"):
            illum_tiers.add("Low")
        if ci2.checkbox("Med", value=True, key="illum_med"):
            illum_tiers.add("Medium")
        if ci3.checkbox("High", value=True, key="illum_high"):
            illum_tiers.add("High")

        baseline_method = st.radio(
            "Baseline correction", BASELINE_METHODS, index=0,
            help="Mean: subtract the average value in a window (default 600–630 nm) "
                 "from each trace. Spline: subtract a penalized-spline asymmetric "
                 "baseline (pybaselines pspline_asls).",
        )
        baseline = None
        baseline_tuple = None
        if baseline_method == BASELINE_MEAN:
            c1, c2 = st.columns(2)
            blo = c1.number_input("Mean window min (nm)", value=BASELINE_MEAN_LO)
            bhi = c2.number_input("Mean window max (nm)", value=BASELINE_MEAN_HI)
            blo, bhi = min(blo, bhi), max(blo, bhi)
            baseline = {"method": "mean", "lo": blo, "hi": bhi}
            baseline_tuple = ("mean", blo, bhi)
        elif baseline_method == BASELINE_SPLINE:
            lam_log = st.slider("Baseline stiffness (log₁₀ λ)", 0.0, 7.0, 3.0, 0.5)
            p_asym = st.slider("Baseline asymmetry (p)", 0.001, 0.100, 0.010, 0.001,
                               format="%.3f")
            n_knots = st.slider("Spline knots", 10, 200, 100, 10)
            baseline = {"method": "spline", "lam": 10.0 ** lam_log, "p": p_asym,
                        "num_knots": n_knots, "niter": 10}
            baseline_tuple = ("spline", baseline["lam"], baseline["p"],
                              baseline["num_knots"], baseline["niter"])

        method = st.radio(
            "Normalize by", NORM_METHODS, index=0,
            help="Mutually exclusive. Volume (r_eff) divides each CSV's intensities "
                 "by V = (4/3)·π·r_eff³ (per-file r_eff below).",
        )
        rng = (0.0, 0.0)
        if method in NORM_RANGE_METHODS:
            c1, c2 = st.columns(2)
            lo = c1.number_input("Range min (nm)", value=600.0)
            hi = c2.number_input("Range max (nm)", value=700.0)
            rng = (min(lo, hi), max(lo, hi))
        volume_norm = method == NORM_VOLUME

        # Per-file legend name, order, color, r_eff (needs the file list here).
        file_colors = {}
        file_volumes = {}
        file_names = {}
        file_order = {}
        file_show = {}
        file_reff = {}
        if csv_files:
            st.divider()
            st.header("Per-file settings")
            st.caption("Rename for the legend, set display order (lower = first), "
                       "and choose a color for each file. Panels below reorder to "
                       "match.")
            n_files = len(csv_files)
            upload_idx = {uf.name: i for i, uf in enumerate(csv_files)}
            # Default color per file: upload order mapped through the default
            # color sequence (plasma 7→1 first, then the rest of the palette).
            color_order = _default_color_indices()
            default_color_idx = {
                uf.name: color_order[i % len(color_order)]
                for i, uf in enumerate(csv_files)
            }
            # Read persisted order first so the panels themselves render in order.
            settings_order = sorted(
                csv_files,
                key=lambda u: (st.session_state.get(f"order_{u.name}",
                                                     upload_idx[u.name] + 1), u.name),
            )
            for uf in settings_order:
                i = upload_idx[uf.name]
                with st.expander(uf.name, expanded=n_files <= 4):
                    file_show[uf.name] = st.checkbox(
                        "Show", value=True, key=f"show_{uf.name}",
                        help="Uncheck to hide this file from the plots, averages, "
                             "and summaries.",
                    )
                    file_names[uf.name] = st.text_input(
                        "Legend name", value=uf.name, key=f"name_{uf.name}",
                        help="Used in the combined-averages legend and summaries.",
                    )
                    file_order[uf.name] = st.number_input(
                        "Order", min_value=1, max_value=n_files,
                        value=min(i + 1, n_files), step=1, key=f"order_{uf.name}",
                        help="Combined-plot legend / stack sequence.",
                    )
                    file_colors[uf.name] = _color_select_ui(uf.name, default_color_idx[uf.name])
                    if volume_norm:
                        # Seed from the persistent store so a previously entered
                        # r_eff survives switching Normalize-by away and back
                        # (Streamlit purges the widget's own key when it's hidden).
                        reff_key = f"reff_{uf.name}"
                        if reff_key not in st.session_state:
                            st.session_state[reff_key] = st.session_state.reff_store.get(
                                uf.name, 10.0)
                        reff = st.number_input(
                            "r_eff (nm)", min_value=0.0, step=0.5, key=reff_key,
                            help="Effective spherical radius for this CSV.",
                        )
                        st.session_state.reff_store[uf.name] = reff
                        file_reff[uf.name] = reff
                        file_volumes[uf.name] = (
                            (4.0 / 3.0) * np.pi * reff ** 3 if reff > 0 else None
                        )
                    else:
                        # Keep the last-entered r_eff available for the legend/CSV.
                        file_reff[uf.name] = st.session_state.reff_store.get(uf.name)
                        file_volumes[uf.name] = None

    if not csv_files:
        st.info("Upload one or more spectra CSVs exported by **Get Spectra** to begin.")
        return

    opts = {
        "show_average": show_average,
        "barplot": barplot,
        "method": method,
        "range": rng,
        "baseline": baseline,
        "illum_tiers": illum_tiers,
    }

    st.caption(f"**Processing applied** — {_processing_summary(method, baseline, volume_norm, rng)}")

    averages = []     # (label, grid, mean, color) for the combined plot
    region_rows = []  # (label, areas) for the stacked region summary

    # Honor the user-set display order (stable tie-break on filename).
    ordered = sorted(csv_files, key=lambda u: (file_order.get(u.name, 1), u.name))

    for uf in ordered:
        if not file_show.get(uf.name, True):   # hidden via the per-file toggle
            continue
        display = file_names.get(uf.name) or uf.name
        st.subheader(display)
        raw = uf.getvalue()
        volume = file_volumes.get(uf.name)
        specs, err = _process_csv(raw, method, rng, volume, baseline_tuple)
        if err:
            st.error(f"{display}: {err}")
            continue

        # file_key stays uf.name so exclusion state is stable across renames/reorders.
        base_color = file_colors.get(uf.name, "#1f77b4")
        grid, mean, n_incl = _render_file(uf.name, specs, opts, base_color, volume)
        if grid.size:
            # Combined legend shows the averaged-spectra count (and r_eff when
            # volume-normalizing).
            reff = file_reff.get(uf.name)
            extra = []
            if volume_norm and reff and reff > 0:
                extra.append(f"r_eff {reff:g} nm")
            extra.append(f"n={n_incl}")
            legend_label = f"{display} ({', '.join(extra)})"
            averages.append((legend_label, grid, mean, base_color))
            if barplot:
                region_rows.append((legend_label, _integrate_regions(grid, mean)))

        st.download_button(
            "Download raw CSV", data=raw, file_name=uf.name,
            mime="text/csv", key=f"dl_{uf.name}",
        )
        st.divider()

    # --- Combined averages (tools live underneath the plot) -----------------
    if st.button("Combine all averages into one plot"):
        st.session_state.show_combined = not st.session_state.show_combined

    if st.session_state.show_combined:
        st.subheader("Combined averages")
        if not averages:
            st.info("No averages to combine (every spectrum is excluded).")
        else:
            _render_combined(averages, method, volume_norm, illum_tiers)

    # --- Stacked region composition summary (very bottom) -------------------
    if barplot and region_rows:
        _render_region_stack(region_rows)


def _render_combined(averages, method, volume_norm, illum_tiers=None):
    """Combined averages figure with an interactive tool selector *underneath* it.

    The active tool is read from session_state BEFORE the chart is built, so the
    selector can live below the plot yet still drive it. Tools:
      * Peak ratio — drag two boxes → each file's Area A / Area B.
      * Population estimation — click to drop peaks; each trace is fit with one
        Gaussian per peak and the component areas give the relative steady-state
        populations (shaded under the traces, tabulated below).

    ``illum_tiers`` (the set of included tiers) drives the figure title so it's
    clear which illumination intensities were filtered out.
    """
    key = "__combined__"
    tool = st.session_state.get("combined_tool", TOOL_NONE)
    centers = sorted(st.session_state.pop_centers.setdefault(key, []))
    ranges = st.session_state.ratio_ranges.setdefault(key, [None, None])

    # Fit population Gaussians up front (needed for both the shadings and table).
    pop_results = {}
    if tool == TOOL_POP and centers:
        for label, grid, mean, _color in averages:
            comps = _fit_population_gaussians(grid, mean, centers)
            if comps:
                pop_results[label] = comps

    # --- Base figure ---
    cfig = go.Figure()
    for label, grid, mean, color in averages:
        cfig.add_trace(go.Scatter(
            x=grid, y=mean, mode="lines",
            line=dict(color=color, width=4),
            opacity=0.85,                         # slight transparency so overlaps read
            name=label,
            hovertemplate=f"{label}<br>%{{x:.1f}} nm, %{{y:.3g}}<extra></extra>",
        ))
        for j, comp in enumerate(pop_results.get(label, [])):
            gy = _gaussian(grid, comp["amp"], comp["mu"], comp["sigma"])
            cfig.add_trace(go.Scatter(
                x=grid, y=gy, mode="lines", line=dict(width=0),
                fill="tozeroy", fillcolor=_rgba(color, POP_FILL_ALPHA),
                name=f"{label} peak {j + 1}", showlegend=False, hoverinfo="skip",
            ))

    if tool == TOOL_RATIO:
        for i, rg in enumerate(ranges):
            if rg:
                cfig.add_vrect(
                    x0=rg[0], x1=rg[1], fillcolor=RATIO_FILL[i], line_width=0,
                    annotation_text=RATIO_LABELS[i], annotation_position="top left",
                )
    if tool == TOOL_POP:
        for x in centers:
            ys = [np.interp(x, g, m, left=np.nan, right=np.nan)
                  for _l, g, m, _c in averages]
            ys = [v for v in ys if np.isfinite(v)]
            if ys:
                cfig.add_annotation(
                    x=x, y=max(ys), text="▼", showarrow=False, yshift=16,
                    font=dict(size=16, color="black"),
                )

    # Title reporting which illumination tiers were filtered out.
    if illum_tiers is None:
        title_text = ""
    else:
        excluded = [t for t in ILLUM_TIERS if t not in illum_tiers]
        if not excluded:
            title_text = "All illumination tiers included"
        elif illum_tiers:
            title_text = ("Illumination: " + ", ".join(t for t in ILLUM_TIERS
                          if t in illum_tiers) + f" (excluded: {', '.join(excluded)})")
        else:
            title_text = "All illumination tiers excluded"

    axis_title_font = dict(size=18, color="black", weight="bold")
    axis_tick_font = dict(size=14, color="black", weight="bold")
    cfig.update_layout(
        title=dict(text=title_text, font=dict(size=14, color="black")),
        xaxis=dict(
            title=dict(text="Wavelength (nm)", font=axis_title_font),
            tickfont=axis_tick_font,
        ),
        yaxis=dict(
            title=dict(text=_y_axis_label(method, volume_norm), font=axis_title_font),
            tickfont=axis_tick_font,
        ),
        margin=dict(l=70, r=10, t=40, b=50), height=480,
        legend=dict(title_text="", font=dict(size=16)),
        dragmode="select" if tool in (TOOL_RATIO, TOOL_POP) else "zoom",
    )

    # --- Render chart (interactive when a tool is active) ---
    if tool == TOOL_NONE:
        st.plotly_chart(cfig, use_container_width=True, key="combined")
    elif tool == TOOL_RATIO:
        ptr = st.session_state.ratio_ptr.setdefault(key, 0)
        st.caption(f"Drag a box to set Region {RATIO_LABELS[ptr]} "
                   "(the integrated-area ratio is computed per file).")
        event = st.plotly_chart(
            cfig, use_container_width=True,
            on_select="rerun", selection_mode="box", key="combined_ratio",
        )
        _handle_ratio_selection(event, key, ranges, ptr)
    else:  # TOOL_POP
        st.caption("Drag a small box over a peak to drop it (or type a wavelength "
                   "below) — a ▼ marks the highest trace there. Each trace is fit "
                   "with one Gaussian per peak.")
        event = st.plotly_chart(
            cfig, use_container_width=True,
            on_select="rerun", selection_mode=["points", "box"], key="combined_pop",
        )
        _handle_pop_click(event, key)

    # --- Tool selector UNDERNEATH the plot ---
    st.radio(
        "Combined-plot tool", COMBINED_TOOLS, key="combined_tool", horizontal=True,
        help="Peak ratio: drag two boxes to compare integrated areas. "
             "Population estimation: click peaks to fit Gaussians and tabulate "
             "each trace's relative populations.",
    )

    # --- Tool results below the selector ---
    if tool == TOOL_RATIO:
        _render_ratio_results(averages, key, ranges)
    elif tool == TOOL_POP:
        _render_population_results(averages, key, centers, pop_results)

    # Wide-format CSV of the combined averages (always available).
    frames = []
    for label, grid, mean, _color in averages:
        frames.append(pd.DataFrame({
            f"{label}::Wavelength_nm": grid,
            f"{label}::Avg_Intensity": mean,
        }))
    st.download_button(
        "Download combined averages (CSV)",
        data=pd.concat(frames, axis=1).to_csv(index=False).encode("utf-8"),
        file_name="combined_averages.csv", mime="text/csv",
    )


def _handle_ratio_selection(event, key, ranges, ptr):
    """Apply a new box selection to the next ratio-region slot (A then B)."""
    try:
        boxes = event["selection"]["box"]
    except (TypeError, KeyError, IndexError):
        boxes = []
    if not boxes:
        return
    xr = boxes[0].get("x") or []
    if len(xr) >= 2:
        lo, hi = sorted([float(xr[0]), float(xr[-1])])
        sig = (round(lo, 4), round(hi, 4))
        if sig != st.session_state.ratio_last.get(key):
            st.session_state.ratio_last[key] = sig
            ranges[ptr] = (lo, hi)
            st.session_state.ratio_ptr[key] = ptr ^ 1
            st.rerun()


def _render_ratio_results(averages, key, ranges):
    """Reset control + per-file Area A / Area B ratio table."""
    rc1, rc2 = st.columns([3, 1])
    with rc2:
        if st.button("Reset regions", key="combined_ratio_reset"):
            st.session_state.ratio_ranges[key] = [None, None]
            st.session_state.ratio_ptr[key] = 0
            st.session_state.ratio_last.pop(key, None)
            st.rerun()
    with rc1:
        a_rg, b_rg = ranges
        if a_rg:
            st.caption(f"Region A: {a_rg[0]:.0f}–{a_rg[1]:.0f} nm")
        if b_rg:
            st.caption(f"Region B: {b_rg[0]:.0f}–{b_rg[1]:.0f} nm")

    if ranges[0] and ranges[1]:
        rows = []
        for label, grid, mean, _color in averages:
            aA = _area_in_range(grid, mean, *ranges[0])
            aB = _area_in_range(grid, mean, *ranges[1])
            rows.append({
                "File": label, "Area A": aA, "Area B": aB,
                "Ratio A/B": (aA / aB) if aB else np.nan,
            })
        st.dataframe(
            pd.DataFrame(rows).style.format(
                {"Area A": "{:.3g}", "Area B": "{:.3g}", "Ratio A/B": "{:.3f}"}
            ),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Drag two boxes (Region A and Region B) to compute ratios.")


def _handle_pop_click(event, key):
    """Add a new population peak from a click (point) or a drag (box center).

    A single click in box-drag mode often yields an empty box rather than a point,
    so we accept either: the median x of any selected points, else the center of a
    dragged box. Deduped within 3 nm and guarded by a signature so it fires once.
    """
    try:
        sel = event["selection"]
        pts = sel.get("points") or []
        boxes = sel.get("box") or []
    except (TypeError, KeyError, IndexError):
        pts, boxes = [], []

    x_new = None
    xs = [float(p["x"]) for p in pts if p.get("x") is not None]
    if xs:
        x_new = float(np.median(xs))
    elif boxes:
        xr = boxes[0].get("x") or []
        if len(xr) >= 2:
            x_new = 0.5 * (float(xr[0]) + float(xr[-1]))
    if x_new is None:
        return

    sig = round(x_new, 2)
    if sig == st.session_state.pop_last.get(key):
        return
    st.session_state.pop_last[key] = sig
    lst = st.session_state.pop_centers.setdefault(key, [])
    if not any(abs(x_new - c) < 3.0 for c in lst):  # ignore near-duplicate clicks
        lst.append(x_new)
        st.rerun()


def _render_population_results(averages, key, centers, pop_results):
    """Manual peak entry + reset + the relative-population table (per file/peak)."""
    # Reliable manual entry (works even if plot clicks don't register).
    m1, m2, m3 = st.columns([2, 1, 1])
    with m1:
        peak_x = st.number_input(
            "Add peak at (nm)", min_value=350.0, max_value=1000.0,
            value=650.0, step=5.0, key="pop_add_x",
        )
    with m2:
        if st.button("Add peak", key="pop_add_btn"):
            lst = st.session_state.pop_centers.setdefault(key, [])
            if not any(abs(peak_x - c) < 3.0 for c in lst):
                lst.append(float(peak_x))
                st.rerun()
    with m3:
        if st.button("Reset peaks", key="combined_pop_reset"):
            st.session_state.pop_centers[key] = []
            st.session_state.pop_last.pop(key, None)
            st.rerun()

    if centers:
        st.caption("Peaks at: " + ", ".join(f"{c:.0f} nm" for c in centers))
    else:
        st.caption("Add peaks by dragging a box on the plot or typing a wavelength "
                   "above.")

    if not centers:
        return
    if not pop_results:
        st.info("Gaussian fit did not converge for the current peaks. Try moving "
                "or removing a peak.")
        return

    rows = []
    for label, comps in pop_results.items():
        total = sum(c["area"] for c in comps) or 1.0
        for j, c in enumerate(comps):
            rows.append({
                "File": label, "Peak": j + 1, "Clicked (nm)": c["center"],
                "Fitted μ (nm)": c["mu"], "σ (nm)": c["sigma"], "Area": c["area"],
                "Population %": 100.0 * c["area"] / total,
            })
    df = pd.DataFrame(rows)
    st.caption("Relative steady-state populations — each trace's Gaussian areas as "
               "a percentage of that trace's total fitted area.")
    st.dataframe(
        df.style.format({
            "Clicked (nm)": "{:.0f}", "Fitted μ (nm)": "{:.1f}", "σ (nm)": "{:.1f}",
            "Area": "{:.3g}", "Population %": "{:.1f}",
        }),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "Download population table (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="population_estimates.csv", mime="text/csv",
        key="combined_pop_dl",
    )


def _render_region_stack(region_rows):
    """100%-stacked bar of integrated-region composition, one bar per sample.
    Region checkboxes (at the bottom) exclude bands; remaining bands are
    re-normalized to still sum to 1."""
    st.divider()
    st.subheader("Integrated region composition")
    st.caption("Each sample's band areas stacked and scaled to sum to 1, to "
               "compare spectral composition across samples.")

    # Read the region toggles first (defined at the bottom) so the plot reflects
    # them; only included bands are stacked and re-normalized to sum to 1.
    include = [j for j, (name, *_r) in enumerate(REGIONS)
               if st.session_state.get(f"regionstack_{name}", True)]

    if not include:
        st.info("Select at least one region below to show the composition.")
    else:
        labels = [name for name, _ in region_rows]
        mat = np.array([areas for _, areas in region_rows], dtype=float)[:, include]
        totals = mat.sum(axis=1, keepdims=True)
        fracs = np.divide(mat, totals, out=np.zeros_like(mat), where=totals > 0)

        sfig = go.Figure()
        for col, j in enumerate(include):
            name, _lo, _hi, color = REGIONS[j]
            sfig.add_trace(go.Bar(
                x=labels, y=fracs[:, col], name=name, marker_color=color,
                hovertemplate="%{x} · " + name + ": %{y:.1%}<extra></extra>",
            ))
        sfig.update_layout(
            barmode="stack", xaxis_title="Sample",
            yaxis_title="Fraction of included integrated area",
            yaxis=dict(range=[0, 1]), height=420, legend_title="Region",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(sfig, use_container_width=True, key="region_stack")

    # Region include/exclude checkboxes (bottom).
    st.caption("Include regions:")
    cols = st.columns(len(REGIONS))
    for c, (name, _lo, _hi, _color) in zip(cols, REGIONS):
        c.checkbox(name, value=True, key=f"regionstack_{name}")


if __name__ == "__main__":
    run()
