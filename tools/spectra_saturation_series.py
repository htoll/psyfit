"""Spectra Saturation Series — how emission spectra evolve with excitation current.

A staged workflow that ties together three existing pieces of the toolkit:

  1. **Extract** — one uploader per *sample*; drag in all of that sample's raw
     ``.sif`` acquisitions (every field of view at once — duplicate filenames
     across FOVs are auto-resolved into FOV replicates). With the spectrometer
     calibration ``.pkl`` pair, run the same localization + spectral-dispersion
     extraction as **Get Spectra** (``remove overlapping spectra`` on by default).
     Each surviving particle also carries its **localization-channel brightness**
     (``brightness_integrated`` from the same fit), so brightness and spectrum
     come from one pass.

  2. **Group & process** — pool every particle of the same *(condition, current)*
     across the three fields of view, then curate them like **Process Spectra**:
     click / box to exclude bad traces, baseline-correct, and read the pooled
     average plus the group's localization brightness.

  3. **Plot** — for each condition, overlay the averaged spectrum at every current
     (colored along a current gradient) with normalization options, so you can
     see how increasing excitation reshapes the spectrum. Current is mapped to
     **power density** (W/cm²) via the lab's 60× calibration (coefficients
     editable), enabling brightness- and region-vs-power-density curves.

The suffix of each ``.sif`` (``…_7.sif`` → 7) is mapped to a laser current through
an editable table that can be seeded from / saved to a plain-text map file.

Heavy extraction runs only on an explicit button press and its result is cached in
``st.session_state`` so curating exclusions and toggling normalization never
re-reads a ``.sif``.
"""

import os
import io
import re
import glob
import colorsys

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go

from utils import file_uploader_with_clear

# Reuse the extraction engine from Get Spectra verbatim so the spectra produced
# here are identical to that tool's (same geometry, same overlap filter).
from tools.get_spectra import (
    _process_files,
    just_read_in,
    read_in_calibration,
    _classify_calibration_uploads,
    _build_filtered_coords,
    get_spectrum,
    _illumination_field,
    _interp_tier,
    ILLUM_TIER_LABELS,
)

# Reuse the signal-processing + color helpers from Process Spectra so curation
# and normalization behave the same across the two tools.
from tools.process_spectra import (
    _normalize_spectrum,
    _average_spectra,
    _traces_in_box,
    _shades,
    _width_for_illum,
    WIDTH_MIN, WIDTH_MAX,
    _area_in_range,
    _y_axis_label,
    _processing_summary,
    CROP_NM,
    NORM_NONE, NORM_MAX, NORM_MAX_RANGE, NORM_AREA, NORM_AREA_RANGE, NORM_VOLUME,
    NORM_RANGE_METHODS,
    BASELINE_OFF, BASELINE_MEAN, BASELINE_SPLINE, BASELINE_METHODS,
    BASELINE_MEAN_LO, BASELINE_MEAN_HI,
)

# Register Crameri colormaps (perceptually uniform, colour-blind safe, well
# separated) with matplotlib as ``cmc.*``. Optional — fall back to matplotlib's
# own sequential maps if the package isn't installed.
try:
    import cmcrameri.cm as _cmc  # noqa: F401  (import registers the colormaps)
    _HAVE_CRAMERI = True
except Exception:
    _HAVE_CRAMERI = False

# Sequential colormaps offered for the saturation series, as (label, matplotlib
# registered name). Crameri maps first when available.
CRAMERI_SEQ = [
    ("batlow (Crameri)", "cmc.batlow"), ("lajolla (Crameri)", "cmc.lajolla"),
    ("lipari (Crameri)", "cmc.lipari"), ("navia (Crameri)", "cmc.navia"),
    ("hawaii (Crameri)", "cmc.hawaii"), ("bamako (Crameri)", "cmc.bamako"),
    ("davos (Crameri)", "cmc.davos"), ("oslo (Crameri)", "cmc.oslo"),
]
MPL_SEQ = [("plasma", "plasma"), ("viridis", "viridis"), ("magma", "magma"),
           ("cividis", "cividis"), ("inferno", "inferno")]

# Normalization choices: Process Spectra's set including Volume (per-sample r_eff).
SSS_NORM_METHODS = [NORM_NONE, NORM_MAX, NORM_MAX_RANGE, NORM_AREA, NORM_AREA_RANGE,
                    NORM_VOLUME]

# Trailing "_<n>.sif" is the acquisition suffix that maps to a laser current.
SUFFIX_RE = re.compile(r"_(\d+)\.sif$", re.IGNORECASE)

# Default number of suffixes when nothing has been uploaded yet (this dataset
# takes 15 currents per FOV).
DEFAULT_N_SUFFIX = 15


# --- Current → power-density infrastructure ---------------------------------
# Mirrors tools.SaturationSeries.convertToPowerDensity60x (60× objective, laser
# diode calibrated 2023-06-27) but with the calibration constants surfaced as
# arguments so they can be edited in the UI. power_out (W) = (slope·mA + b)/1000;
# the beam is a Gaussian relayed through the objective, so the illuminated area
# is π·r² with r derived from the post-objective spot sigma.
PD_SLOPE_DEFAULT = 0.29187857       # mW per mA
PD_INTERCEPT_DEFAULT = -17.90535715  # mW at 0 mA
PD_SIGMA_DEFAULT = 0.388             # beam sigma (mm, pre-objective)
PD_RELAY = 3.3 / 150.0               # post-objective demagnification (mm/mm)


def current_to_power_density(current_ma, slope=PD_SLOPE_DEFAULT,
                             intercept=PD_INTERCEPT_DEFAULT,
                             sigma=PD_SIGMA_DEFAULT):
    """Convert laser drive current (mA) to excitation power density (W/cm²).

    ``slope``/``intercept`` are the linear power-vs-current fit (mW); ``sigma`` is
    the beam sigma (mm) before the objective. Returns W/cm². Vectorized over
    ``current_ma``. Sub-threshold currents can yield a negative modeled power —
    the caller decides whether to clip (we clip to 0 for display).
    """
    current_ma = np.asarray(current_ma, dtype=float)
    power_out_w = (slope * current_ma + intercept) / 1000.0
    sigma_post_obj = sigma * PD_RELAY          # mm
    radius_cm = 2.0 * sigma_post_obj / 10.0    # mm → cm (2σ radius)
    area_cm2 = np.pi * radius_cm ** 2
    return power_out_w / area_cm2


# --- Small helpers ----------------------------------------------------------
class _MemFile:
    """Minimal stand-in for a Streamlit ``UploadedFile`` backed by raw bytes.

    The Get Spectra helpers only ever touch ``.name`` and ``.getbuffer()``, so a
    bytes-backed shim lets us re-run extraction from cached bytes without holding
    live upload handles across reruns."""

    def __init__(self, name, data):
        self.name = name
        self._data = bytes(data)

    def getbuffer(self):
        return memoryview(self._data)

    def getvalue(self):
        return self._data


def _suffix_of(name):
    """Trailing acquisition index of a ``.sif`` filename (``…_7.sif`` → 7), or None."""
    m = SUFFIX_RE.search(name)
    return int(m.group(1)) if m else None


def _fov_from_dirname(dirname):
    """FOV number from a subfolder name (``fov3`` / ``li40_fov2`` → the last integer)."""
    nums = re.findall(r"(\d+)", dirname)
    return int(nums[-1]) if nums else None


def _scan_folder(folder):
    """Recursively find ``.sif`` files under ``folder``.

    Returns a sorted list of ``(path, fov)`` where ``fov`` is derived from the
    file's immediate parent-directory name (so ``…/li40_fov2/…_7.sif`` → FOV 2).
    Files sitting directly in ``folder`` get ``fov=None`` (resolved by occurrence
    order at extraction). The path itself is used to read bytes lazily."""
    out = []
    for path in glob.glob(os.path.join(folder, "**", "*.sif"), recursive=True):
        parent = os.path.basename(os.path.dirname(path))
        same = os.path.normpath(os.path.dirname(path)) == os.path.normpath(folder)
        out.append((path, None if same else _fov_from_dirname(parent)))
    return sorted(out)


def _particle_key(p):
    """Stable identifier for one extracted particle (used for exclusion state)."""
    return f"{p['condition']}|{p['current']}|{p['fov']}|{p['file']}|{p['particle_id']}"


def _sequential_color(t, cmap_name="plasma"):
    """Sample a matplotlib sequential colormap at ``t`` ∈ [0, 1] → ``#rrggbb``."""
    import matplotlib
    t = float(np.clip(t, 0.0, 1.0))
    try:
        cmap = matplotlib.colormaps[cmap_name]
    except Exception:
        cmap = matplotlib.colormaps["viridis"]
    return matplotlib.colors.to_hex(cmap(t))


def _categorical_color(i):
    """A distinct constant color per sample index (matplotlib ``tab10``)."""
    import matplotlib
    return matplotlib.colors.to_hex(matplotlib.colormaps["tab10"](i % 10))


# Plotly dash styles cycled per integration region (color stays per sample).
DASH_CYCLE = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

# Fixed plot size giving a ~1:2 (height:width) aspect — rendered with
# use_container_width=False so charts don't stretch to the full (wide) column.
PLOT_W = 780
PLOT_H = 390

# Color choices offered in the region table's dropdown (plotly-valid CSS names).
REGION_COLORS = ["firebrick", "royalblue", "green", "darkorange", "purple",
                 "teal", "deeppink", "goldenrod", "black", "gray", "cyan", "brown"]


def _style_axes(fig):
    """Make every axis title and tick label black + bold (not the default grey)."""
    title_font = dict(color="black", weight="bold", size=15)
    tick_font = dict(color="black", weight="bold", size=13)
    fig.update_xaxes(title_font=title_font, tickfont=tick_font)
    fig.update_yaxes(title_font=title_font, tickfont=tick_font)
    return fig


# --- Current-map editor -----------------------------------------------------
def _parse_map_text(text):
    """Parse a suffix→current text map: ``suffix,current`` (or whitespace) per
    line, ``#`` comments allowed. Returns ``{suffix:int -> current:float}``."""
    mapping = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = re.split(r"[,\t ]+", line)
        if len(parts) < 2:
            continue
        try:
            mapping[int(float(parts[0]))] = float(parts[1])
        except ValueError:
            continue
    return mapping


def _map_to_text(map_df):
    """Serialize the current-map editor DataFrame to a text map file body."""
    lines = ["# suffix,current_mA", "# maps the trailing _<n>.sif index to laser current"]
    for _, r in map_df.sort_values("suffix").iterrows():
        cur = r["current_mA"]
        if pd.notna(cur):
            lines.append(f"{int(r['suffix'])},{cur:g}")
    return "\n".join(lines) + "\n"


def _default_map_df(n):
    """A suffix→current table for suffixes 1..n (current pre-filled with the
    suffix number as an editable placeholder)."""
    return pd.DataFrame({"suffix": list(range(1, n + 1)),
                         "current_mA": [float(s) for s in range(1, n + 1)]})


def _current_map_editor():
    """Stable, fast suffix→current editor kept entirely in session_state.

    The authoritative table lives in ``st.session_state.sss_map_df`` and is passed
    to ``st.data_editor`` unchanged each rerun (then read back). Nothing rebuilds
    the input from a derived dict, so typed values are never dropped. The row set
    is driven by a simple "number of currents" input, not by scanning uploads.
    Returns ``{suffix:int -> current:float}``.
    """
    st.caption("Map each acquisition suffix (`…_<n>.sif`) to its laser current (mA).")

    n = st.number_input("Number of currents (suffixes)", min_value=1, max_value=99,
                        value=DEFAULT_N_SUFFIX, step=1, key="sss_map_n")
    if "sss_map_df" not in st.session_state:
        st.session_state.sss_map_df = _default_map_df(int(n))

    # Reconcile the row count with n, preserving already-entered currents.
    cur_df = st.session_state.sss_map_df
    if len(cur_df) != int(n):
        keep = {int(r["suffix"]): r["current_mA"] for _, r in cur_df.iterrows()}
        st.session_state.sss_map_df = pd.DataFrame({
            "suffix": list(range(1, int(n) + 1)),
            "current_mA": [keep.get(s, float(s)) for s in range(1, int(n) + 1)],
        })

    # Optional: seed from a text map file (overwrites matching suffixes in place).
    with st.expander("Seed from / save text map", expanded=False):
        up = st.file_uploader("Current-map text file", type=["txt", "csv"],
                              key="sss_map_upload",
                              help="Lines of `suffix,current_mA` (`#` comments allowed).")
        if up is not None and st.button("Apply map file", key="sss_apply_map"):
            seed = _parse_map_text(up.getvalue().decode("utf-8", "replace"))
            df = st.session_state.sss_map_df.copy()
            df["current_mA"] = [seed.get(int(s), c) for s, c in
                                zip(df["suffix"], df["current_mA"])]
            st.session_state.sss_map_df = df
            st.success(f"Applied {len(seed)} entries.")
            st.rerun()
        st.download_button(
            "Download current map (txt)",
            data=_map_to_text(st.session_state.sss_map_df).encode("utf-8"),
            file_name="Sat_Series_params.txt", mime="text/plain", key="sss_map_dl",
        )

    edited = st.data_editor(
        st.session_state.sss_map_df, key="sss_map_editor", hide_index=True,
        use_container_width=True, num_rows="fixed",
        column_config={
            "suffix": st.column_config.NumberColumn("Suffix (_n)", disabled=True),
            "current_mA": st.column_config.NumberColumn("Current (mA)", step=1.0),
        },
    )
    st.session_state.sss_map_df = edited   # persist the typed values verbatim
    return {int(r["suffix"]): float(r["current_mA"])
            for _, r in edited.iterrows() if pd.notna(r["current_mA"])}


# --- Sample manager ---------------------------------------------------------
def _sample_manager(mode):
    """One entry per **sample** (condition). ``mode`` is ``"folder"`` or
    ``"upload"``.

    * **folder** — point the sample at a parent folder; every ``.sif`` under it
      (walking ``fov1/fov2/fov3`` subfolders) is used, with FOV taken from the
      subfolder name. This is how the raw acquisitions are already organized on
      disk, so it handles "multiple folders" in one shot. Local runs only.
    * **upload** — drag in the sample's SIFs (FOV resolved by repeated filename).

    Returns a list of ``{"sid", "name", "mode", "folder", "files"}``.
    """
    st.session_state.setdefault("sss_sample_ids", [0, 1, 2])
    ids = st.session_state.sss_sample_ids

    c1, c2 = st.columns(2)
    if c1.button("➕ Add sample", key="sss_add_sample"):
        ids.append((max(ids) + 1) if ids else 0)
        st.rerun()
    if c2.button("➖ Remove last sample", key="sss_rm_sample", disabled=len(ids) <= 1):
        ids.pop()
        st.rerun()

    samples = []
    for sid in ids:
        name = st.session_state.get(f"sss_sname_{sid}", "") or "unnamed"
        with st.expander(f"Sample: {name}", expanded=True):
            folder, files = "", []
            if mode == "folder":
                folder = st.text_input(
                    "Sample folder (contains the fov1/fov2/fov3 subfolders)",
                    key=f"sss_sfolder_{sid}",
                    placeholder=r"G:\Shared drives\…\ITA01_097L_Er02Yb05_Li0satseries",
                    help="Every .sif under this folder is used; FOV comes from the "
                         "subfolder name.",
                ).strip().strip('"')
                # Default the sample name to the folder's basename if left blank.
                default_name = os.path.basename(folder.rstrip("/\\")) if folder else ""
                if folder and os.path.isdir(folder):
                    found = _scan_folder(folder)
                    fovs = sorted({f for _p, f in found if f is not None})
                    st.caption(f"✓ {len(found)} SIF(s) found · "
                               f"FOV subfolders: {fovs or 'none (flat)'}")
                elif folder:
                    st.error("Folder not found on this machine.")
                    default_name = ""
            else:
                default_name = ""
                files = st.file_uploader(
                    "All SIFs for this sample (every FOV — drag them all in; "
                    "identical names across FOVs are fine)",
                    type=["sif"], accept_multiple_files=True,
                    key=f"sss_sfiles_{sid}",
                ) or []

            skey = f"sss_sname_{sid}"
            if skey not in st.session_state and default_name:
                st.session_state[skey] = default_name  # seed before widget creation
            name = st.text_input(
                "Sample / condition name", key=skey,
                placeholder="e.g. Li0 / Li20 / Li40",
                help="Pooled across all its fields of view.",
            )
            samples.append({"sid": sid, "name": (name or "").strip(),
                            "mode": mode, "folder": folder, "files": files})
    return samples


# --- Extraction -------------------------------------------------------------
def _extract_sample(sample, calibration, calib_fits, illum, params):
    """Localize + extract every particle's spectrum for one **sample**.

    All of the sample's FOVs are uploaded into one bucket; duplicate filenames
    (the same ``…_7.sif`` in fov1/fov2/fov3) are the FOV replicates. We uniquify
    each file's name for the temp-file/dict plumbing and derive the FOV number
    from the k-th occurrence of a given original name. Each surviving particle
    carries its localization-channel brightness from the same fit.
    """
    illum_interp, illum_max = illum

    # Resolve the sample's SIFs to a common (name, fov_hint, bytes) list for
    # either input mode. Folder mode reads bytes off disk; FOV comes from the
    # subfolder. Upload mode uses the uploaded bytes; FOV is derived below.
    resolved = []   # (orig_name, fov_hint, bytes)
    if sample["mode"] == "folder":
        folder = sample.get("folder", "")
        if not folder or not os.path.isdir(folder):
            return []
        for path, fov_hint in _scan_folder(folder):
            with open(path, "rb") as fh:
                resolved.append((os.path.basename(path), fov_hint, fh.read()))
    else:
        for f in sample["files"]:
            resolved.append((f.name, None, f.getvalue()))
    if not resolved:
        return []

    # Uniquify by index (collision-proof) and settle each file's FOV: the
    # subfolder-derived hint when present, else the k-th occurrence of the name.
    seen, shims, meta = {}, [], {}
    for idx, (orig, fov_hint, data) in enumerate(resolved):
        if fov_hint is None:
            k = seen.get(orig, 0)
            seen[orig] = k + 1
            fov = k + 1
        else:
            fov = fov_hint
        uniq = f"s{sample['sid']}_i{idx}_{orig}"
        shims.append(_MemFile(uniq, data))
        meta[uniq] = (orig, fov, _suffix_of(orig))

    processed, _ = _process_files(
        shims, threshold=params["threshold"], signal=params["signal"],
    )
    full_frames = just_read_in(shims)

    out, diag = [], []
    for uniq, val in processed.items():
        df = val.get("df")
        frame = full_frames.get(uniq)
        orig, fov, suffix = meta[uniq]
        current = params["current_map"].get(suffix)
        n_loc = 0 if (df is None or df.empty) else len(df)
        d = {"sample": sample["name"] or "unnamed", "file": orig, "fov": fov,
             "suffix": suffix, "current": current, "localized": n_loc,
             "after_filter": 0, "extracted": 0, "spec_failed": 0}
        if df is None or df.empty or frame is None:
            diag.append(d)
            continue
        coords = _build_filtered_coords(
            df, calibration, params["no_dim"], params["remove_overlapping"],
            params["left_edge_cutoff"],
        )
        d["after_filter"] = len(coords)
        for pid, (x, y) in coords.items():
            try:
                wvl, spec, nms_per_pixel = get_spectrum(
                    np.array([x, y]), frame, calibration, calib_fits,
                )
            except Exception:
                d["spec_failed"] += 1
                continue
            inten = spec - float(np.min(spec))   # simple baseline, as Get Spectra
            if params["scale_per_nm"]:
                inten = inten / np.asarray(nms_per_pixel, dtype=float)
            raw_illum, rel_illum, tier = _interp_tier(illum_interp, illum_max, x, y)
            row = df.iloc[pid]
            out.append({
                "condition": sample["name"] or "unnamed",
                "fov": fov,
                "file": orig,
                "particle_id": int(pid),
                "suffix": suffix,
                "current": current,
                "wvl": np.asarray(wvl, dtype=float),
                "inten": np.asarray(inten, dtype=float),
                "brightness_integrated": float(row.get("brightness_integrated", np.nan)),
                "brightness_fit": float(row.get("brightness_fit", np.nan)),
                "rel_illum": float(rel_illum),
                "tier": tier,
            })
            d["extracted"] += 1
        diag.append(d)
    return out, diag


def _run_extraction(samples, cal_file, fit_file, params):
    """Extract every sample with a progress bar; store particles in session_state."""
    calibration, calib_fits = read_in_calibration([cal_file, fit_file])
    illum = _illumination_field(calibration)

    particles, diag = [], []
    prog = st.progress(0.0, text="Extracting spectra…")
    n = max(len(samples), 1)
    for k, sample in enumerate(samples):
        has_input = bool(sample["files"]) or (
            sample["mode"] == "folder" and os.path.isdir(sample.get("folder", "")))
        if not has_input:
            continue
        prog.progress(k / n, text=f"Extracting {sample['name'] or 'unnamed'}…")
        out, sdiag = _extract_sample(sample, calibration, calib_fits, illum, params)
        particles.extend(out)
        diag.extend(sdiag)
    prog.progress(1.0, text="Extraction complete.")
    st.session_state.sss_particles = particles
    st.session_state.sss_diag = diag
    st.session_state.sss_params_summary = params
    # Reset curation state tied to a previous extraction.
    st.session_state.sss_excluded = {}
    st.session_state.sss_group_nonce = {}


def _particles_long_df(particles):
    """Long-form DataFrame of all particles' spectra (Process-Spectra compatible,
    with saturation-series columns appended)."""
    rows = []
    for p in particles:
        for w, inten in zip(p["wvl"], p["inten"]):
            rows.append({
                "File": f"{p['condition']}_FOV{p['fov']}_{p['file']}",
                "Particle_ID": p["particle_id"],
                "Wavelength_nm": w,
                "Intensity": inten,
                "Relative_Illumination": p["rel_illum"],
                "Condition": p["condition"],
                "FOV": p["fov"],
                "Suffix": p["suffix"],
                "Current_mA": p["current"],
                "Brightness_Integrated": p["brightness_integrated"],
            })
    return pd.DataFrame(rows)


# --- Curation (group & process) ---------------------------------------------
def _baseline_controls(prefix):
    """Baseline radio + params, returning the baseline dict Process Spectra uses."""
    method = st.radio(
        "Baseline correction", BASELINE_METHODS, index=0, key=f"{prefix}_bl_method",
        horizontal=True,
        help="Mean: subtract the average in a window. Spline: pybaselines "
             "penalized-spline asymmetric baseline.",
    )
    if method == BASELINE_MEAN:
        c1, c2 = st.columns(2)
        lo = c1.number_input("Mean window min (nm)", value=BASELINE_MEAN_LO,
                             key=f"{prefix}_bl_lo")
        hi = c2.number_input("Mean window max (nm)", value=BASELINE_MEAN_HI,
                             key=f"{prefix}_bl_hi")
        return {"method": "mean", "lo": min(lo, hi), "hi": max(lo, hi)}
    if method == BASELINE_SPLINE:
        lam_log = st.slider("Baseline stiffness (log₁₀ λ)", 0.0, 7.0, 3.0, 0.5,
                            key=f"{prefix}_bl_lam")
        p_asym = st.slider("Baseline asymmetry (p)", 0.001, 0.100, 0.010, 0.001,
                           format="%.3f", key=f"{prefix}_bl_p")
        return {"method": "spline", "lam": 10.0 ** lam_log, "p": p_asym,
                "num_knots": 100, "niter": 10}
    return None


def _norm_controls(prefix):
    """Normalization radio (+ range) shared by the process and plot stages."""
    method = st.radio("Normalize by", SSS_NORM_METHODS, index=0,
                      key=f"{prefix}_norm", horizontal=True)
    rng = (0.0, 0.0)
    if method in NORM_RANGE_METHODS:
        c1, c2 = st.columns(2)
        lo = c1.number_input("Range min (nm)", value=600.0, key=f"{prefix}_rlo")
        hi = c2.number_input("Range max (nm)", value=700.0, key=f"{prefix}_rhi")
        rng = (min(lo, hi), max(lo, hi))
    return method, rng


def _tier_filter_control(prefix):
    """Multiselect of illumination tiers to include. Returns the selected set
    (defaults to all four; an empty selection is treated as 'all')."""
    sel = st.multiselect(
        "Illumination tiers to include", ILLUM_TIER_LABELS,
        default=ILLUM_TIER_LABELS, key=f"{prefix}_tiers",
        help="Relative to the brightest calibration grid point: "
             "Very low <40% · Low <60% · Medium <80% · High ≥80%. "
             "Line thickness also scales with relative illumination.",
    )
    return set(sel) if sel else set(ILLUM_TIER_LABELS)


def _filter_by_tier(particles, tiers):
    """Keep only particles whose illumination tier is in ``tiers``."""
    return [p for p in particles if p.get("tier") in tiers]


def _processed_specs(particles, method, rng, baseline, volume=None):
    """Baseline/normalize + crop every particle's spectrum → list of
    ``(key, wvl, y, rel, brightness)`` for a group. ``volume`` (nm³) is used only
    when ``method`` is Volume (per-sample r_eff)."""
    specs = []
    for p in particles:
        wvl = p["wvl"]
        m = wvl <= CROP_NM
        w, y = _normalize_spectrum(wvl[m], p["inten"][m], method, rng, volume, baseline)
        specs.append((_particle_key(p), w, y, p["rel_illum"],
                      p["brightness_integrated"]))
    return specs


def _render_group(group_key, particles, method, rng, baseline, base_color, volume=None):
    """One (condition, current) group: interactive exclude + pooled average.

    Reuses Process Spectra's one-directional exclusion pattern (click / box only
    ever excludes; a per-figure nonce remounts the chart so a selection fires
    once). Returns ``(grid, mean, sd, included_particles)``.
    """
    st.session_state.setdefault("sss_excluded", {})
    st.session_state.setdefault("sss_group_nonce", {})
    excluded = st.session_state.sss_excluded.setdefault(group_key, set())
    nonce = st.session_state.sss_group_nonce.setdefault(group_key, 0)

    specs = _processed_specs(particles, method, rng, baseline, volume)
    keys = [s[0] for s in specs]
    excluded.intersection_update(keys)
    shades = _shades(base_color, len(specs))

    fig = go.Figure()
    included = []          # (wvl, y)
    incl_bright = []       # brightness of included particles
    box_specs = []         # 4-tuples for _traces_in_box
    has_illum = any(np.isfinite(rel) for (_k, _w, _y, rel, _b) in specs)
    for idx, (key, wvl, y, rel, bright) in enumerate(specs):
        is_excl = key in excluded
        box_specs.append((key, wvl, y, rel))
        # Line thickness ∝ relative illumination (thicker = brighter excitation).
        width = _width_for_illum(rel) if has_illum else 2.5
        illum_txt = f" · illum {rel:.0%}" if np.isfinite(rel) else ""
        fig.add_trace(go.Scatter(
            x=wvl, y=y, mode="lines",
            line=dict(color="lightgrey" if is_excl else shades[idx], width=width),
            opacity=0.4 if is_excl else 1.0,
            name=key.split("|")[-2] + ":" + key.split("|")[-1],
            showlegend=False,
            hovertemplate=(f"{key}{illum_txt}<br>brightness {bright:.3g} pps"
                           "<br>%{x:.1f} nm, %{y:.3g}<extra></extra>"),
        ))
        if not is_excl:
            included.append((wvl, y))
            if np.isfinite(bright):
                incl_bright.append(bright)

    grid, mean, sd = _average_spectra(included)
    if grid.size:
        fig.add_trace(go.Scatter(
            x=grid, y=mean, mode="lines", line=dict(color="black", width=5),
            name="Average", showlegend=True, hoverinfo="skip",
        ))

    if has_illum:  # legend proxies explaining the line-thickness encoding
        for lbl, w in (("Low illumination", WIDTH_MIN), ("High illumination", WIDTH_MAX)):
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines", line=dict(color="grey", width=w),
                name=lbl, showlegend=True, hoverinfo="skip",
            ))

    fig.update_layout(
        xaxis_title="Wavelength (nm)", yaxis_title=_y_axis_label(method, None),
        margin=dict(l=60, r=10, t=10, b=40), width=PLOT_W, height=PLOT_H,
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
        dragmode="select",
    )

    n_excl = len(excluded)
    c_cap, c_btn = st.columns([4, 1])
    with c_cap:
        st.caption(f"{len(specs)} particles · {len(included)} included · "
                   f"{n_excl} excluded — click a trace or drag a box to **exclude**.")
    with c_btn:
        if st.button("Clear", key=f"sss_clear_{group_key}", disabled=not excluded):
            excluded.clear()
            st.session_state.sss_group_nonce[group_key] = nonce + 1
            st.rerun()

    event = st.plotly_chart(
        _style_axes(fig), use_container_width=False, on_select="rerun",
        selection_mode=["points", "box"], key=f"sss_plot_{group_key}_{nonce}",
    )
    try:
        sel = event["selection"]
        boxes = sel.get("box") or []
        pts = sel.get("points") or []
    except (TypeError, KeyError, IndexError):
        boxes, pts = [], []

    targets = set()
    if boxes:
        xr = boxes[0].get("x") or []
        yr = boxes[0].get("y") or []
        if len(xr) >= 2 and len(yr) >= 2:
            targets = set(_traces_in_box(box_specs, xr, yr))
    elif pts:
        for pt in pts:
            cn = pt.get("curve_number")
            if cn is not None and cn < len(keys):
                targets.add(cn)
    newly = [keys[i] for i in targets if keys[i] not in excluded]
    if newly:
        excluded.update(newly)
        st.session_state.sss_group_nonce[group_key] = nonce + 1
        st.rerun()

    # Localization-channel brightness readout for the pooled group.
    if incl_bright:
        arr = np.asarray(incl_bright)
        st.caption(
            f"**Localization brightness** (n={arr.size}): "
            f"mean {arr.mean():.3g} · median {np.median(arr):.3g} pps"
        )

    with st.expander(f"Excluded ({len(excluded)}) — click to re-include"):
        if excluded:
            for k in sorted(excluded):
                if st.button(f"↩ {k}", key=f"sss_reinc_{group_key}_{k}"):
                    excluded.discard(k)
                    st.session_state.sss_group_nonce[group_key] = nonce + 1
                    st.rerun()
        else:
            st.caption("None excluded.")

    return grid, mean, sd, included, incl_bright


# --- Stage renderers --------------------------------------------------------
def _reff_sidebar():
    """Per-sample effective radius r_eff → sphere volume, for Volume normalization.

    Rendered once (in the sidebar) so both the process and plot stages can read
    the volumes from ``st.session_state.sss_volumes`` without a duplicate-key
    clash. Volume = (4/3)·π·r_eff³ (nm³); left blank / 0 disables scaling for
    that sample."""
    st.session_state.setdefault("sss_reff_store", {})
    parts = st.session_state.get("sss_particles")
    conds = sorted({p["condition"] for p in parts}) if parts else []
    volumes = {}
    if not conds:
        st.caption("Extract first, then set each sample's r_eff here to enable "
                   "**Volume (r_eff)** normalization.")
    for c in conds:
        rk = f"sss_reff_{c}"
        if rk not in st.session_state:      # seed before widget creation
            st.session_state[rk] = st.session_state.sss_reff_store.get(c, 10.0)
        r = st.number_input(f"r_eff — {c} (nm)", min_value=0.0, step=0.5, key=rk,
                            help="Effective spherical radius for this sample.")
        st.session_state.sss_reff_store[c] = r
        volumes[c] = (4.0 / 3.0) * np.pi * r ** 3 if r > 0 else None
    st.session_state.sss_volumes = volumes


def _sidebar_settings():
    """Global inputs (input mode, calibration, current map, detection) in the
    sidebar. Returns ``(mode, cal_file, fit_file, params)``."""
    with st.sidebar:
        st.header("Input")
        mode_label = st.radio(
            "Input mode (samples)", ["Local folder", "Upload files"], index=0,
            key="sss_input_mode", horizontal=True,
            help="How the SIF samples are provided. Local folder: point each "
                 "sample at its folder of FOV subfolders (handles multiple folders "
                 "at once; local runs only). Upload: drag in files (works on "
                 "Streamlit Cloud). The calibration is always uploaded.",
        )
        mode = "folder" if mode_label == "Local folder" else "upload"

        st.divider()
        st.header("Calibration")
        cal_file = fit_file = None
        cal_uploads = file_uploader_with_clear(
            "saving_info + fits (.pkl)", key="sss_cal_uploads",
            type=["pkl"], accept_multiple_files=True,
        )
        if cal_uploads:
            cal_file, fit_file, cal_err = _classify_calibration_uploads(cal_uploads)
            if cal_err:
                st.error(cal_err)
            elif cal_file and fit_file:
                st.caption(f"✓ {cal_file.name} · {fit_file.name}")

        st.divider()
        st.header("Suffix → current")
        current_map = _current_map_editor()

        st.divider()
        st.header("Detection")
        threshold = st.number_input("Threshold", min_value=0, value=2, key="sss_thr",
                                    help="Localization stringency (higher = stricter).")
        signal = st.selectbox("Signal", ["UCNP", "dye"], key="sss_signal")
        remove_overlapping = st.checkbox("Remove overlapping spectra", value=True,
                                         key="sss_overlap",
                                         help="On by default for saturation series.")
        with st.expander("Advanced", expanded=False):
            left_edge_cutoff = st.number_input("Left-edge cutoff", value=0,
                                               key="sss_lec")
            scale_per_nm = st.checkbox("Scale intensity per nm", value=False,
                                       key="sss_pernm")
            no_dim = st.number_input("Min brightness (dim cutoff)", value=0.0,
                                     key="sss_nodim")

        st.divider()
        st.header("Volume normalization")
        _reff_sidebar()

    params = {
        "threshold": int(threshold), "signal": signal,
        "left_edge_cutoff": float(left_edge_cutoff),
        "remove_overlapping": bool(remove_overlapping),
        "scale_per_nm": bool(scale_per_nm), "no_dim": float(no_dim),
        "current_map": current_map,
    }
    return mode, cal_file, fit_file, params


def _sample_sif_names(sample):
    """Filenames of the SIFs a sample will contribute, for either input mode."""
    if sample["mode"] == "folder":
        folder = sample.get("folder", "")
        if folder and os.path.isdir(folder):
            return [os.path.basename(p) for p, _f in _scan_folder(folder)]
        return []
    return [f.name for f in sample["files"]]


def _stage_extract(mode, cal_file, fit_file, params):
    st.subheader("1 · Add samples & extract")
    if mode == "folder":
        st.caption(
            "One box per **sample** — point it at the sample's folder (the one "
            "holding its `fov1/fov2/fov3` subfolders). Every SIF underneath is "
            "used and FOVs are pooled automatically. Upload the calibration and "
            "set the suffix→current map in the sidebar."
        )
    else:
        st.caption(
            "One box per **sample** — drag in *all* of that sample's SIFs (every "
            "field of view together). Set the calibration and suffix→current map "
            "in the sidebar. FOVs are pooled automatically."
        )

    samples = _sample_manager(mode)

    sample_names = {s["sid"]: _sample_sif_names(s) for s in samples}
    present = {suf for names in sample_names.values()
               for suf in (_suffix_of(n) for n in names) if suf is not None}
    unmapped = sorted(present - set(params["current_map"]))
    if unmapped:
        st.warning(f"Suffixes present in the SIFs but not mapped to a current "
                   f"(skipped later): {unmapped} — add them in the sidebar map.")

    n_files = sum(len(v) for v in sample_names.values())
    ready = n_files > 0 and cal_file is not None and fit_file is not None
    if not ready:
        need = "point a sample at a folder of SIFs" if mode == "folder" else \
            "add SIFs to at least one sample"
        st.info(f"To enable extraction: {need} **and** upload both calibration "
                f"`.pkl` files (sidebar).")

    if st.button("🔬 Extract spectra", type="primary", disabled=not ready):
        _run_extraction(samples, cal_file, fit_file, params)

    parts = st.session_state.get("sss_particles")
    diag = st.session_state.get("sss_diag")
    if parts:
        df = pd.DataFrame([{
            "Condition": p["condition"], "FOV": p["fov"], "Current (mA)": p["current"],
            "File": p["file"], "Particle": p["particle_id"],
            "Brightness (pps)": p["brightness_integrated"],
        } for p in parts])
        n_groups = df[["Condition", "Current (mA)"]].drop_duplicates().shape[0]
        st.success(f"Extracted {len(parts)} particle spectra · "
                   f"{df['Condition'].nunique()} sample(s) · {n_groups} "
                   f"(sample, current) groups. Head to **Saturation plots** for one "
                   f"figure per sample.")
        with st.expander("Extraction summary table", expanded=False):
            st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download all spectra (long CSV)",
            data=_particles_long_df(parts).to_csv(index=False).encode("utf-8"),
            file_name="saturation_series_spectra.csv", mime="text/csv",
        )
    elif diag is not None:
        st.error("Extraction produced **0 usable spectra**. The per-file breakdown "
                 "below shows where particles dropped out (localized → after "
                 "overlap/edge/brightness filters → spectrum extracted).")

    # Per-file diagnostics: always shown after a run so a zero-result is never
    # silent (localized → after_filter → extracted, plus spectrum-fit failures).
    if diag:
        dd = pd.DataFrame(diag)
        tot = dd[["localized", "after_filter", "extracted", "spec_failed"]].sum()
        st.caption(
            f"Totals — localized {int(tot['localized'])} · after filters "
            f"{int(tot['after_filter'])} · extracted {int(tot['extracted'])} · "
            f"spectrum-fit failures {int(tot['spec_failed'])}."
        )
        if tot["localized"] == 0:
            st.warning("No particles were **localized** in any file — try lowering "
                       "the detection Threshold (sidebar) or check the Signal type.")
        elif tot["after_filter"] == 0:
            st.warning("Particles were localized but **all removed by filters** — "
                       "try turning off *Remove overlapping spectra* or lowering "
                       "*Left-edge cutoff* / *Min brightness* (sidebar → Detection).")
        elif tot["extracted"] == 0:
            st.warning("Particles survived filtering but **every spectrum extraction "
                       "failed** — usually a calibration mismatch (wrong "
                       "saving_info/fits pair).")
        with st.expander("Per-file extraction diagnostics", expanded=not parts):
            st.dataframe(dd, use_container_width=True, hide_index=True)


def _grouped_particles(particles):
    """Group extracted particles by (condition, current), skipping unmapped
    currents. Returns ``{condition: {current: [particles]}}`` (sorted)."""
    groups = {}
    for p in particles:
        if p["current"] is None or not np.isfinite(p["current"]):
            continue
        groups.setdefault(p["condition"], {}).setdefault(float(p["current"]), []).append(p)
    return groups


def _stage_process():
    st.subheader("2 · Group & process")
    particles = st.session_state.get("sss_particles")
    if not particles:
        st.info("Run extraction in **Add samples & extract** first.")
        return

    groups = _grouped_particles(particles)
    if not groups:
        st.warning("No particles have a mapped current — set the suffix→current "
                   "map and re-extract.")
        return

    st.caption("Particles are pooled by **(condition, current)** across all FOVs. "
               "Curate each group like Process Spectra; the pooled average and "
               "localization brightness update live.")

    with st.container(border=True):
        method, rng = _norm_controls("proc")
        baseline = _baseline_controls("proc")
        tiers = _tier_filter_control("proc")
    volume_norm = method == NORM_VOLUME
    volumes = st.session_state.get("sss_volumes", {})
    st.caption("**Processing** — " + _processing_summary(method, baseline, volume_norm, rng)
               + f" · tiers: {', '.join(t for t in ILLUM_TIER_LABELS if t in tiers)}")
    if volume_norm and not any(volumes.values()):
        st.warning("Volume normalization selected but no r_eff set — enter r_eff "
                   "per sample in the sidebar (**Volume normalization**).")

    conditions = sorted(groups)
    condition = st.selectbox("Condition", conditions, key="sss_proc_cond")
    cond_groups = groups[condition]
    vol = volumes.get(condition)
    reff = st.session_state.get("sss_reff_store", {}).get(condition)
    reff_txt = f" · r_eff {reff:g} nm" if (volume_norm and reff) else ""

    for current in sorted(cond_groups):
        pd_val = float(current_to_power_density(current))
        parts = _filter_by_tier(cond_groups[current], tiers)
        n_hidden = len(cond_groups[current]) - len(parts)
        note = f" · {n_hidden} hidden by tier filter" if n_hidden else ""
        st.markdown(f"#### {condition} · {current:g} mA "
                    f"(≈ {max(pd_val, 0.0):.1f} W/cm²){reff_txt} — "
                    f"{len(parts)} particles{note}")
        if not parts:
            st.caption("No particles in the selected illumination tiers.")
            st.divider()
            continue
        group_key = f"{condition}__{current:g}"
        color = _sequential_color(
            (sorted(cond_groups).index(current) + 1) / (len(cond_groups) + 1))
        _render_group(group_key, parts, method, rng, baseline, color, volume=vol)
        st.divider()


def _stage_plot():
    st.subheader("3 · Saturation plots")
    particles = st.session_state.get("sss_particles")
    if not particles:
        st.info("Run extraction in **Add samples & extract** first.")
        return
    groups = _grouped_particles(particles)
    if not groups:
        st.warning("No particles have a mapped current.")
        return

    cmap_opts = (CRAMERI_SEQ if _HAVE_CRAMERI else []) + MPL_SEQ
    with st.container(border=True):
        method, rng = _norm_controls("plot")
        baseline = _baseline_controls("plot")
        tiers = _tier_filter_control("plot")
        cmap_label = st.selectbox(
            "Series colormap", [lbl for lbl, _n in cmap_opts], key="sss_plot_cmap",
            help="Crameri maps (perceptually uniform, colour-blind safe) give the "
                 "most distinct series. Colors are spaced equally by number of "
                 "currents, not by current value.",
        )
        cmap_name = dict(cmap_opts)[cmap_label]
        x_is_power = st.checkbox("X axis / legend as power density (W/cm²)",
                                 value=False, key="sss_plot_power")
        common_y = st.checkbox("Common y-axis across sample plots", value=False,
                               key="sss_plot_commony",
                               help="Use one shared y-range for every sample's "
                                    "evolution plot so they're directly comparable.")
    volume_norm = method == NORM_VOLUME
    volumes = st.session_state.get("sss_volumes", {})
    st.caption("**Processing** — " + _processing_summary(method, baseline, volume_norm, rng)
               + f" · tiers: {', '.join(t for t in ILLUM_TIER_LABELS if t in tiers)}")
    if volume_norm and not any(volumes.values()):
        st.warning("Volume normalization selected but no r_eff set — enter r_eff "
                   "per sample in the sidebar (**Volume normalization**).")

    excluded = st.session_state.get("sss_excluded", {})

    # Build per-(condition, current) averages honoring both the tier filter and
    # the manual exclusions curated in stage 2.
    evolution = {}   # condition -> list of (current, grid, mean, n, brightness)
    for condition, cur_groups in sorted(groups.items()):
        vol = volumes.get(condition)
        rows = []
        for current in sorted(cur_groups):
            gkey = f"{condition}__{current:g}"
            excl = excluded.get(gkey, set())
            specs = _processed_specs(_filter_by_tier(cur_groups[current], tiers),
                                     method, rng, baseline, vol)
            included = [(w, y) for (k, w, y, _r, _b) in specs if k not in excl]
            brights = [b for (k, _w, _y, _r, b) in specs
                       if k not in excl and np.isfinite(b)]
            grid, mean, _sd = _average_spectra(included)
            if grid.size:
                rows.append((current, grid, mean, len(included),
                             float(np.median(brights)) if brights else np.nan))
        if rows:
            evolution[condition] = rows

    if not evolution:
        st.warning("Every spectrum is excluded — nothing to plot.")
        return

    # Equal-distance colors: space by the *index* of each current in the global
    # sorted list (so N series get N evenly spaced colors regardless of how the
    # current values are spaced), consistent across conditions.
    all_currents = sorted({c for rows in evolution.values() for (c, *_x) in rows})
    n_series = len(all_currents)
    cindex = {c: i for i, c in enumerate(all_currents)}

    def _cval(c):
        return 0.5 if n_series <= 1 else cindex[c] / (n_series - 1)

    # Shared y-range across every sample's evolution plot (opt-in).
    common_yr = None
    if common_y:
        allv = np.concatenate([m[np.isfinite(m)] for rows in evolution.values()
                               for (_c, _g, m, _n, _b) in rows if m.size]) \
            if evolution else np.array([])
        if allv.size:
            lo, hi = float(allv.min()), float(allv.max())
            pad = 0.05 * (hi - lo) if hi > lo else (abs(hi) * 0.05 or 1.0)
            common_yr = [lo - pad, hi + pad]

    # --- Per-condition spectral evolution figures ---
    reff_store = st.session_state.get("sss_reff_store", {})
    st.markdown("### Spectra Saturation Series")
    for condition, rows in evolution.items():
        reff = reff_store.get(condition)
        # Surface the r_eff used whenever the spectra are volume-normalized.
        reff_txt = f" · r_eff {reff:g} nm" if (volume_norm and reff) else ""
        reff_tag = f"_reff{reff:g}nm" if (volume_norm and reff) else ""
        fig = go.Figure()
        for (current, grid, mean, n, _bright) in rows:
            xval = max(float(current_to_power_density(current)), 0.0) if x_is_power else current
            label = (f"{xval:.1f} W/cm²" if x_is_power else f"{current:g} mA") + f" (n={n})"
            fig.add_trace(go.Scatter(
                x=grid, y=mean, mode="lines",
                line=dict(color=_sequential_color(_cval(current), cmap_name), width=3),
                name=label,
                hovertemplate=f"{label}<br>%{{x:.1f}} nm, %{{y:.3g}}<extra></extra>",
            ))
        fig.update_layout(
            title=f"{condition} — spectrum vs excitation current{reff_txt}",
            xaxis_title="Wavelength (nm)", yaxis_title=_y_axis_label(method, volume_norm),
            width=PLOT_W, height=PLOT_H, margin=dict(l=70, r=10, t=40, b=50),
            legend=dict(title="Current"),
        )
        if common_yr:
            fig.update_yaxes(range=common_yr)
        st.plotly_chart(_style_axes(fig), use_container_width=False,
                        key=f"sss_evo_{condition}")

        # Per-sample wide CSV: shared wavelength grid + one averaged column per
        # current — exactly the figure above, ready to re-plot elsewhere. The
        # r_eff used is recorded (column + filename) when volume-normalized.
        lo = min(float(g.min()) for (_c, g, _m, _n, _b) in rows)
        hi = max(float(g.max()) for (_c, g, _m, _n, _b) in rows)
        shared = np.linspace(lo, hi, 400)
        wide = {"Wavelength_nm": shared}
        for (current, grid, mean, _n, _b) in rows:
            order = np.argsort(grid)
            wide[f"{current:g}mA"] = np.interp(shared, grid[order], mean[order],
                                               left=np.nan, right=np.nan)
        if volume_norm and reff:
            wide["r_eff_nm"] = reff   # broadcast constant column documenting r_eff
        st.download_button(
            f"Download {condition} averaged spectra (wide CSV){reff_txt}",
            data=pd.DataFrame(wide).to_csv(index=False).encode("utf-8"),
            file_name=f"{condition}_avg_spectra_by_current{reff_tag}.csv",
            mime="text/csv", key=f"sss_wide_{condition}",
        )

    # x-axis shared by the response plots: power density or raw current.
    def _xval(c):
        return max(float(current_to_power_density(c)), 0.0) if x_is_power else float(c)
    x_title = "Power density (W/cm²)" if x_is_power else "Current (mA)"

    cond_color = {cond: _categorical_color(ci)
                  for ci, cond in enumerate(evolution)}

    # --- Multiple integrated regions vs excitation ---
    st.markdown("### Integrated regions vs excitation")
    st.caption("Define one or more integration regions (pick each region's color "
               "from the dropdown). In the plot, **color = region**, **line style "
               "= sample**.")
    if "sss_regions_df" not in st.session_state:
        st.session_state.sss_regions_df = pd.DataFrame({
            "name": ["Green", "Red"], "lo_nm": [505.0, 640.0], "hi_nm": [560.0, 700.0],
            "color": ["green", "firebrick"],
        })
    # Back-compat: add a color column if an older session's df lacks one.
    if "color" not in st.session_state.sss_regions_df.columns:
        st.session_state.sss_regions_df["color"] = REGION_COLORS[0]
    regions_df = st.data_editor(
        st.session_state.sss_regions_df, key="sss_regions_editor", num_rows="dynamic",
        use_container_width=True, hide_index=True,
        column_config={
            "name": st.column_config.TextColumn("Region"),
            "lo_nm": st.column_config.NumberColumn("From (nm)", step=5.0),
            "hi_nm": st.column_config.NumberColumn("To (nm)", step=5.0),
            "color": st.column_config.SelectboxColumn("Color", options=REGION_COLORS),
        },
    )
    st.session_state.sss_regions_df = regions_df
    regions = []   # (name, lo, hi, color)
    for _i, r in regions_df.iterrows():
        try:
            lo, hi = float(r["lo_nm"]), float(r["hi_nm"])
            nm = str(r["name"]) if pd.notna(r["name"]) else f"{lo:.0f}-{hi:.0f}"
        except (TypeError, ValueError):
            continue
        col = r["color"] if ("color" in regions_df.columns and pd.notna(r["color"])) \
            else REGION_COLORS[len(regions) % len(REGION_COLORS)]
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            regions.append((nm, lo, hi, str(col)))

    # Line style per sample (color now encodes region).
    sample_dash = {cond: DASH_CYCLE[i % len(DASH_CYCLE)]
                   for i, cond in enumerate(evolution)}

    export_rows, row_by_key = [], {}
    for condition, rows in evolution.items():
        for (current, grid, mean, n, bright) in rows:
            row = {"Condition": condition, "Current_mA": current,
                   "Power_density_Wcm2": max(float(current_to_power_density(current)), 0.0),
                   "Median_brightness_pps": bright, "N": n}
            for (nm, lo, hi, _col) in regions:
                row[f"Area[{nm}]"] = _area_in_range(grid, mean, lo, hi)
            export_rows.append(row)
            row_by_key[(condition, current)] = row

    if regions:
        area_fig = go.Figure()
        for condition, rows in evolution.items():
            dash = sample_dash[condition]
            xs = np.array([_xval(c) for (c, _g, _m, _n, _b) in rows])
            order = np.argsort(xs)
            for (nm, lo, hi, col) in regions:
                areas = np.array([_area_in_range(g, mn, lo, hi)
                                  for (_c, g, mn, _n, _b) in rows])
                area_fig.add_trace(go.Scatter(
                    x=xs[order], y=areas[order], mode="lines+markers",
                    name=f"{nm} · {condition}", legendgroup=nm,
                    line=dict(color=col, width=3, dash=dash),
                    hovertemplate=(f"{condition} · {nm} ({lo:.0f}–{hi:.0f} nm)"
                                   "<br>%{x:.3g}, area %{y:.3g}<extra></extra>"),
                ))
        area_fig.update_layout(
            title="Integrated region area vs excitation",
            xaxis_title=x_title,
            yaxis_title=f"Integrated area ({_y_axis_label(method, volume_norm)})",
            width=PLOT_W, height=PLOT_H, margin=dict(l=70, r=10, t=40, b=50),
            legend=dict(title="Region · sample"))
        st.plotly_chart(_style_axes(area_fig), use_container_width=False, key="sss_area_pd")
    else:
        st.info("Add at least one valid region (To > From) to plot integrated areas.")

    # --- Range ratio vs excitation ---
    st.markdown("### Range ratio vs excitation")
    st.caption("Ratio of integrated area in range A to range B, per sample "
               "(constant color per sample).")
    rc = st.columns(4)
    a_lo = rc[0].number_input("A from (nm)", value=640.0, step=5.0, key="sss_ratio_alo")
    a_hi = rc[1].number_input("A to (nm)", value=700.0, step=5.0, key="sss_ratio_ahi")
    b_lo = rc[2].number_input("B from (nm)", value=505.0, step=5.0, key="sss_ratio_blo")
    b_hi = rc[3].number_input("B to (nm)", value=560.0, step=5.0, key="sss_ratio_bhi")
    a_lo, a_hi = min(a_lo, a_hi), max(a_lo, a_hi)
    b_lo, b_hi = min(b_lo, b_hi), max(b_lo, b_hi)

    ratio_fig = go.Figure()
    for condition, rows in evolution.items():
        color = cond_color[condition]
        xs = np.array([_xval(c) for (c, _g, _m, _n, _b) in rows])
        order = np.argsort(xs)
        ratios = []
        for i, (current, grid, mean, _n, _b) in enumerate(rows):
            aA = _area_in_range(grid, mean, a_lo, a_hi)
            aB = _area_in_range(grid, mean, b_lo, b_hi)
            ratio = aA / aB if aB else np.nan
            ratios.append(ratio)
            row_by_key[(condition, current)]["Ratio_A_B"] = ratio
        ratio_fig.add_trace(go.Scatter(
            x=xs[order], y=np.asarray(ratios)[order], mode="lines+markers",
            name=condition, line=dict(color=color, width=3),
            hovertemplate=f"{condition}<br>%{{x:.3g}}, A/B %{{y:.3g}}<extra></extra>"))
    ratio_fig.update_layout(
        title=(f"Area ratio A[{a_lo:.0f}–{a_hi:.0f}] / B[{b_lo:.0f}–{b_hi:.0f}] nm "
               "vs excitation"),
        xaxis_title=x_title, yaxis_title="Area ratio A / B",
        width=PLOT_W, height=PLOT_H, margin=dict(l=70, r=10, t=40, b=50),
        legend=dict(title=""))
    st.plotly_chart(_style_axes(ratio_fig), use_container_width=False, key="sss_ratio_pd")

    # --- Localization brightness vs excitation ---
    bright_fig = go.Figure()
    for condition, rows in evolution.items():
        color = cond_color[condition]
        xs = np.array([_xval(c) for (c, _g, _m, _n, _b) in rows])
        order = np.argsort(xs)
        brights = np.array([b for (_c, _g, _m, _n, b) in rows])
        bright_fig.add_trace(go.Scatter(
            x=xs[order], y=brights[order], mode="lines+markers",
            name=condition, line=dict(color=color, width=3)))
    bright_fig.update_layout(
        title="Median localization brightness vs excitation",
        xaxis_title=x_title, yaxis_title="Brightness (pps)", yaxis_type="log",
        width=PLOT_W, height=PLOT_H, margin=dict(l=70, r=10, t=40, b=50),
        legend=dict(title=""))
    st.plotly_chart(_style_axes(bright_fig), use_container_width=False, key="sss_bright_pd")

    if export_rows:
        st.download_button(
            "Download response vs excitation (CSV)",
            data=pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8"),
            file_name="saturation_response.csv", mime="text/csv",
        )

    # --- Compare all samples at one excitation ---
    st.markdown("### Compare samples at one excitation")
    st.caption("Overlay every sample's averaged spectrum at a single chosen "
               "current, to compare samples head-to-head at fixed excitation.")
    pick = st.selectbox(
        "Current (mA)", all_currents, key="sss_cmp_current",
        format_func=lambda c: f"{c:g} mA  (≈ {max(float(current_to_power_density(c)), 0.0):.1f} W/cm²)",
    )
    cmp_fig = go.Figure()
    lo_all, hi_all, wide_cols = [], [], {}
    n_plotted = 0
    for condition, rows in evolution.items():
        match = next((r for r in rows if r[0] == pick), None)
        if match is None:
            continue
        _c, grid, mean, n, _b = match
        cmp_fig.add_trace(go.Scatter(
            x=grid, y=mean, mode="lines", line=dict(color=cond_color[condition], width=3),
            name=f"{condition} (n={n})",
            hovertemplate=f"{condition}<br>%{{x:.1f}} nm, %{{y:.3g}}<extra></extra>",
        ))
        lo_all.append(float(grid.min()))
        hi_all.append(float(grid.max()))
        wide_cols[condition] = (grid, mean)
        n_plotted += 1

    if n_plotted:
        pdv = max(float(current_to_power_density(pick)), 0.0)
        cmp_fig.update_layout(
            title=f"All samples at {pick:g} mA (≈ {pdv:.1f} W/cm²)",
            xaxis_title="Wavelength (nm)", yaxis_title=_y_axis_label(method, volume_norm),
            width=PLOT_W, height=PLOT_H, margin=dict(l=70, r=10, t=40, b=50),
            legend=dict(title="Sample"))
        st.plotly_chart(_style_axes(cmp_fig), use_container_width=False, key="sss_cmp_plot")

        # Wide CSV: shared grid + one averaged column per sample at this current.
        shared = np.linspace(min(lo_all), max(hi_all), 400)
        wide = {"Wavelength_nm": shared}
        for condition, (grid, mean) in wide_cols.items():
            order = np.argsort(grid)
            wide[condition] = np.interp(shared, grid[order], mean[order],
                                        left=np.nan, right=np.nan)
        st.download_button(
            f"Download all samples @ {pick:g} mA (wide CSV)",
            data=pd.DataFrame(wide).to_csv(index=False).encode("utf-8"),
            file_name=f"all_samples_at_{pick:g}mA.csv", mime="text/csv",
            key="sss_cmp_dl",
        )
    else:
        st.info("No sample has an included spectrum at the selected current.")


def _stage_power_density():
    """Editable current→power-density calibration + a preview curve."""
    st.subheader("Power-density calibration (60× objective)")
    st.caption("Maps laser drive current (mA) to excitation power density (W/cm²) "
               "using the lab's 60× calibration. Edit the constants to match your "
               "measured power-vs-current fit; changes apply to the plots.")
    c1, c2, c3 = st.columns(3)
    slope = c1.number_input("Slope (mW/mA)", value=PD_SLOPE_DEFAULT, format="%.6f",
                            key="sss_pd_slope")
    intercept = c2.number_input("Intercept (mW)", value=PD_INTERCEPT_DEFAULT,
                                format="%.5f", key="sss_pd_int")
    sigma = c3.number_input("Beam sigma (mm)", value=PD_SIGMA_DEFAULT, format="%.4f",
                            key="sss_pd_sigma")
    st.session_state.sss_pd_coeffs = (slope, intercept, sigma)

    cur = np.linspace(0, 800, 200)
    pdv = np.clip(current_to_power_density(cur, slope, intercept, sigma), 0, None)
    fig = go.Figure(go.Scatter(x=cur, y=pdv, mode="lines", line=dict(width=3)))
    fig.update_layout(xaxis_title="Current (mA)", yaxis_title="Power density (W/cm²)",
                      width=PLOT_W, height=PLOT_H, margin=dict(l=70, r=10, t=10, b=40))
    st.plotly_chart(_style_axes(fig), use_container_width=False, key="sss_pd_curve")
    st.caption("Note: the linear fit models a threshold current below which the "
               "modeled power is negative (clipped to 0 here).")


# --- App --------------------------------------------------------------------
def run():
    st.caption(
        "Extract emission spectra from a saturation series, pool particles by "
        "condition and laser current across fields of view, curate them, and see "
        "how increasing excitation reshapes each condition's spectrum."
    )
    # Global inputs (input mode, calibration, current map, detection) → sidebar.
    mode, cal_file, fit_file, params = _sidebar_settings()

    tab_extract, tab_process, tab_plot, tab_power = st.tabs(
        ["Add samples & extract", "Group & process", "Saturation plots",
         "Power-density calibration"]
    )
    with tab_extract:
        _stage_extract(mode, cal_file, fit_file, params)
    with tab_process:
        _stage_process()
    with tab_plot:
        _stage_plot()
    with tab_power:
        _stage_power_density()


if __name__ == "__main__":
    run()
