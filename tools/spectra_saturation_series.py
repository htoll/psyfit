"""Spectra Saturation Series — how emission spectra evolve with excitation current.

A staged workflow that ties together three existing pieces of the toolkit:

  1. **Extract** — upload raw ``.sif`` acquisitions grouped into *(condition, FOV)*
     batches plus the spectrometer calibration ``.pkl`` pair, then run the same
     localization + spectral-dispersion extraction as **Get Spectra**
     (``remove overlapping spectra`` on by default). Each surviving particle also
     carries its **localization-channel brightness** (``brightness_integrated``
     from the same fit), so brightness and spectrum come from one pass.

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

import io
import re
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
)

# Reuse the signal-processing + color helpers from Process Spectra so curation
# and normalization behave the same across the two tools.
from tools.process_spectra import (
    _normalize_spectrum,
    _average_spectra,
    _traces_in_box,
    _shades,
    _y_axis_label,
    _processing_summary,
    CROP_NM,
    NORM_NONE, NORM_MAX, NORM_MAX_RANGE, NORM_AREA, NORM_AREA_RANGE,
    NORM_RANGE_METHODS,
    BASELINE_OFF, BASELINE_MEAN, BASELINE_SPLINE, BASELINE_METHODS,
    BASELINE_MEAN_LO, BASELINE_MEAN_HI,
)

# Normalization choices for this tool (Process Spectra's, minus Volume/r_eff —
# there is no per-particle radius here).
SSS_NORM_METHODS = [NORM_NONE, NORM_MAX, NORM_MAX_RANGE, NORM_AREA, NORM_AREA_RANGE]

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


def _current_map_editor(present_suffixes):
    """Editable suffix→current table, seeded from an optional uploaded map file.

    ``present_suffixes`` is the set of suffixes actually seen in the uploads (so
    the table covers exactly what will be extracted). Returns
    ``(map_dict, map_df)``.
    """
    st.caption(
        "Map each acquisition suffix (`…_<n>.sif`) to its laser current (mA). "
        "Optionally seed the table from a text map, edit inline, then re-save."
    )
    up = st.file_uploader(
        "Seed from current-map text file (optional)", type=["txt", "csv"],
        key="sss_map_upload",
        help="Lines of `suffix,current_mA` (`#` comments allowed). "
             "Overwrites matching rows when applied.",
    )
    seed = st.session_state.get("sss_map_seed", {})
    if up is not None and st.button("Apply uploaded map", key="sss_apply_map"):
        seed = _parse_map_text(up.getvalue().decode("utf-8", "replace"))
        st.session_state.sss_map_seed = seed
        st.success(f"Loaded {len(seed)} suffix→current entries.")

    suffixes = sorted(present_suffixes) if present_suffixes else list(
        range(1, DEFAULT_N_SUFFIX + 1))
    # Seed current from (1) any prior edit in session, (2) the uploaded map,
    # (3) the suffix number itself as a visible placeholder to overwrite.
    prior = st.session_state.get("sss_map_edited", {})
    rows = []
    for s in suffixes:
        if s in prior:
            cur = prior[s]
        elif s in seed:
            cur = seed[s]
        else:
            cur = float(s)
        rows.append({"suffix": s, "current_mA": cur})
    map_df = pd.DataFrame(rows)

    edited = st.data_editor(
        map_df, key="sss_map_data_editor", hide_index=True,
        use_container_width=True,
        column_config={
            "suffix": st.column_config.NumberColumn("Suffix (_n)", disabled=True),
            "current_mA": st.column_config.NumberColumn("Current (mA)", step=1.0),
        },
    )
    # Persist edits so they survive reruns / re-uploads.
    st.session_state.sss_map_edited = {
        int(r["suffix"]): (float(r["current_mA"]) if pd.notna(r["current_mA"]) else None)
        for _, r in edited.iterrows()
    }
    st.download_button(
        "Download current map (txt)", data=_map_to_text(edited).encode("utf-8"),
        file_name="Sat_Series_params.txt", mime="text/plain", key="sss_map_dl",
    )
    mapping = {int(r["suffix"]): float(r["current_mA"])
               for _, r in edited.iterrows() if pd.notna(r["current_mA"])}
    return mapping, edited


# --- FOV batch manager ------------------------------------------------------
def _fov_batch_manager():
    """Render the add/remove FOV-batch uploaders; return a list of batch dicts.

    Each batch is ``{"bid", "condition", "fov", "files"}`` where ``files`` is the
    list of uploaded ``.sif`` handles. Identical filenames across FOVs are fine —
    each batch is extracted independently and tagged with its own (condition, FOV).
    """
    st.session_state.setdefault("sss_batch_ids", [0, 1, 2])
    ids = st.session_state.sss_batch_ids

    c1, c2 = st.columns(2)
    if c1.button("➕ Add FOV batch", key="sss_add_batch"):
        ids.append((max(ids) + 1) if ids else 0)
        st.rerun()
    if c2.button("➖ Remove last batch", key="sss_rm_batch", disabled=len(ids) <= 1):
        ids.pop()
        st.rerun()

    batches = []
    for bid in ids:
        with st.expander(
            f"FOV batch {bid} — "
            f"{st.session_state.get(f'sss_cond_{bid}', '') or 'unnamed'} "
            f"/ FOV {st.session_state.get(f'sss_fov_{bid}', 1)}",
            expanded=True,
        ):
            cc1, cc2 = st.columns([2, 1])
            condition = cc1.text_input(
                "Condition", key=f"sss_cond_{bid}",
                placeholder="e.g. Li0 / Li20 / Li40",
                help="Particle condition — pooled across FOVs of the same name.",
            )
            fov = cc2.number_input(
                "FOV #", min_value=1, max_value=99, value=1, step=1,
                key=f"sss_fov_{bid}",
            )
            files = st.file_uploader(
                "SIF files for this (condition, FOV)", type=["sif"],
                accept_multiple_files=True, key=f"sss_files_{bid}",
            )
            batches.append({
                "bid": bid, "condition": (condition or "").strip(),
                "fov": int(fov), "files": files or [],
            })
    return batches


# --- Extraction -------------------------------------------------------------
def _extract_batch(batch, cal_file, fit_file, params):
    """Localize + extract every particle's spectrum for one (condition, FOV) batch.

    Returns a list of particle dicts, each with the spectrum arrays and the
    localization-channel brightness pulled from the same fit. Mirrors the Get
    Spectra per-particle loop (no background subtraction; optional per-nm scale).
    """
    files = batch["files"]
    if not files:
        return []

    processed, _ = _process_files(
        files, threshold=params["threshold"], signal=params["signal"],
    )
    full_frames = just_read_in(files)
    calibration, calib_fits = read_in_calibration([cal_file, fit_file])
    illum_interp, illum_max = _illumination_field(calibration)

    out = []
    for name, val in processed.items():
        df = val.get("df")
        frame = full_frames.get(name)
        if df is None or df.empty or frame is None:
            continue
        suffix = _suffix_of(name)
        current = params["current_map"].get(suffix)
        coords = _build_filtered_coords(
            df, calibration, params["no_dim"], params["remove_overlapping"],
            params["left_edge_cutoff"],
        )
        for pid, (x, y) in coords.items():
            try:
                wvl, spec, nms_per_pixel = get_spectrum(
                    np.array([x, y]), frame, calibration, calib_fits,
                )
            except Exception:
                continue
            inten = spec - float(np.min(spec))   # simple baseline, as Get Spectra
            if params["scale_per_nm"]:
                inten = inten / np.asarray(nms_per_pixel, dtype=float)
            raw_illum, rel_illum, tier = _interp_tier(illum_interp, illum_max, x, y)
            row = df.iloc[pid]
            out.append({
                "condition": batch["condition"] or "unnamed",
                "fov": batch["fov"],
                "file": name,
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
    return out


def _run_extraction(batches, cal_file, fit_file, params):
    """Extract every batch with a progress bar; store particles in session_state."""
    particles = []
    prog = st.progress(0.0, text="Extracting spectra…")
    n = max(len(batches), 1)
    for k, batch in enumerate(batches):
        if not batch["files"]:
            continue
        label = f"{batch['condition'] or 'unnamed'} / FOV {batch['fov']}"
        prog.progress(k / n, text=f"Extracting {label}…")
        particles.extend(_extract_batch(batch, cal_file, fit_file, params))
    prog.progress(1.0, text="Extraction complete.")
    st.session_state.sss_particles = particles
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


def _processed_specs(particles, method, rng, baseline):
    """Baseline/normalize + crop every particle's spectrum → list of
    ``(key, wvl, y, rel, brightness)`` for a group."""
    specs = []
    for p in particles:
        wvl = p["wvl"]
        m = wvl <= CROP_NM
        w, y = _normalize_spectrum(wvl[m], p["inten"][m], method, rng, None, baseline)
        specs.append((_particle_key(p), w, y, p["rel_illum"],
                      p["brightness_integrated"]))
    return specs


def _render_group(group_key, particles, method, rng, baseline, base_color):
    """One (condition, current) group: interactive exclude + pooled average.

    Reuses Process Spectra's one-directional exclusion pattern (click / box only
    ever excludes; a per-figure nonce remounts the chart so a selection fires
    once). Returns ``(grid, mean, sd, included_particles)``.
    """
    st.session_state.setdefault("sss_excluded", {})
    st.session_state.setdefault("sss_group_nonce", {})
    excluded = st.session_state.sss_excluded.setdefault(group_key, set())
    nonce = st.session_state.sss_group_nonce.setdefault(group_key, 0)

    specs = _processed_specs(particles, method, rng, baseline)
    keys = [s[0] for s in specs]
    excluded.intersection_update(keys)
    shades = _shades(base_color, len(specs))

    fig = go.Figure()
    included = []          # (wvl, y)
    incl_bright = []       # brightness of included particles
    box_specs = []         # 4-tuples for _traces_in_box
    for idx, (key, wvl, y, rel, bright) in enumerate(specs):
        is_excl = key in excluded
        box_specs.append((key, wvl, y, rel))
        fig.add_trace(go.Scatter(
            x=wvl, y=y, mode="lines",
            line=dict(color="lightgrey" if is_excl else shades[idx], width=2.5),
            opacity=0.4 if is_excl else 1.0,
            name=key.split("|")[-2] + ":" + key.split("|")[-1],
            showlegend=False,
            hovertemplate=(f"{key}<br>brightness {bright:.3g} pps"
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

    fig.update_layout(
        xaxis_title="Wavelength (nm)", yaxis_title=_y_axis_label(method, None),
        margin=dict(l=60, r=10, t=10, b=40), height=380,
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
        fig, use_container_width=True, on_select="rerun",
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
def _stage_extract():
    st.subheader("1 · Upload & extract")
    st.caption(
        "Add one batch per **(condition, field of view)** — identical filenames "
        "across FOVs are fine. Provide the spectrometer calibration once; it "
        "applies to every batch."
    )

    batches = _fov_batch_manager()

    st.divider()
    st.markdown("**Calibration** (shared across all batches)")
    cal_uploads = file_uploader_with_clear(
        "Calibration files (saving_info + fits .pkl)", key="sss_cal_uploads",
        type=["pkl"], accept_multiple_files=True,
    )
    cal_file = fit_file = None
    cal_err = None
    if cal_uploads:
        cal_file, fit_file, cal_err = _classify_calibration_uploads(cal_uploads)
        if cal_err:
            st.error(cal_err)

    st.divider()
    st.markdown("**Suffix → current map**")
    present = {s for b in batches for s in (_suffix_of(f.name) for f in b["files"])
               if s is not None}
    current_map, _map_df = _current_map_editor(present)

    st.divider()
    st.markdown("**Detection & extraction settings**")
    c1, c2, c3 = st.columns(3)
    threshold = c1.number_input("Threshold", min_value=0, value=2, key="sss_thr",
                                help="Localization stringency (higher = stricter).")
    signal = c2.selectbox("Signal", ["UCNP", "dye"], key="sss_signal")
    left_edge_cutoff = c3.number_input("Left-edge cutoff", value=0, key="sss_lec",
                                       help="Drop spectra running off the detector's "
                                            "left edge.")
    c4, c5, c6 = st.columns(3)
    remove_overlapping = c4.checkbox("Remove overlapping spectra", value=True,
                                     key="sss_overlap",
                                     help="On by default for saturation series.")
    scale_per_nm = c5.checkbox("Scale intensity per nm", value=False, key="sss_pernm")
    no_dim = c6.number_input("Min brightness (dim cutoff)", value=0.0, key="sss_nodim",
                             help="Exclude particles dimmer than this (brightness_fit).")

    params = {
        "threshold": int(threshold), "signal": signal,
        "left_edge_cutoff": float(left_edge_cutoff),
        "remove_overlapping": bool(remove_overlapping),
        "scale_per_nm": bool(scale_per_nm), "no_dim": float(no_dim),
        "current_map": current_map,
    }

    n_files = sum(len(b["files"]) for b in batches)
    ready = n_files > 0 and cal_file is not None and fit_file is not None
    unmapped = sorted(present - set(current_map))
    if unmapped:
        st.warning(f"Suffixes with no current mapped (their particles get NaN "
                   f"current and are skipped in later stages): {unmapped}")

    if st.button("🔬 Extract spectra", type="primary", disabled=not ready):
        _run_extraction(batches, cal_file, fit_file, params)

    if not ready:
        st.info("Upload SIFs into at least one batch and both calibration `.pkl` "
                "files to enable extraction.")

    parts = st.session_state.get("sss_particles")
    if parts:
        df = pd.DataFrame([{
            "Condition": p["condition"], "FOV": p["fov"], "Current (mA)": p["current"],
            "File": p["file"], "Particle": p["particle_id"],
            "Brightness (pps)": p["brightness_integrated"],
        } for p in parts])
        st.success(f"Extracted {len(parts)} particle spectra across "
                   f"{df['Condition'].nunique()} condition(s) and "
                   f"{df[['Condition', 'Current (mA)']].drop_duplicates().shape[0]} "
                   f"(condition, current) groups.")
        with st.expander("Extraction summary table", expanded=False):
            st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download all spectra (CSV — Process-Spectra compatible)",
            data=_particles_long_df(parts).to_csv(index=False).encode("utf-8"),
            file_name="saturation_series_spectra.csv", mime="text/csv",
        )


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
        st.info("Run extraction in **Upload & extract** first.")
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
    st.caption("**Processing** — " + _processing_summary(method, baseline, False, rng))

    conditions = sorted(groups)
    condition = st.selectbox("Condition", conditions, key="sss_proc_cond")
    cond_groups = groups[condition]
    base_color = _sequential_color(0.5)

    for current in sorted(cond_groups):
        pd_val = float(current_to_power_density(current))
        st.markdown(f"#### {condition} · {current:g} mA "
                    f"(≈ {max(pd_val, 0.0):.1f} W/cm²) — {len(cond_groups[current])} particles")
        group_key = f"{condition}__{current:g}"
        color = _sequential_color(
            (sorted(cond_groups).index(current) + 1) / (len(cond_groups) + 1))
        _render_group(group_key, cond_groups[current], method, rng, baseline, color)
        st.divider()


def _stage_plot():
    st.subheader("3 · Saturation plots")
    particles = st.session_state.get("sss_particles")
    if not particles:
        st.info("Run extraction in **Upload & extract** first.")
        return
    groups = _grouped_particles(particles)
    if not groups:
        st.warning("No particles have a mapped current.")
        return

    with st.container(border=True):
        method, rng = _norm_controls("plot")
        baseline = _baseline_controls("plot")
        cmap_name = st.selectbox("Current colormap", ["plasma", "viridis", "magma",
                                 "cividis", "inferno"], key="sss_plot_cmap")
        x_is_power = st.checkbox("X axis / legend as power density (W/cm²)",
                                 value=False, key="sss_plot_power")
    st.caption("**Processing** — " + _processing_summary(method, baseline, False, rng))

    excluded = st.session_state.get("sss_excluded", {})

    # Build per-(condition, current) averages honoring exclusions from stage 2.
    evolution = {}   # condition -> list of (current, grid, mean, n, brightness)
    for condition, cur_groups in sorted(groups.items()):
        rows = []
        for current in sorted(cur_groups):
            gkey = f"{condition}__{current:g}"
            excl = excluded.get(gkey, set())
            specs = _processed_specs(cur_groups[current], method, rng, baseline)
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

    all_currents = sorted({c for rows in evolution.values() for (c, *_x) in rows})
    cmin, cmax = min(all_currents), max(all_currents)

    def _cval(c):
        return 0.5 if cmax == cmin else (c - cmin) / (cmax - cmin)

    # --- Per-condition spectral evolution figures ---
    st.markdown("### Spectral evolution with excitation")
    for condition, rows in evolution.items():
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
            title=f"{condition} — spectrum vs excitation current",
            xaxis_title="Wavelength (nm)", yaxis_title=_y_axis_label(method, None),
            height=460, margin=dict(l=70, r=10, t=40, b=50),
            legend=dict(title="Current"),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"sss_evo_{condition}")

    # --- Region-area & brightness vs power density ---
    st.markdown("### Integrated response vs power density")
    c1, c2 = st.columns(2)
    r_lo = c1.number_input("Region min (nm)", value=630.0, key="sss_reg_lo")
    r_hi = c2.number_input("Region max (nm)", value=700.0, key="sss_reg_hi")
    r_lo, r_hi = min(r_lo, r_hi), max(r_lo, r_hi)

    area_fig = go.Figure()
    bright_fig = go.Figure()
    export_rows = []
    n_cond = len(evolution)
    for ci, (condition, rows) in enumerate(evolution.items()):
        color = _sequential_color((ci + 0.5) / n_cond, "viridis")
        xs, areas, brights = [], [], []
        for (current, grid, mean, n, bright) in rows:
            pdv = max(float(current_to_power_density(current)), 0.0)
            m = (grid >= r_lo) & (grid <= r_hi) & np.isfinite(mean)
            area = float(np.trapz(mean[m], grid[m])) if m.sum() >= 2 else np.nan
            xs.append(pdv)
            areas.append(area)
            brights.append(bright)
            export_rows.append({
                "Condition": condition, "Current_mA": current,
                "Power_density_Wcm2": pdv, f"Area_{r_lo:.0f}_{r_hi:.0f}nm": area,
                "Median_brightness_pps": bright, "N": n,
            })
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        area_fig.add_trace(go.Scatter(
            x=xs, y=np.asarray(areas)[order], mode="lines+markers",
            name=condition, line=dict(color=color, width=3)))
        bright_fig.add_trace(go.Scatter(
            x=xs, y=np.asarray(brights)[order], mode="lines+markers",
            name=condition, line=dict(color=color, width=3)))

    area_fig.update_layout(
        title=f"Integrated {r_lo:.0f}–{r_hi:.0f} nm area vs power density",
        xaxis_title="Power density (W/cm²)",
        yaxis_title=f"Integrated area ({_y_axis_label(method, None)})",
        height=420, margin=dict(l=70, r=10, t=40, b=50))
    bright_fig.update_layout(
        title="Median localization brightness vs power density",
        xaxis_title="Power density (W/cm²)", yaxis_title="Brightness (pps)",
        yaxis_type="log", height=420, margin=dict(l=70, r=10, t=40, b=50))
    st.plotly_chart(area_fig, use_container_width=True, key="sss_area_pd")
    st.plotly_chart(bright_fig, use_container_width=True, key="sss_bright_pd")

    if export_rows:
        st.download_button(
            "Download response vs power density (CSV)",
            data=pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8"),
            file_name="saturation_response_vs_power.csv", mime="text/csv",
        )


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
                      height=340, margin=dict(l=70, r=10, t=10, b=40))
    st.plotly_chart(fig, use_container_width=True, key="sss_pd_curve")
    st.caption("Note: the linear fit models a threshold current below which the "
               "modeled power is negative (clipped to 0 here).")


# --- App --------------------------------------------------------------------
def run():
    st.caption(
        "Extract emission spectra from a saturation series, pool particles by "
        "condition and laser current across fields of view, curate them, and see "
        "how increasing excitation reshapes each condition's spectrum."
    )
    tab_extract, tab_process, tab_plot, tab_power = st.tabs(
        ["Upload & extract", "Group & process", "Saturation plots",
         "Power-density calibration"]
    )
    with tab_extract:
        _stage_extract()
    with tab_process:
        _stage_process()
    with tab_plot:
        _stage_plot()
    with tab_power:
        _stage_power_density()


if __name__ == "__main__":
    run()
