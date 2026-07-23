# movie_brightness.py
# Per-frame PSF brightness analysis for .sif *movies*.
#
# Companion to the "Brightness (WF)" tool (tools/analyze_single_sif.py), but for
# time-series movies rather than single accumulated frames. It:
#   1. builds an accumulation image (first N frames) and finds emitter positions
#      with the same peak-finder used by the widefield tool, letting the user
#      tune threshold / min-distance / signal live before committing;
#   2. fits every detected PSF in every frame (fixed positions) to produce a
#      brightness-vs-time trace per numbered emitter;
#   3. lets the user draw a box and read out summed / background-subtracted
#      intensity within it over time.
#
# Per-frame 2-D Gaussian fitting of many emitters across many frames is heavy,
# so this tool is intentionally LOCAL-ONLY: it refuses to run on Streamlit
# Community Cloud to avoid crashing the shared instance. Set the environment
# variable PSYFIT_ALLOW_MOVIE_BRIGHTNESS=1 to override the guard.

from __future__ import annotations

import gc
import io
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    detect_peaks,
    fit_psf_brightness,
    file_uploader_with_clear,
    detect_steps_kv,
    staircase_from_bounds,
    plot_histogram,
)
from tools import roi as roi_tool

try:
    import sif_parser
except Exception as e:  # pragma: no cover - surfaced in the UI
    sif_parser = None
    _sif_import_error = e


# --- Safeguards -------------------------------------------------------------
MAX_UPLOAD_MB = 1500          # hard block above this (matches .streamlit config)
WARN_UPLOAD_MB = 500          # soft warning above this
WARN_FIT_COUNT = 100_000      # frames * PSFs above which we warn + gate fitting
DEFAULT_ACCUM_FRAMES = 50     # accumulate first N frames for initial peak finding

_PLOTLY_CMAPS = {
    "gray": "Gray", "grey": "Gray", "magma": "Magma", "viridis": "Viridis",
    "plasma": "Plasma", "hot": "Hot", "hsv": "HSV", "cividis": "Cividis",
    "inferno": "Inferno",
}


def _running_on_streamlit_cloud() -> bool:
    """Best-effort detection of Streamlit Community Cloud (vs a local machine).

    The heavy per-frame fitting can OOM/CPU-starve the shared cloud instance, so
    we block there. Community Cloud mounts the repo under ``/mount/src`` and runs
    on Linux; a couple of env markers are checked as backstops. An explicit
    ``PSYFIT_ALLOW_MOVIE_BRIGHTNESS`` env var forces the tool on regardless.
    """
    if os.environ.get("PSYFIT_ALLOW_MOVIE_BRIGHTNESS"):
        return False
    if os.path.isdir("/mount/src"):
        return True
    host = os.environ.get("HOSTNAME", "")
    if host.startswith("streamlit"):
        return True
    if "streamlit.app" in os.environ.get("STREAMLIT_SERVER_BASE_URL_PATH", ""):
        return True
    return False


# --- Frame helpers ----------------------------------------------------------
def _to_cps(frame2d: np.ndarray, meta: dict) -> np.ndarray:
    """Convert a raw frame to counts-per-second (same formula as the WF tool)."""
    gain = meta.get("GainDAC", 1) or 1
    exposure = meta.get("ExposureTime", 1.0) or 1.0
    acc = meta.get("AccumulatedCycles", 1) or 1
    return frame2d * (5.0 / gain) / exposure / acc


def _crop_region(arr: np.ndarray, region: str) -> np.ndarray:
    """Crop a 2-D array to a named quadrant (midpoint split) or return all."""
    region = str(region)
    h, w = arr.shape[-2], arr.shape[-1]
    mh, mw = h // 2, w // 2
    if region == "3":
        return arr[0:mh, 0:mw]
    if region == "4":
        return arr[0:mh, mw:w]
    if region == "1":
        return arr[mh:h, 0:mw]
    if region == "2":
        return arr[mh:h, mw:w]
    return arr  # 'all' / 'Custom' (Custom is handled via an ROI slice)


def _frame_cps_cropped(frames, i, meta, region, roi):
    """Return frame ``i`` as cps, cropped to region (or a Custom ROI slice)."""
    cps = _to_cps(np.asarray(frames[i], dtype=float), meta)
    if roi is not None:
        r0, r1, c0, c1 = roi
        return cps[r0:r1, c0:c1]
    return _crop_region(cps, region)


def _accumulation_cps(frames, n_accum, meta, region, roi):
    """Mean-cps accumulation image of the first ``n_accum`` frames (cropped)."""
    stack = np.asarray(frames[:n_accum], dtype=float).mean(axis=0)
    cps = _to_cps(stack, meta)
    if roi is not None:
        r0, r1, c0, c1 = roi
        return cps[r0:r1, c0:c1]
    return _crop_region(cps, region)


def _frame_interval_s(meta: dict) -> float:
    """Inter-frame time in seconds, best-effort from SIF metadata."""
    for key in ("CycleTime", "KineticCycleTime", "AccumulatedCycleTime", "ExposureTime"):
        val = meta.get(key)
        if val:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return 1.0


def _plot_accum_with_peaks(img, coords, *, cmap="gray", log=False, psf_ids=None,
                           title="Accumulation"):
    """Plotly image of the accumulation with detected peaks (optionally labelled)."""
    img = np.asarray(img, dtype=float)
    disp = np.log10(np.clip(img + 1.0, 1e-9, None)) if log else img
    fig = px.imshow(
        disp, origin="lower", aspect="equal",
        color_continuous_scale=_PLOTLY_CMAPS.get(str(cmap).lower(), "Gray"),
    )
    fig.data[0].hovertemplate = "x=%{x:.0f}px<br>y=%{y:.0f}px<extra></extra>"
    if coords is not None and len(coords):
        ys = coords[:, 0]
        xs = coords[:, 1]
        labels = [str(i) for i in (psf_ids if psf_ids is not None else range(len(xs)))]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=11, color="rgba(0,0,0,0)",
                        line=dict(color="cyan", width=1.5)),
            text=labels, textposition="top center",
            textfont=dict(color="cyan", size=11),
            hovertemplate="PSF %{text}<br>x=%{x:.0f}px<br>y=%{y:.0f}px<extra></extra>",
            showlegend=False,
        ))
    h, w = img.shape
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center",
                   font=dict(color="black", size=15)),
        margin=dict(l=0, r=0, t=34, b=0),
        coloraxis_colorbar=dict(title="pps"),
    )
    fig.update_xaxes(range=[-0.5, w - 0.5], constrain="domain",
                     showgrid=False, zeroline=False)
    fig.update_yaxes(range=[-0.5, h - 0.5], scaleanchor="x", scaleratio=1,
                     showgrid=False, zeroline=False)
    return fig


def _fit_movie(frames, meta, coords, region, roi, *, pix_size_um, sig_threshold,
               sigma_ub):
    """Fit every PSF in ``coords`` across every frame. Returns a long DataFrame.

    Positions are held fixed (from the accumulation) so each emitter keeps its
    numbered identity across frames; a blinking-off frame is recorded with a
    low brightness and ``fit_ok=False`` rather than dropped.
    """
    T = int(frames.shape[0])
    n_psf = len(coords)
    interval = _frame_interval_s(meta)

    rows = []
    progress = st.progress(0.0, text="Fitting frames…")
    for i in range(T):
        frame_cps = _frame_cps_cropped(frames, i, meta, region, roi)
        for pid, (cy, cx) in enumerate(coords):
            fit = fit_psf_brightness(
                frame_cps, cx, cy,
                pix_size_um=pix_size_um, sig_threshold=sig_threshold,
                sigma_ub=sigma_ub, refine=False, strict=False,
            )
            if fit is None:
                rows.append({
                    "psf_id": pid, "frame": i + 1, "time_s": i * interval,
                    "x_pix": float(cx), "y_pix": float(cy),
                    "brightness_integrated": np.nan, "brightness_fit": np.nan,
                    "amp_fit": np.nan, "sigx_fit": np.nan, "sigy_fit": np.nan,
                    "fit_ok": False,
                })
            else:
                rows.append({
                    "psf_id": pid, "frame": i + 1, "time_s": i * interval,
                    "x_pix": float(cx), "y_pix": float(cy),
                    "brightness_integrated": fit["brightness_integrated"],
                    "brightness_fit": fit["brightness_fit"],
                    "amp_fit": fit["amp_fit"],
                    "sigx_fit": fit["sigx_fit"], "sigy_fit": fit["sigy_fit"],
                    "fit_ok": fit["fit_ok"],
                })
        if (i % 2 == 0) or (i == T - 1):
            progress.progress((i + 1) / T, text=f"Fitting frame {i + 1}/{T}…")
    progress.empty()
    df = pd.DataFrame(rows)
    df.attrs["n_psf"] = n_psf
    df.attrs["frame_interval_s"] = interval
    return df


def _box_trace(frames, meta, region, roi, box, *, bg_percentile=20):
    """Per-frame summed and background-subtracted intensity within ``box``.

    ``box`` is ``(row0, row1, col0, col1)`` in the cropped-image coordinate
    frame. Background per frame is the ``bg_percentile`` percentile inside the
    box; integrated = sum - N*background.
    """
    r0, r1, c0, c1 = box
    T = int(frames.shape[0])
    interval = _frame_interval_s(meta)
    rows = []
    for i in range(T):
        cps = _frame_cps_cropped(frames, i, meta, region, roi)
        sub = cps[r0:r1, c0:c1]
        if sub.size == 0:
            summed = integrated = np.nan
        else:
            summed = float(np.sum(sub))
            bg = float(np.percentile(sub, bg_percentile))
            integrated = summed - sub.size * bg
        rows.append({
            "frame": i + 1, "time_s": i * interval,
            "summed_intensity": summed, "integrated_brightness": integrated,
        })
    return pd.DataFrame(rows)


def _psf_bleaching_steps(fit_df, psf_id, *, metric="brightness_integrated",
                         penalty=1.0, min_drop=0.0):
    """Run KV step detection on one PSF's trace.

    Returns ``(trace, steps)`` where ``trace`` is the cleaned (finite) per-frame
    DataFrame with a ``level`` column (the fitted staircase), and ``steps`` is a
    list of downward-step dicts (``frame``, ``time_s``, ``level_before``,
    ``level_after``, ``drop``) with ``drop >= min_drop``.
    """
    sub = fit_df[fit_df["psf_id"] == psf_id].sort_values("frame").copy()
    sub = sub[np.isfinite(sub[metric])]
    if sub.empty:
        return sub, []
    y = sub[metric].to_numpy(dtype=float)
    bounds = detect_steps_kv(y, penalty=penalty)
    levels, raw_steps = staircase_from_bounds(y, bounds)
    sub["level"] = levels

    frames_arr = sub["frame"].to_numpy()
    times_arr = sub["time_s"].to_numpy()
    steps = []
    for s in raw_steps:
        if s["drop"] >= max(min_drop, 0.0) and s["drop"] > 0:  # downward only
            i = s["index"]
            steps.append({
                "psf_id": psf_id,
                "frame": int(frames_arr[i]) if i < len(frames_arr) else int(frames_arr[-1]),
                "time_s": float(times_arr[i]) if i < len(times_arr) else float(times_arr[-1]),
                "level_before": s["level_before"],
                "level_after": s["level_after"],
                "drop": s["drop"],
            })
    return sub, steps


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# --- Main entry point -------------------------------------------------------
def run():
    # 1) Hard gate: local-only.
    if _running_on_streamlit_cloud():
        st.error(
            "🚫 **Movie Brightness is disabled on the cloud deployment.**\n\n"
            "Per-frame PSF fitting across a full movie is compute/memory heavy "
            "and can crash the shared Streamlit instance. Please run PsyFit "
            "locally to use this tool.\n\n"
            "_(Developers: set `PSYFIT_ALLOW_MOVIE_BRIGHTNESS=1` to override.)_"
        )
        return

    if sif_parser is None:
        st.error(f"`sif_parser` is not importable: {_sif_import_error}")
        return

    st.caption(
        "🖥️ Local-only tool. Builds an accumulation image, finds emitters, then "
        "fits each PSF in **every** frame to produce brightness-vs-time traces."
    )

    # 2) Sidebar controls.
    with st.sidebar:
        st.header("Movie Brightness")
        uploaded = file_uploader_with_clear(
            "Upload a .sif movie", key="mb_upload", type=["sif"],
            accept_multiple_files=False,
        )
        st.divider()
        st.subheader("Detection")
        signal = st.selectbox(
            "Signal", options=["UCNP", "dye"],
            help="UCNP = high-SNR peak finder; dye = low-SNR blob detection.",
        )
        threshold = st.number_input(
            "Threshold", min_value=0, value=10,
            help="Higher = more selective peak detection on the accumulation image.",
        )
        min_distance = st.number_input(
            "Minimum distance (px)", min_value=1, value=5,
            help="Minimum separation between detected PSFs.",
        )
        diagram = """Quadrants (midpoint split):
        ┌───┬───┐
        │ 1 │ 2 │
        ├───┼───┤
        │ 3 │ 4 │
        └───┴───┘"""
        region = st.selectbox(
            "Region", options=["all", "1", "2", "3", "4", "Custom"], help=diagram,
        )
        st.divider()
        st.subheader("Fit / display")
        pix_size_um = st.number_input("Pixel size (µm)", min_value=0.01, value=0.1)
        sig_threshold = st.number_input(
            "Max σ accept (µm)", min_value=0.05, value=0.3, step=0.05,
            help="Fits wider than this are flagged fit_ok=False (still recorded).",
        )
        sigma_ub = st.number_input(
            "σ fit bound (µm)", min_value=0.1, value=0.8, step=0.1,
            help="Upper bound on fitted σ during least-squares.",
        )
        cmap = st.selectbox(
            "Colormap", options=["gray", "magma", "viridis", "plasma", "hot",
                                 "inferno", "cividis", "hsv"],
        )
        log_scale = st.checkbox("Log image scaling", value=False)

    if not uploaded:
        st.info("Upload a `.sif` movie in the sidebar to begin.")
        return

    # 3) Upload-size safeguard (before we read anything heavy).
    size_mb = (uploaded.size or 0) / 1e6
    if size_mb > MAX_UPLOAD_MB:
        st.error(
            f"File is {size_mb:.0f} MB, above the {MAX_UPLOAD_MB} MB limit. "
            "Please trim the movie (fewer frames / cropped sensor) before uploading."
        )
        return
    if size_mb > WARN_UPLOAD_MB:
        st.warning(
            f"Large file ({size_mb:.0f} MB). Reading and fitting may take a while "
            "and use a lot of memory."
        )

    # 4) Read frames once per file; keep in session_state (avoids re-reading on
    #    every slider tweak). Fitting results are invalidated when the file
    #    changes.
    file_key = (uploaded.name, uploaded.size)
    if st.session_state.get("mb_file_key") != file_key:
        with st.spinner("Reading movie…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".sif") as tmp:
                tmp.write(uploaded.getbuffer())
                sif_path = tmp.name
            try:
                frames, meta = sif_parser.np_open(sif_path, ignore_corrupt=True)
                frames = np.asarray(frames)
                if frames.ndim == 2:
                    frames = frames[None, ...]
            finally:
                try:
                    os.remove(sif_path)
                except OSError:
                    pass
        st.session_state["mb_file_key"] = file_key
        st.session_state["mb_frames"] = frames
        st.session_state["mb_meta"] = meta
        # Invalidate any prior fit results for the old file.
        for k in ("mb_fit_df", "mb_coords", "mb_fit_context"):
            st.session_state.pop(k, None)

    frames = st.session_state["mb_frames"]
    meta = st.session_state["mb_meta"]
    T, H, W = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    interval_default = _frame_interval_s(meta)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames", f"{T}")
    c2.metric("Resolution", f"{H}×{W}")
    c3.metric("Frame interval", f"{interval_default:.4g} s")
    c4.metric("Duration", f"{(T - 1) * interval_default:.3g} s")

    if T < 2:
        st.warning(
            "This file has a single frame — it looks like an accumulated image, "
            "not a movie. You can still run it, but time-series plots will be "
            "trivial. For single frames use **Brightness (WF)**."
        )

    # 5) Accumulation image + live peak-finding preview.
    st.subheader("1 · Initial peak finding (accumulation)")
    max_accum = T
    default_accum = min(DEFAULT_ACCUM_FRAMES, T)
    n_accum = st.slider(
        "Frames to accumulate for initial guesses", min_value=1,
        max_value=max_accum, value=default_accum,
        help="Peaks are found on the mean of the first N frames "
             "(first 50 or all frames, whichever is smaller, by default).",
    )

    # Custom ROI is drawn on the (uncropped) accumulation image.
    custom_roi = None
    accum_full = _accumulation_cps(frames, n_accum, meta, "all", None)
    if region == "Custom":
        st.caption("Draw a rectangle to restrict the analysis region.")
        custom_roi = roi_tool.draw_roi(
            accum_full, key="mb_custom_roi", cmap=cmap, log=log_scale,
        )
        if custom_roi is None:
            st.info("Draw a rectangle above to set the Custom region, then continue.")
            return

    accum_cps = _accumulation_cps(frames, n_accum, meta, region, custom_roi)

    try:
        coords = detect_peaks(
            accum_cps, threshold=threshold, signal=signal, min_distance=min_distance,
        )
    except Exception as e:
        st.error(f"Peak detection failed: {e}")
        return

    n_peaks = len(coords)
    prev_col, info_col = st.columns([3, 1])
    with prev_col:
        fig_prev = _plot_accum_with_peaks(
            accum_cps, coords, cmap=cmap, log=log_scale,
            title=f"Accumulation of {n_accum} frame(s) — {n_peaks} PSFs",
        )
        st.plotly_chart(fig_prev, use_container_width=True,
                        config={"displaylogo": False})
    with info_col:
        st.metric("PSFs detected", n_peaks)
        st.caption(
            "Tune **Threshold**, **Minimum distance**, and **Signal** in the "
            "sidebar until the overlay looks right, then fit."
        )

    if n_peaks == 0:
        st.warning("No PSFs detected — adjust the detection parameters.")
        return

    # 6) Fit gate + safeguard on the total fit workload.
    est_fits = T * n_peaks
    st.subheader("2 · Fit every PSF in every frame")
    st.caption(
        f"About **{est_fits:,}** PSF fits ({T} frames × {n_peaks} PSFs). "
        f"Fixed positions; dark/blinking frames are recorded with `fit_ok=False`."
    )

    proceed_ok = True
    if est_fits > WARN_FIT_COUNT:
        proceed_ok = st.checkbox(
            f"⚠️ This is a large job ({est_fits:,} fits) and may take several "
            "minutes. I understand — proceed.",
            value=False,
        )

    frame_interval = st.number_input(
        "Frame interval for time axis (s)", min_value=0.0,
        value=float(interval_default), format="%.6f",
        help="Defaults to the SIF CycleTime; override if needed.",
    )

    if st.button("Fit all frames", type="primary", disabled=not proceed_ok):
        df = _fit_movie(
            frames, meta, coords, region, custom_roi,
            pix_size_um=pix_size_um, sig_threshold=sig_threshold, sigma_ub=sigma_ub,
        )
        # Apply the (possibly overridden) frame interval to the time axis.
        if frame_interval != df.attrs.get("frame_interval_s", interval_default):
            df["time_s"] = (df["frame"] - 1) * frame_interval
        st.session_state["mb_fit_df"] = df
        st.session_state["mb_coords"] = coords
        st.session_state["mb_fit_context"] = {
            "region": region, "roi": custom_roi, "n_accum": n_accum,
            "cmap": cmap, "log": log_scale, "frame_interval": frame_interval,
        }
        st.success(f"Fit complete: {len(df):,} rows for {n_peaks} PSFs across {T} frames.")

    # Everything below needs a completed fit.
    fit_df = st.session_state.get("mb_fit_df")
    if fit_df is None or fit_df.empty:
        st.info("Run **Fit all frames** to unlock the time-series analyses below.")
        return

    coords = st.session_state.get("mb_coords", coords)
    ctx = st.session_state.get("mb_fit_context", {})
    ctx_region = ctx.get("region", region)
    ctx_roi = ctx.get("roi", custom_roi)

    st.download_button(
        "Download per-frame fit data (CSV)",
        data=_csv_bytes(fit_df),
        file_name=f"{os.path.splitext(uploaded.name)[0]}_movie_brightness.csv",
        mime="text/csv",
    )

    # 7) Per-PSF brightness vs time.
    st.divider()
    st.subheader("3 · Per-PSF brightness over time")
    all_ids = sorted(fit_df["psf_id"].unique().tolist())
    a1, a2, a3 = st.columns([2, 1, 1])
    with a1:
        default_sel = all_ids[: min(5, len(all_ids))]
        sel_ids = st.multiselect(
            "PSF IDs to plot", options=all_ids, default=default_sel,
            help="IDs match the cyan labels on the accumulation image.",
        )
    with a2:
        metric = st.selectbox(
            "Metric", options=["brightness_integrated", "brightness_fit"],
        )
    with a3:
        x_axis = st.radio("X axis", options=["Time (s)", "Frame #"], index=0)

    only_ok = st.checkbox(
        "Show only good fits (fit_ok)", value=False,
        help="Hide frames flagged fit_ok=False (e.g. blinking-off / bad fits).",
    )

    if sel_ids:
        xcol = "time_s" if x_axis == "Time (s)" else "frame"
        xlabel = "Time (s)" if x_axis == "Time (s)" else "Frame #"
        fig_tr = go.Figure()
        for pid in sel_ids:
            sub = fit_df[fit_df["psf_id"] == pid].sort_values("frame")
            if only_ok:
                sub = sub[sub["fit_ok"]]
            fig_tr.add_trace(go.Scatter(
                x=sub[xcol], y=sub[metric], mode="markers+lines",
                name=f"PSF {pid}", marker=dict(size=5),
                line=dict(width=1),
            ))
        fig_tr.update_layout(
            xaxis_title=xlabel, yaxis_title=f"{metric} (pps)",
            margin=dict(l=0, r=0, t=10, b=0), height=420,
            legend=dict(title="PSF"),
        )
        st.plotly_chart(fig_tr, use_container_width=True,
                        config={"displaylogo": False})

        # Also show the accumulation with IDs so users can map traces → emitters.
        with st.expander("Show PSF ID map"):
            fig_map = _plot_accum_with_peaks(
                _accumulation_cps(frames, ctx.get("n_accum", n_accum), meta,
                                  ctx_region, ctx_roi),
                coords, cmap=ctx.get("cmap", cmap), log=ctx.get("log", log_scale),
                psf_ids=range(len(coords)), title="PSF IDs",
            )
            st.plotly_chart(fig_map, use_container_width=True,
                            config={"displaylogo": False})

        st.download_button(
            "Download selected traces (CSV)",
            data=_csv_bytes(fit_df[fit_df["psf_id"].isin(sel_ids)]),
            file_name="movie_brightness_selected_traces.csv",
            mime="text/csv",
        )
    else:
        st.info("Select one or more PSF IDs to plot their traces.")

    # 8) Bleaching step analysis (single-dye brightness).
    st.divider()
    st.subheader("4 · Bleaching step analysis")
    st.caption(
        "Rank PSFs by initial brightness, take the brightest clusters, detect "
        "photobleaching steps (Kalafut–Visscher), then compile every step-drop "
        "into a single-dye brightness histogram."
    )

    metric_bl = "brightness_integrated"
    b1, b2, b3 = st.columns(3)
    with b1:
        n_init = st.slider(
            "Frames for initial brightness", min_value=1,
            max_value=min(20, T), value=min(5, T),
            help="Per-PSF ranking metric = mean brightness over the first N frames "
                 "(brightest while all dyes are still on).",
        )
    with b2:
        top_pct = st.slider(
            "Top % brightest to analyze", min_value=1, max_value=100, value=10,
        )
    with b3:
        penalty = st.slider(
            "Step sensitivity (BIC penalty)", min_value=0.5, max_value=3.0,
            value=1.0, step=0.1,
            help="Higher = fewer, larger steps (conservative); lower = more steps.",
        )

    # Rank PSFs by mean brightness over the first n_init frames.
    init_series = (
        fit_df[fit_df["frame"] <= n_init]
        .groupby("psf_id")[metric_bl].mean()
        .dropna()
        .sort_values(ascending=False)
    )
    if init_series.empty:
        st.warning("No finite brightness values in the initial frames.")
        return

    init_vals = init_series.to_numpy()
    cutoff = float(np.percentile(init_vals, 100 - top_pct))
    selected_ids = init_series[init_series >= cutoff].index.tolist()

    rank_col, sel_col = st.columns([2, 1])
    with rank_col:
        fig_rank, ax_rank = plt.subplots(figsize=(5, 3))
        ax_rank.hist(init_vals, bins="auto", color="#88CCEE",
                     edgecolor="#5599bb", alpha=0.8)
        ax_rank.axvline(cutoff, color="crimson", ls="--",
                        label=f"top {top_pct}% cutoff")
        ax_rank.set_xlabel("Initial brightness (pps)")
        ax_rank.set_ylabel("PSF count")
        ax_rank.set_title(f"Per-PSF initial brightness (n={len(init_vals)})")
        ax_rank.legend(fontsize=8)
        fig_rank.tight_layout()
        st.pyplot(fig_rank, use_container_width=True)
        plt.close(fig_rank)
    with sel_col:
        st.metric("PSFs selected", f"{len(selected_ids)} / {len(init_vals)}")
        st.caption(f"Cutoff ≥ {cutoff:.3g} pps")
        min_drop = st.number_input(
            "Min step drop (pps)", min_value=0.0, value=0.0,
            help="Ignore down-steps smaller than this (noise floor).",
        )

    if not selected_ids:
        st.info("No PSFs pass the brightness cutoff. Lower the threshold.")
        return

    # Detect steps for each selected PSF.
    all_steps = []
    traces = {}
    for pid in selected_ids:
        trace, steps = _psf_bleaching_steps(
            fit_df, pid, metric=metric_bl, penalty=penalty, min_drop=min_drop,
        )
        traces[pid] = trace
        all_steps.extend(steps)

    n_show = min(len(selected_ids), 12)
    st.markdown(f"**Traces with detected steps** (showing {n_show} of {len(selected_ids)})")
    ncols = min(4, n_show)
    nrows = int(np.ceil(n_show / ncols))
    fig_grid, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows),
                                  squeeze=False)
    axes_flat = axes.flatten()
    for ax, pid in zip(axes_flat, selected_ids[:n_show]):
        tr = traces[pid]
        ax.plot(tr["time_s"], tr[metric_bl], ".", ms=3, color="#4477AA", alpha=0.6)
        ax.plot(tr["time_s"], tr["level"], "-", color="crimson", lw=1.2)
        pid_steps = [s for s in all_steps if s["psf_id"] == pid]
        if pid_steps:
            ax.scatter([s["time_s"] for s in pid_steps],
                       [s["level_after"] for s in pid_steps],
                       marker="v", color="black", s=25, zorder=5)
        ax.set_title(f"PSF {pid} · {len(pid_steps)} steps", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes_flat[n_show:]:
        ax.axis("off")
    fig_grid.supxlabel("Time (s)", fontsize=10)
    fig_grid.supylabel("Brightness (pps)", fontsize=10)
    fig_grid.tight_layout()
    st.pyplot(fig_grid, use_container_width=True)
    grid_buf = io.BytesIO()
    fig_grid.savefig(grid_buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig_grid)
    st.download_button(
        "Download step-trace grid (PNG)", data=grid_buf.getvalue(),
        file_name="bleaching_step_traces.png", mime="image/png",
    )

    # Optional single-PSF detail view.
    detail_pid = st.selectbox("Inspect a single PSF trace", options=selected_ids)
    tr = traces[detail_pid]
    fig_detail = go.Figure()
    fig_detail.add_trace(go.Scatter(
        x=tr["time_s"], y=tr[metric_bl], mode="markers", name="brightness",
        marker=dict(size=5, color="#4477AA"),
    ))
    fig_detail.add_trace(go.Scatter(
        x=tr["time_s"], y=tr["level"], mode="lines", name="KV staircase",
        line=dict(color="crimson", width=2),
    ))
    detail_steps = [s for s in all_steps if s["psf_id"] == detail_pid]
    if detail_steps:
        fig_detail.add_trace(go.Scatter(
            x=[s["time_s"] for s in detail_steps],
            y=[s["level_after"] for s in detail_steps],
            mode="markers", name="step", marker=dict(symbol="triangle-down",
                                                      size=11, color="black"),
        ))
    fig_detail.update_layout(
        xaxis_title="Time (s)", yaxis_title="Brightness (pps)",
        margin=dict(l=0, r=0, t=10, b=0), height=360,
    )
    st.plotly_chart(fig_detail, use_container_width=True,
                    config={"displaylogo": False})

    # Single-dye brightness histogram from all step drops.
    st.markdown("**Single-dye brightness (compiled step drops)**")
    if len(all_steps) < 2:
        st.info(
            "Not enough steps detected to build a histogram. Try lowering the "
            "step sensitivity, the min step drop, or widening the selection."
        )
    else:
        steps_df = pd.DataFrame(all_steps)
        h1, h2 = st.columns([2, 1])
        with h2:
            n_comp = st.number_input(
                "GMM components", min_value=1, max_value=4, value=1,
                help="1 = single-dye Gaussian; increase to resolve multi-dye "
                     "step populations.",
            )
            st.metric("Total steps", len(steps_df))
            st.metric("Median drop", f"{steps_df['drop'].median():.3g}")
        with h1:
            hist_input = pd.DataFrame({"brightness_integrated": steps_df["drop"].values})
            fig_hist, mu, sigma = plot_histogram(
                hist_input, n_components=int(n_comp),
            )
            st.pyplot(fig_hist, use_container_width=True)
            plt.close(fig_hist)
            if mu is not None:
                st.success(f"Single-dye brightness ≈ **{mu:.3g} ± {sigma:.2g} pps** "
                           f"(primary GMM component)")
        st.download_button(
            "Download step drops (CSV)", data=_csv_bytes(steps_df),
            file_name="bleaching_step_drops.csv", mime="text/csv",
        )

    # 9) Box region intensity over time.
    st.divider()
    st.subheader("5 · Region-of-interest intensity over time")
    st.caption(
        "Draw a box on the accumulation image to read out the summed and "
        "background-subtracted intensity inside it, frame by frame."
    )
    accum_for_box = _accumulation_cps(frames, ctx.get("n_accum", n_accum), meta,
                                      ctx_region, ctx_roi)
    box = roi_tool.draw_roi(
        accum_for_box, key="mb_box_roi", cmap=ctx.get("cmap", cmap),
        log=ctx.get("log", log_scale),
    )
    if box is None:
        st.info("Draw a rectangle above to compute its intensity trace.")
        return

    bg_pct = st.slider(
        "Background percentile (for integrated brightness)", min_value=0,
        max_value=50, value=20,
        help="Per-frame background = this percentile within the box; "
             "integrated = sum − N·background.",
    )
    box_df = _box_trace(frames, meta, ctx_region, ctx_roi, box, bg_percentile=bg_pct)

    which = st.radio(
        "Show", options=["Summed intensity", "Integrated brightness", "Both"],
        index=2, horizontal=True,
    )
    box_x_axis = st.radio("X axis ", options=["Time (s)", "Frame #"], index=0,
                          horizontal=True, key="mb_box_xaxis")
    bxcol = "time_s" if box_x_axis == "Time (s)" else "frame"
    bxlabel = "Time (s)" if box_x_axis == "Time (s)" else "Frame #"

    fig_box = go.Figure()
    if which in ("Summed intensity", "Both"):
        fig_box.add_trace(go.Scatter(
            x=box_df[bxcol], y=box_df["summed_intensity"],
            mode="markers+lines", name="Summed", marker=dict(size=5),
            line=dict(width=1),
        ))
    if which in ("Integrated brightness", "Both"):
        fig_box.add_trace(go.Scatter(
            x=box_df[bxcol], y=box_df["integrated_brightness"],
            mode="markers+lines", name="Integrated (bg-sub)", marker=dict(size=5),
            line=dict(width=1),
        ))
    r0, r1, c0, c1 = box
    fig_box.update_layout(
        title=dict(text=f"<b>Box [{r0}:{r1}, {c0}:{c1}] — {(r1-r0)}×{(c1-c0)} px</b>",
                   x=0.5, xanchor="center", font=dict(color="black", size=14)),
        xaxis_title=bxlabel, yaxis_title="Intensity (pps)",
        margin=dict(l=0, r=0, t=34, b=0), height=400,
    )
    st.plotly_chart(fig_box, use_container_width=True,
                    config={"displaylogo": False})

    st.download_button(
        "Download box intensity trace (CSV)",
        data=_csv_bytes(box_df),
        file_name="movie_brightness_box_trace.csv",
        mime="text/csv",
    )

    gc.collect()
