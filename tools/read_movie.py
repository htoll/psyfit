# Streamlit: Movie SIF Processing Tool (no fitting, crash‑resistant)
# ------------------------------------------------------------------
# This page **does not** call `integrate_sif` or perform any PSF fitting.
# It renders a SIF movie quickly by converting each raw frame to cps using
# metadata from `sif_parser.np_open`. This avoids the heavy fitting step
# that was crashing Streamlit.
#
# Controls (requested + stability extras):
# - save_format (default: TIFF)
# - univ_min_max (default: False) → global scaling across frames
# - colorbar (default: True)
# - colormap (default: "gray")
# - title (default: None)
# - frame count (default: True; starts at 1)
# - show exposure/gain (default: True)
# - log scale (optional)
# - grid columns (layout)
# - **Render stride** (safety): render every Nth frame (default 1)
# - **Downsample factor** (safety): integer shrink factor for preview (default 1)
#
# Implementation notes for stability:
# - Avoids loading/keeping large Python lists of images when possible.
# - Uses two‑pass global min/max only when univ_min_max=True (first pass scans
#   frames to compute vmin/vmax without storing full images; second pass renders).
# - Allows stride/downsample to reduce memory/compute pressure on huge movies.
# - Closes Matplotlib figures promptly and forces GC after download.

from __future__ import annotations

import io
import math
import gc
import textwrap
import tempfile
from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

# Rely only on sif_parser.np_open
try:
    import sif_parser  # must provide .np_open(sif, ignore_corrupt=True) -> (frames[T,H,W], meta)
except Exception as e:
    sif_parser = None
    _sif_import_error = e


@dataclass
class Meta:
    exposure_ms: Optional[float]
    gain: Optional[float]


def _np_open_all(path: str):
    if sif_parser is None:
        raise RuntimeError(f"`sif_parser` not importable: {_sif_import_error}")
    frames, meta = sif_parser.np_open(path, ignore_corrupt=True)
    frames = np.asarray(frames)
    if frames.ndim == 2:
        frames = frames[None, ...]
    return frames, meta


def _frame_to_cps(frame2d: np.ndarray, meta: Dict) -> tuple[np.ndarray, Meta]:
    gain_dac = meta.get("GainDAC", 1) or 1
    exposure_time = meta.get("ExposureTime", 1.0) or 1.0  # seconds
    accumulate_cycles = meta.get("AccumulatedCycles", 1) or 1
    cps = frame2d * (5.0 / gain_dac) / exposure_time / accumulate_cycles
    return cps, Meta(exposure_ms=float(exposure_time) * 1000.0, gain=float(gain_dac))


def _maybe_downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    # Simple stride downsample to minimize compute/memory
    return arr[::factor, ::factor]


def _compute_global_minmax(frames: np.ndarray, meta: Dict, stride: int, downsample: int, log_scale: bool) -> Normalize | None:
    # One fast pass to compute min/max of converted cps frames
    vmin = np.inf
    vmax = -np.inf
    T = frames.shape[0]
    for i in range(0, T, max(1, stride)):
        cps, _ = _frame_to_cps(frames[i], meta)
        cps = _maybe_downsample(cps, downsample)
        fi_min = float(np.nanmin(cps))
        fi_max = float(np.nanmax(cps))
        if fi_min < vmin:
            vmin = fi_min
        if fi_max > vmax:
            vmax = fi_max
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return LogNorm() if log_scale else None
    return LogNorm(vmin=vmin, vmax=vmax) if log_scale else Normalize(vmin=vmin, vmax=vmax)


def _title(base_title: Optional[str], i: int, show_frame_count: bool, show_exp_gain: bool, meta: Meta) -> str:
    lines = []
    if base_title:
        lines.append(textwrap.fill(str(base_title), width=28))
    parts = []
    if show_frame_count:
        parts.append(f"Frame {i+1}")
    if show_exp_gain:
        eg = []
        if meta.exposure_ms is not None:
            eg.append(f"Exp {meta.exposure_ms:g} ms")
        if meta.gain is not None:
            eg.append(f"Gain {meta.gain:g}")
        if eg:
            parts.append(" | ".join(eg))
    if parts:
        lines.append(" · ".join(parts))
    return "
".join(lines)


def run():
    st.title("Movie SIF Processor (fast, no fitting)")
    st.caption("Converts each frame to cps using metadata, with crash‑resistant rendering options.")

    with st.sidebar:
        st.header("Controls")
        save_format = st.selectbox("Save format", ["TIFF", "PNG", "SVG", "JPEG"], index=0)
        univ_min_max = st.checkbox("Universal min/max across frames", value=False)
        show_colorbar = st.checkbox("Show colorbar", value=True)
        colormap = st.selectbox("Colormap", ["gray", "magma", "viridis", "plasma", "hot", "hsv", "cividis", "inferno"], index=0)
        base_title = st.text_input("Title (optional)")
        show_frame_count = st.checkbox("Show frame count", value=True)
        show_exp_gain = st.checkbox("Show exposure / gain", value=True)
        log_scale = st.checkbox("Log intensity scaling", value=False)
        max_cols = st.slider("Grid columns", 1, 6, 4)
        stride = st.number_input("Render every Nth frame", min_value=1, value=1, help="Increase to reduce memory/compute (e.g., 2 = 1,3,5,…) ")
        downsample = st.number_input("Downsample factor", min_value=1, value=1, help="Preview shrink factor; larger is faster and safer.")

    uploaded = st.file_uploader("Upload a .sif movie", type=["sif"], accept_multiple_files=False)
    if not uploaded:
        st.info("Upload a .sif file to begin.")
        return

    # Persist to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sif") as tmp:
        tmp.write(uploaded.getbuffer())
        sif_path = tmp.name

    # Load frames (np_open returns all frames; we manage memory after)
    try:
        frames, meta_raw = _np_open_all(sif_path)
    except Exception as e:
        st.error(f"Could not load frames with sif_parser.np_open: {e}")
        return

    T = int(frames.shape[0])
    if T == 0:
        st.error("No frames found in SIF file.")
        return

    # Build normalization
    norm = None
    if univ_min_max:
        norm = _compute_global_minmax(frames, meta_raw, stride=int(stride), downsample=int(downsample), log_scale=bool(log_scale))
    else:
        norm = LogNorm() if log_scale else None

    # Precompute grid and figure
    # Determine how many frames will be rendered with stride
    ids = list(range(0, T, int(stride)))
    n = len(ids)
    n_cols = min(int(max_cols), max(1, n))
    n_rows = int(math.ceil(n / n_cols))

    # Safety: cap huge grids
    MAX_SUBPLOTS = 600  # adjustable safety guard
    if n > MAX_SUBPLOTS:
        st.warning(f"Rendering is capped to {MAX_SUBPLOTS} subplots for stability. Increase stride or downsample to view all frames.")
        ids = ids[:MAX_SUBPLOTS]
        n = len(ids)
        n_rows = int(math.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    last_im = None
    stats_rows = []

    progress = st.progress(0.0, text="Rendering frames…")
    for j, i in enumerate(ids):
        frame = frames[i]
        cps, meta = _frame_to_cps(frame, meta_raw)
        cps = _maybe_downsample(cps, int(downsample))

        ax = axes[j]
        im = ax.imshow(cps, cmap=colormap, origin="lower", norm=norm)
        last_im = im
        ttl = _title(base_title if base_title else None, i, show_frame_count, show_exp_gain, meta)
        if ttl:
            ax.set_title(ttl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        # Stats without retaining full image
        stats_rows.append({
            "frame": i + 1,
            "min": float(np.nanmin(cps)),
            "max": float(np.nanmax(cps)),
            "mean": float(np.nanmean(cps)),
            "exposure_ms": meta.exposure_ms,
            "gain": meta.gain,
        })

        progress.progress((j + 1) / n, text=f"Rendered frame {i+1} ({j+1}/{n})")

    # Turn off unused axes
    for ax in axes[n:]:
        ax.axis("off")

    if show_colorbar and last_im is not None:
        plt.colorbar(last_im, ax=axes[:n], fraction=0.046, pad=0.04, label="Intensity (cps)")

    plt.tight_layout()
    st.pyplot(fig)

    # Downloads
    buf = io.BytesIO()
    fmt = save_format.lower()
    mime = {
        "tiff": "image/tiff",
        "png": "image/png",
        "svg": "image/svg+xml",
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
    }.get(fmt, "application/octet-stream")
    savefmt = "jpg" if fmt == "jpeg" else fmt

    fig.savefig(buf, format=savefmt, bbox_inches="tight", dpi=200)  # moderate DPI for stability
    data = buf.getvalue()
    buf.close()

    today = date.today().strftime("%Y%m%d")
    st.download_button(
        f"Download figure as {save_format.upper()}", data=data, file_name=f"movie_sif_grid_{today}.{savefmt}", mime=mime
    )

    # Per‑frame stats CSV
    df_stats = pd.DataFrame(stats_rows)
    st.dataframe(df_stats, use_container_width=True)
    st.download_button(
        "Download per-frame stats (CSV)", data=df_stats.to_csv(index=False).encode("utf-8"),
        file_name=f"movie_sif_stats_{today}.csv", mime="text/csv"
    )

    # Free memory aggressively
    plt.close(fig)
    del frames
    gc.collect()


if __name__ == "__main__":
    movie_sif_tool()
