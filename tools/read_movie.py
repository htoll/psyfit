

from __future__ import annotations

import io
import os
import math
import textwrap
import tempfile
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

# --- Your project imports ---
try:
    from utils import integrate_sif  # do not modify this function per requirements
except Exception as e:
    integrate_sif = None
    _import_error = e

# We'll rely only on `sif_parser.np_open` — no SifFile class
try:
    import sif_parser  # must provide .np_open(sif, ignore_corrupt=True) -> (frames[T,H,W], meta)
except Exception as e:
    sif_parser = None
    _sif_import_error = e


# =============== Helpers ===============
@dataclass
class FrameOut:
    index: int
    image_cps: np.ndarray  # image returned by integrate_sif (counts per second)
    exposure_ms: Optional[float]
    gain: Optional[float]


def _load_all_frames_meta(path: str) -> Tuple[np.ndarray, Dict]:
    """Load all frames with sif_parser.np_open. Returns (frames[T,H,W], metadata)."""
    if sif_parser is None:
        raise RuntimeError(
            "Could not import `sif_parser`. Install/ensure it is available."
        )
    frames, meta = sif_parser.np_open(path, ignore_corrupt=True)
    frames = np.asarray(frames)
    if frames.ndim == 2:
        # single frame shaped (H,W) → convert to (1,H,W)
        frames = frames[None, ...]
    return frames, meta


def _run_integrate_for_frame(path: str, frame_idx: int) -> Tuple[np.ndarray, Dict]:
    """Run your unchanged integrate_sif on *one* frame by patching np_open.

    integrate_sif(sif_path) internally does:
        frames, meta = sif_parser.np_open(sif)
        image = frames[0]
    We temporarily replace `sif_parser.np_open` so that `frames[0]` is the
    requested frame, while leaving metadata intact.
    """
    if integrate_sif is None:
        raise RuntimeError(
            f"`integrate_sif` not importable from utils: {_import_error}"
        )
    if sif_parser is None:
        raise RuntimeError(
            f"`sif_parser` not importable: {_sif_import_error}"
        )

    real_np_open = sif_parser.np_open

    def patched_np_open(sif, ignore_corrupt=True):  # signature compatible with your call
        frames, meta = real_np_open(sif, ignore_corrupt=ignore_corrupt)
        frames = np.asarray(frames)
        if frames.ndim == 2:
            # Single frame file; nothing to patch — ensure 3D for consistent indexing
            frames = frames[None, ...]
        # Slice the requested frame and make it appear at index 0
        chosen = frames[frame_idx]
        return np.asarray([chosen]), meta

    # Monkey‑patch, call, then restore
    sif_parser.np_open = patched_np_open
    try:
        df, image_cps = integrate_sif(path)  # use defaults; df is ignored for movie view
        # Extract exposure/gain from metadata by calling the real np_open once
        frames, meta = real_np_open(path, ignore_corrupt=True)
        exposure = meta.get("ExposureTime") if isinstance(meta, dict) else None
        gain_dac = meta.get("GainDAC") if isinstance(meta, dict) else None
        return image_cps, {"exposure_ms": exposure * 1000 if exposure is not None and exposure < 10 else exposure, "gain": gain_dac}
    finally:
        sif_parser.np_open = real_np_open


def _compute_norm(images: List[np.ndarray], log_scale: bool, univ_min_max: bool) -> Optional[Normalize]:
    if not univ_min_max:
        return LogNorm() if log_scale else None
    stack = np.stack(images)
    vmin = float(np.nanmin(stack))
    vmax = float(np.nanmax(stack))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return LogNorm() if log_scale else None
    return LogNorm(vmin=vmin, vmax=vmax) if log_scale else Normalize(vmin=vmin, vmax=vmax)


def _title_lines(base_title: Optional[str], frame_idx: int, show_frame_count: bool,
                 show_exp_gain: bool, exposure_ms: Optional[float], gain: Optional[float]) -> str:
    lines = []
    if base_title:
        lines.append(textwrap.fill(str(base_title), width=28))
    parts = []
    if show_frame_count:
        parts.append(f"Frame {frame_idx + 1}")
    if show_exp_gain:
        eg = []
        if exposure_ms is not None:
            eg.append(f"Exp {exposure_ms:g} ms")
        if gain is not None:
            eg.append(f"Gain {gain}")
        if eg:
            parts.append(" | ".join(eg))
    if parts:
        lines.append(" · ".join(parts))
    return "".join(lines)


# =============== Streamlit UI ===============

def movie_sif_tool():
    st.title("Movie SIF Processor")
    st.caption("Apply your existing `integrate_sif` to every frame, visualize, and export.")

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

    uploaded = st.file_uploader("Upload a .sif movie", type=["sif"], accept_multiple_files=False)
    if not uploaded:
        st.info("Upload a .sif file to begin.")
        return

    # Save uploaded to a temp file for libraries that require a real path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sif") as tmp:
        tmp.write(uploaded.getbuffer())
        sif_path = tmp.name

    # 1) Load all frames once to know T and to compute optional global scale
    try:
        all_frames, meta = _load_all_frames_meta(sif_path)
    except Exception as e:
        st.error(f"Could not load frames with sif_parser.np_open: {e}")
        return

    T = int(all_frames.shape[0])

    # 2) Process each frame via integrate_sif using the monkey‑patch trick
    images: List[np.ndarray] = []
    expos: List[Optional[float]] = []
    gains: List[Optional[float]] = []

    progress = st.progress(0.0, text="Integrating frames…")
    errors: List[Tuple[int, str]] = []
    for i in range(T):
        try:
            img_cps, md = _run_integrate_for_frame(sif_path, i)
            images.append(np.asarray(img_cps))
            expos.append(md.get("exposure_ms"))
            gains.append(md.get("gain"))
        except Exception as e:
            errors.append((i, str(e)))
            images.append(np.zeros_like(all_frames[0], dtype=float))
            expos.append(None)
            gains.append(None)
        progress.progress((i + 1) / T, text=f"Processed frame {i+1}/{T}")
    progress.empty()

    # 3) Plot grid
    norm = _compute_norm(images, log_scale, univ_min_max)

    n = T
    n_cols = min(max_cols, max(1, n))
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    ims = []
    for i in range(n):
        ax = axes[i]
        im = ax.imshow(images[i], cmap=colormap, origin="lower", norm=norm)
        ims.append(im)
        ttl = _title_lines(base_title if base_title else None, i, show_frame_count, show_exp_gain, expos[i], gains[i])
        if ttl:
            ax.set_title(ttl, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Turn off any extra axes
    for ax in axes[n:]:
        ax.axis("off")

    if show_colorbar and ims:
        plt.colorbar(ims[-1], ax=axes[:n], fraction=0.046, pad=0.04, label="Intensity (cps)")

    plt.tight_layout()
    st.pyplot(fig)

    # 4) Downloads
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
    fig.savefig(buf, format=savefmt, bbox_inches="tight", dpi=300)
    data = buf.getvalue()
    buf.close()

    today = date.today().strftime("%Y%m%d")
    dl_name = f"movie_sif_grid_{today}.{savefmt}"

    st.download_button(
        f"Download figure as {save_format.upper()}", data=data, file_name=dl_name, mime=mime
    )

    # Optional: stats CSV
    rows = []
    for i, img in enumerate(images):
        arr = np.asarray(img)
        rows.append({
            "frame": i + 1,
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "exposure_ms": expos[i],
            "gain": gains[i],
        })
    df_stats = pd.DataFrame(rows)
    st.dataframe(df_stats, use_container_width=True)
    st.download_button(
        "Download per-frame stats (CSV)", data=df_stats.to_csv(index=False).encode("utf-8"),
        file_name=f"movie_sif_stats_{today}.csv", mime="text/csv"
    )

    if errors:
        with st.expander("Frame errors (if any)"):
            for i, msg in errors:
                st.code(f"Frame {i+1}: {msg}")


if __name__ == "__main__":
    movie_sif_tool()
