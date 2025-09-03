from __future__ import annotations

import io
import os
import re
import math
import textwrap
import tempfile
from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple


import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

from utils import integrate_sif, plot_brightness, plot_histogram

# =============== Utilities ===============
@dataclass
class FrameResult:
    index: int
    image: np.ndarray
    exposure_ms: Optional[float] = None
    gain: Optional[float] = None


def _read_frame_count_with_fallback(path: str) -> int:
    """Try to get frame count using sif_parser; fallback to 1 if unknown."""
    if SifFile is not None:
        try:
            with SifFile(path) as f:
                # Many SIF movies store frames as a time axis
                n_frames = getattr(f, "num_frames", None)
                if n_frames is None:
                    # Some versions expose .frames (list-like)
                    frames = getattr(f, "frames", None)
                    if frames is not None:
                        return len(frames)
                else:
                    return int(n_frames)
        except Exception:
            pass
    # If we can't detect, process as a single frame
    return 1


def _integrate_frame_with_fallback(path: str, frame_index: int) -> Tuple[np.ndarray, Dict]:
    """Adapter to call your project's integrate_sif; fallback to sif_parser mean."""
    # 1) Preferred: your project's `integrate_sif`
    if integrate_sif is not None:
        try:
            out = integrate_sif(path, frame_index=frame_index)  # adjust if needed
            # Expected: (image, meta) or just image
            if isinstance(out, tuple) and len(out) == 2:
                return out  # (img, meta)
            else:
                return out, {}
        except TypeError:
            # Perhaps the function uses a different signature (e.g., frame=...)
            try:
                out = integrate_sif(path, frame=frame_index)
                if isinstance(out, tuple) and len(out) == 2:
                    return out
                return out, {}
            except Exception:
                pass
        except Exception:
            pass

    # 2) Fallback: use sif_parser to read a single frame and return it
    if SifFile is not None:
        try:
            with SifFile(path) as f:
                # Extract 2D array for the requested frame
                data = np.array(f.get_frame(frame_index))  # type: ignore[attr-defined]
                md = {
                    "exposure_ms": getattr(f, "exposure_time", None),
                    "gain": getattr(f, "gain", None),
                }
                return data, md
        except Exception:
            pass

    # 3) Last resort: raise a helpful error
    raise RuntimeError(
        "Could not integrate frame: neither `integrate_sif` nor `sif_parser` succeeded. "
        "Please wire `_integrate_frame_with_fallback` to your project's helpers."
    )


def _compute_normalization(frames: List[np.ndarray], use_log: bool, univ_min_max: bool) -> Optional[Normalize]:
    if not univ_min_max:
        return LogNorm() if use_log else None
    # Compute global min/max across all frames to lock scale
    stacked = np.stack(frames)
    vmin = float(np.nanmin(stacked))
    vmax = float(np.nanmax(stacked))
    # Avoid degenerate ranges
    if vmax <= vmin:
        vmax = vmin + 1e-9
    base = Normalize(vmin=vmin, vmax=vmax)
    return LogNorm(vmin=vmin, vmax=vmax) if use_log else base


def _format_title(base_title: Optional[str], i: int, show_frame_count: bool, show_exp_gain: bool,
                  exposure_ms: Optional[float], gain: Optional[float]) -> str:
    lines = []
    if base_title:
        lines.append(textwrap.fill(str(base_title), width=28))
    parts = []
    if show_frame_count:
        parts.append(f"Frame {i+1}")
    if show_exp_gain:
        eg_bits = []
        if exposure_ms is not None:
            eg_bits.append(f"Exp {exposure_ms:g} ms")
        if gain is not None:
            eg_bits.append(f"Gain {gain}")
        if eg_bits:
            parts.append(" | ".join(eg_bits))
    if parts:
        lines.append(" · ".join(parts))
    return "\n".join(lines) if lines else ""


# =============== Streamlit App ===============

def run():
    st.title("Movie SIF Processor")
    st.caption("Apply `integrate_sif` to each frame, visualize, and export.")

    with st.sidebar:
        st.header("Controls")
        save_format = st.selectbox("Save format", ["TIFF", "PNG", "SVG", "JPEG"], index=0)
        univ_min_max = st.checkbox("Universal min/max across frames", value=False,
                                   help="If enabled, computes global intensity range over all frames.")
        show_colorbar = st.checkbox("Show colorbar", value=True)
        colormap = st.selectbox("Colormap", [
            "gray", "magma", "viridis", "plasma", "hot", "hsv", "cividis", "inferno"
        ], index=0)
        base_title = st.text_input("Title (optional)", value="")
        show_frame_count = st.checkbox("Show frame count", value=True,
                                       help="Displays 1-based frame numbers in subplot titles.")
        show_exp_gain = st.checkbox("Show exposure / gain", value=True)
        use_log_scale = st.checkbox("Log intensity scaling", value=False)
        max_cols = st.slider("Grid columns", min_value=1, max_value=6, value=4)

    uploaded = st.file_uploader("Upload a .sif movie", type=["sif"], accept_multiple_files=False)

    if not uploaded:
        st.info("Upload a .sif file to begin.")
        return

    # Persist the upload to a temp file so external libs can open it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sif") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    try:
        frame_count = _read_frame_count_with_fallback(tmp_path)
    except Exception as e:
        st.error(f"Could not read frame count: {e}")
        return

    # Process every frame
    frames: List[FrameResult] = []
    errors: List[Tuple[int, str]] = []

    progress = st.progress(0.0, text="Integrating frames…")
    for i in range(frame_count):
        try:
            img, meta = _integrate_frame_with_fallback(tmp_path, i)
            exp = meta.get("exposure_ms") if isinstance(meta, dict) else None
            gain = meta.get("gain") if isinstance(meta, dict) else None
            frames.append(FrameResult(index=i, image=np.asarray(img), exposure_ms=exp, gain=gain))
        except Exception as e:
            errors.append((i, str(e)))
        if frame_count > 0:
            progress.progress((i + 1) / frame_count, text=f"Processed frame {i+1}/{frame_count}")
    progress.empty()

    if not frames:
        st.error("No frames could be processed.")
        if errors:
            with st.expander("Errors"):
                for i, msg in errors:
                    st.code(f"Frame {i}: {msg}")
        return

    # Build normalization (possibly global)
    norm = _compute_normalization([f.image for f in frames], use_log=use_log_scale, univ_min_max=univ_min_max)

    # Draw grid
    n = len(frames)
    n_cols = min(max_cols, max(1, n))
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for ax in axes[n:]:
        ax.axis("off")

    ims = []
    for i, fr in enumerate(frames):
        ax = axes[i]
        im = ax.imshow(fr.image, cmap=colormap, origin="lower", norm=norm)
        ims.append(im)
        title = _format_title(base_title if base_title else None, fr.index, show_frame_count, show_exp_gain,
                              fr.exposure_ms, fr.gain)
        if title:
            ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    if show_colorbar and ims:
        # Put colorbar on the last used axis
        plt.colorbar(ims[-1], ax=axes[:n], fraction=0.046, pad=0.04, label="Intensity (a.u.)")

    plt.tight_layout()
    st.pyplot(fig)

    # Downloads: figure and CSV
    buf = io.BytesIO()
    # Map save format to matplotlib format + MIME
    fmt = save_format.lower()
    mime = {
        "tiff": "image/tiff",
        "png": "image/png",
        "svg": "image/svg+xml",
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
    }.get(fmt, "application/octet-stream")

    # Ensure format string is acceptable to `savefig`
    savefmt = "jpg" if fmt == "jpeg" else fmt
    fig.savefig(buf, format=savefmt, bbox_inches="tight", dpi=300)
    data = buf.getvalue()
    buf.close()

    today = date.today().strftime("%Y%m%d")
    dl_name = f"movie_sif_grid_{today}.{savefmt}"
    st.download_button(
        f"Download figure as {save_format.upper()}", data=data, file_name=dl_name, mime=mime
    )

    # Build a per-frame stats CSV
    stats_rows = []
    for fr in frames:
        arr = np.asarray(fr.image)
        stats_rows.append({
            "frame_index": fr.index,
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "exposure_ms": fr.exposure_ms,
            "gain": fr.gain,
        })
    df_stats = pd.DataFrame(stats_rows)
    csv_bytes = df_stats.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download per-frame stats (CSV)", data=csv_bytes,
        file_name=f"movie_sif_stats_{today}.csv", mime="text/csv"
    )

    # Show errors if any
    if errors:
        with st.expander("Frame errors (if any)"):
            for i, msg in errors:
                st.code(f"Frame {i}: {msg}")

