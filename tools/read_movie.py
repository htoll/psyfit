# Streamlit: SIF Movie Exporter (MP4/MOV/TIFF stack) — with Region + robust cleanup
# -------------------------------------------------------------------------------
# Fast, no‑fitting pipeline. Converts each SIF frame to cps and encodes to a
# video (MP4/MOV) or multipage TIFF. Shows a playable preview (MP4 or GIF
# fallback) and a download button. Includes region cropping and defensive
# cleanup to prevent UnboundLocalError.

from __future__ import annotations

import io
import os
import gc
import math
import tempfile
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# Lightweight colormap support
from matplotlib import cm as mpl_cm
from matplotlib.colors import Normalize, LogNorm

# I/O backends for video/TIFF
try:
    import imageio
except Exception as e:
    imageio = None
    _imageio_err = e

# Prefer imageio-ffmpeg for MP4/MOV
try:
    import imageio_ffmpeg  # noqa: F401
    _has_ffmpeg = True
except Exception:
    _has_ffmpeg = False

try:
    import tifffile
except Exception as e:
    tifffile = None
    _tif_err = e

# SIF reader
try:
    import sif_parser  # must provide np_open(path, ignore_corrupt=True) -> (frames[T,H,W], meta)
except Exception as e:
    sif_parser = None
    _sif_import_error = e


@dataclass
class Meta:
    exposure_ms: Optional[float]
    gain: Optional[float]


def _np_open_all(path: str) -> Tuple[np.ndarray, Dict]:
    if sif_parser is None:
        raise RuntimeError(f"`sif_parser` not importable: {_sif_import_error}")
    frames, meta = sif_parser.np_open(path, ignore_corrupt=True)
    frames = np.asarray(frames)
    if frames.ndim == 2:
        frames = frames[None, ...]
    return frames, meta


def _to_cps(frame2d: np.ndarray, meta: Dict) -> Tuple[np.ndarray, Meta]:
    gain_dac = meta.get("GainDAC", 1) or 1
    exposure_time = meta.get("ExposureTime", 1.0) or 1.0  # seconds
    acc = meta.get("AccumulatedCycles", 1) or 1
    cps = frame2d * (5.0 / gain_dac) / exposure_time / acc
    return cps, Meta(exposure_ms=float(exposure_time) * 1000.0, gain=float(gain_dac))


def _downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    return arr[::factor, ::factor]


def _crop_region(arr: np.ndarray, region: str) -> np.ndarray:
    """Crop to a quadrant matching the integrate_sif convention.
       Full frame assumed 512x512. Region: '1','2','3','4','all','custom'."""
    region = str(region)
    h, w = arr.shape[-2], arr.shape[-1]
    # Guard if frames aren't 512; compute midpoints dynamically
    mid_h, mid_w = h // 2, w // 2
    if region == '3':
        return arr[0:mid_h, 0:mid_w]
    elif region == '4':
        return arr[0:mid_h, mid_w:w]
    elif region == '1':
        return arr[mid_h:h, 0:mid_w]
    elif region == '2':
        return arr[mid_h:h, mid_w:w]
    elif region == 'custom':
        # Matches user's special case; clamp to bounds if non-512
        y0, y1, x0, x1 = 312, min(512, h), 56, min(256, w)
        return arr[y0:y1, x0:x1]
    # 'all'
    return arr


def _compute_norm(selected_idxs: List[int], frames: np.ndarray, meta: Dict, log_scale: bool, region: str, downsample: int) -> Tuple[Optional[Normalize], float, float]:
    vmin = np.inf
    vmax = -np.inf
    for i in selected_idxs:
        cps, _ = _to_cps(frames[i], meta)
        cps = _crop_region(cps, region)
        cps = _downsample(cps, downsample)
        fi_min = float(np.nanmin(cps))
        fi_max = float(np.nanmax(cps))
        if fi_min < vmin:
            vmin = fi_min
        if fi_max > vmax:
            vmax = fi_max
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return (LogNorm() if log_scale else None, np.nan, np.nan)
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else Normalize(vmin=vmin, vmax=vmax)
    return norm, vmin, vmax


def _apply_norm_to_uint8(frame_cps: np.ndarray, norm: Optional[Normalize]) -> np.ndarray:
    # Map CPS -> [0, 255]
    if norm is None:
        mn = float(np.nanmin(frame_cps))
        mx = float(np.nanmax(frame_cps))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return np.zeros_like(frame_cps, dtype=np.uint8)
        scaled = (frame_cps - mn) / (mx - mn)
    else:
        scaled = norm(frame_cps)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    scaled = np.clip(scaled, 0, 1)
    return (scaled * 255.0).astype(np.uint8)


def _apply_colormap(gray_u8: np.ndarray, cmap_name: str) -> np.ndarray:
    if cmap_name.lower() in ("gray", "grey"):
        return gray_u8  # single-channel grayscale
    cmap = mpl_cm.get_cmap(cmap_name)
    rgba = cmap(gray_u8.astype(np.float32) / 255.0)  # (H,W,4) float
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)   # (H,W,3)
    return rgb


def _encode_video(frames_u8: List[np.ndarray], fps: int, format_ext: str) -> bytes:
    if imageio is None:
        raise RuntimeError(f"imageio not available: {_imageio_err}")
    if not _has_ffmpeg:
        raise RuntimeError("FFmpeg backend not available. Install `imageio-ffmpeg` to enable MP4/MOV export.")

    # Ensure 3-channel for common codecs
    safe_frames = []
    for f in frames_u8:
        if f.ndim == 2:
            f = np.stack([f, f, f], axis=-1)
        safe_frames.append(f)

    if format_ext.lower() == "mp4":
        codec = "libx264"; pixelformat = "yuv420p"
    elif format_ext.lower() == "mov":
        codec = "libx264"; pixelformat = "yuv420p"
    else:
        raise ValueError("Unsupported video format for encoder")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format_ext}") as tmp:
        tmp_path = tmp.name
    try:
        writer = imageio.get_writer(tmp_path, fps=fps, codec=codec, pixelformat=pixelformat, quality=8, format="FFMPEG")
        for f in safe_frames:
            writer.append_data(f)
        writer.close()
        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _encode_tiff_stack(frames_u8: List[np.ndarray]) -> bytes:
    # Use tifffile if available; fall back to imageio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tiff") as tmp:
        tmp_path = tmp.name
    try:
        if tifffile is not None:
            arr = np.stack(frames_u8, axis=0)
            tifffile.imwrite(tmp_path, arr)
        else:
            if imageio is None:
                raise RuntimeError(f"Neither tifffile nor imageio available (tifffile error: {_tif_err}, imageio error: {_imageio_err})")
            imageio.mimwrite(tmp_path, frames_u8, format="TIFF")
        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ===================== PUBLIC ENTRYPOINT =====================

def run():
    st.title("SIF Movie Exporter")
    st.caption("Preview as a video/GIF, then download as MP4/MOV/TIFF stack. No fitting.")

    with st.sidebar:
        st.header("Controls")
        # Visual/normalization
        colormap = st.selectbox("Colormap", ["gray", "magma", "viridis", "plasma", "hot", "hsv", "cividis", "inferno"], index=0)
        univ_min_max = st.checkbox("Universal min/max across frames", value=False)
        log_scale = st.checkbox("Log intensity scaling", value=False)
        region = st.selectbox("Region", options=["all", "1", "2", "3", "4", "custom"], index=0,
                              help="Quadrants: 1=bottom-left, 2=bottom-right, 3=top-left, 4=top-right; custom matches your integrate_sif crop.")
        # Temporal/size
        fps = st.slider("FPS (preview & video)", 1, 60, 15)
        stride = st.number_input("Use every Nth frame", min_value=1, value=1)
        downsample = st.number_input("Downsample factor", min_value=1, value=1)
        # Export format
        export_fmt = st.selectbox("Download format", ["MP4", "MOV", "TIFF"], index=0)

    uploaded = st.file_uploader("Upload a .sif movie", type=["sif"], accept_multiple_files=False)
    if not uploaded:
        st.info("Upload a .sif file to begin.")
        return

    # Ensure variables exist for finally/cleanup
    raw_frames = None
    frames_u8: List[np.ndarray] = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".sif") as tmp:
        tmp.write(uploaded.getbuffer())
        sif_path = tmp.name

    try:
        raw_frames, meta_raw = _np_open_all(sif_path)
        T = int(raw_frames.shape[0])
        if T == 0:
            st.error("No frames found in SIF file.")
            return

        # Select frames by stride
        idxs = list(range(0, T, int(stride)))
        # Compute normalization (if requested)
        norm = None
        if univ_min_max:
            norm, _, _ = _compute_norm(idxs, raw_frames, meta_raw, log_scale, region, int(downsample))

        progress = st.progress(0.0, text="Preparing frames…")
        for j, i in enumerate(idxs):
            cps, _meta = _to_cps(raw_frames[i], meta_raw)
            cps = _crop_region(cps, region)
            cps = _downsample(cps, int(downsample))
            u8 = _apply_norm_to_uint8(cps, norm)
            rgb_or_gray = _apply_colormap(u8, colormap)
            frames_u8.append(rgb_or_gray)
            progress.progress((j + 1) / len(idxs), text=f"Processed frame {i+1} ({j+1}/{len(idxs)})")
        progress.empty()

        # Preview: MP4 if FFmpeg, else GIF
        if _has_ffmpeg:
            try:
                preview_bytes = _encode_video(frames_u8, fps=fps, format_ext="mp4")
                st.subheader("Preview")
                st.video(preview_bytes)
            except Exception as e:
                st.warning(f"MP4 preview unavailable: {e}")
                if imageio is not None:
                    from io import BytesIO
                    gif_buf = BytesIO()
                    imageio.mimsave(gif_buf, frames_u8, format="GIF", duration=1.0 / max(1, fps))
                    st.image(gif_buf.getvalue(), caption="GIF preview (encoding fallback)", output_format="GIF")
                else:
                    st.info("Install `imageio` for GIF preview.")
        else:
            if imageio is not None:
                from io import BytesIO
                gif_buf = BytesIO()
                imageio.mimsave(gif_buf, frames_u8, format="GIF", duration=1.0 / max(1, fps))
                st.subheader("Preview")
                st.image(gif_buf.getvalue(), caption="GIF preview (FFmpeg not available)", output_format="GIF")
            else:
                st.info("Install `imageio` for GIF preview.")

        # Build the requested download
        try:
            if export_fmt in ("MP4", "MOV"):
                if not _has_ffmpeg:
                    st.error("FFmpeg backend not available. Install `imageio-ffmpeg` to enable MP4/MOV export.")
                    # Offer TIFF as a fallback without throwing
                    export_fmt = "TIFF"
                else:
                    ext = export_fmt.lower()
                    dl_bytes = _encode_video(frames_u8, fps=fps, format_ext=ext)
                    mime = "video/mp4" if ext == "mp4" else "video/quicktime"
                    today = date.today().strftime("%Y%m%d")
                    st.download_button(
                        label=f"Download {export_fmt}",
                        data=dl_bytes,
                        file_name=f"sif_movie_{today}.{ext}",
                        mime=mime,
                    )
                    return  # done

            # TIFF path (explicit or fallback)
            dl_bytes = _encode_tiff_stack(frames_u8)
            mime = "image/tiff"
            ext = "tiff"
            today = date.today().strftime("%Y%m%d")
            st.download_button(
                label=f"Download TIFF",
                data=dl_bytes,
                file_name=f"sif_movie_{today}.{ext}",
                mime=mime,
            )
        except Exception as e:
            st.error(f"Export failed: {e}")

    finally:
        # Cleanup aggressively but safely
        try:
            raw_frames  # noqa: F823
            del raw_frames
        except Exception:
            pass
    
        try:
            frames_u8  # noqa: F823
            del frames_u8
        except Exception:
            pass
    
        gc.collect()


# Allow running standalone
if __name__ == "__main__":
    run()
