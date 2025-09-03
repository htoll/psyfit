# Streamlit: SIF Movie Exporter (MP4/MOV/TIFF) — Region, Labels, Colorbar
# ------------------------------------------------------------------------
# Fast, no‑fitting pipeline. Converts each SIF frame to cps and encodes to a
# video (MP4/MOV via FFmpeg) or multipage TIFF. Shows a playable preview
# (MP4 when FFmpeg available, else GIF). Adds region cropping, per‑frame
# overlays (frame #, exposure, gain), and an embedded colorbar strip.

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

# PIL for text/legend overlays
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:
    Image = None
    ImageDraw = None
    ImageFont = None
    _pil_err = e

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


def _crop_region(arr: np.ndarray, region: str) -> np.ndarray:
    region = str(region)
    h, w = arr.shape[-2], arr.shape[-1]
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
        y0, y1, x0, x1 = 312, min(512, h), 56, min(256, w)
        return arr[y0:y1, x0:x1]
    return arr


def _compute_norm(selected_idxs: List[int], frames: np.ndarray, meta: Dict, log_scale: bool, region: str) -> Optional[Normalize]:
    vmin = np.inf
    vmax = -np.inf
    for i in selected_idxs:
        cps, _ = _to_cps(frames[i], meta)
        cps = _crop_region(cps, region)
        fi_min = float(np.nanmin(cps))
        fi_max = float(np.nanmax(cps))
        if fi_min < vmin:
            vmin = fi_min
        if fi_max > vmax:
            vmax = fi_max
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return LogNorm() if log_scale else None
    return LogNorm(vmin=vmin, vmax=vmax) if log_scale else Normalize(vmin=vmin, vmax=vmax)


def _cps_to_uint8(cps: np.ndarray, norm: Optional[Normalize]) -> np.ndarray:
    if norm is None:
        mn = float(np.nanmin(cps))
        mx = float(np.nanmax(cps))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return np.zeros_like(cps, dtype=np.uint8)
        scaled = (cps - mn) / (mx - mn)
    else:
        scaled = norm(cps)
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


def _make_colorbar_strip(norm: Optional[Normalize], cmap_name: str, height: int, vmin_val: Optional[float], vmax_val: Optional[float]) -> Optional[np.ndarray]:
    if norm is None:
        return None
    # Vertical strip  (height x 24)
    H = int(height)
    W = 28
    gradient = np.linspace(1, 0, H, dtype=np.float32)[:, None]  # 1 at top -> 0 at bottom
    # Map to RGB via colormap
    if cmap_name.lower() in ("gray", "grey"):
        gray = (gradient * 255.0).astype(np.uint8)
        strip = np.repeat(gray, W, axis=1)
        strip_rgb = np.stack([strip, strip, strip], axis=-1)
    else:
        cmap = mpl_cm.get_cmap(cmap_name)
        rgba = cmap(gradient)
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        strip_rgb = np.repeat(rgb, W, axis=1)
    # Add min/max labels using PIL if available
    if Image is not None:
        pil = Image.fromarray(strip_rgb)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        top = "max" if vmin_val is None else (f"{vmax_val:.2g}")
        bot = "min" if vmax_val is None else (f"{vmin_val:.2g}")
        # small padding
        draw.text((2, 2), top, fill=(255, 255, 255), font=font)
        draw.text((2, H - 12), bot, fill=(255, 255, 255), font=font)
        strip_rgb = np.array(pil)
    return strip_rgb


def _overlay_labels(frame_rgb_or_gray: np.ndarray, text: str) -> np.ndarray:
    if Image is None:
        # Minimal fallback: do nothing
        return frame_rgb_or_gray
    if frame_rgb_or_gray.ndim == 2:
        base = np.stack([frame_rgb_or_gray]*3, axis=-1)
    else:
        base = frame_rgb_or_gray
    pil = Image.fromarray(base)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    # Draw a translucent box behind text for readability
    margin = 4
    text_w, text_h = draw.textbbox((0, 0), text, font=font)[2:]
    box = [margin, margin, margin + text_w + 6, margin + text_h + 6]
    draw.rectangle(box, fill=(0, 0, 0, 128))
    draw.text((margin + 3, margin + 3), text, fill=(255, 255, 255), font=font)
    return np.array(pil)


def _concat_right(img: np.ndarray, right: Optional[np.ndarray]) -> np.ndarray:
    if right is None:
        return img
    H = img.shape[0]
    if right.shape[0] != H:
        # resize right to match height using simple nearest neighbor
        scale = H / right.shape[0]
        new_w = max(1, int(round(right.shape[1] * scale)))
        if Image is not None:
            pil = Image.fromarray(right)
            pil = pil.resize((new_w, H), resample=Image.NEAREST)
            right = np.array(pil)
        else:
            # crude fallback: tile or trim
            right = np.resize(right, (H, new_w, right.shape[-1]))
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if right.ndim == 2:
        right = np.stack([right]*3, axis=-1)
    return np.concatenate([img, right], axis=1)


def _encode_video(frames_u8: List[np.ndarray], fps: int, format_ext: str) -> bytes:
    if imageio is None:
        raise RuntimeError(f"imageio not available: {_imageio_err}")
    if not _has_ffmpeg:
        raise RuntimeError("FFmpeg backend not available. Install `imageio-ffmpeg` to enable MP4/MOV export.")
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
    st.caption("Preview as a video/GIF, then download as MP4/MOV/TIFF stack. Region, labels & colorbar overlays.")

    with st.sidebar:
        st.header("Controls")
        colormap = st.selectbox("Colormap", ["gray", "magma", "viridis", "plasma", "hot", "hsv", "cividis", "inferno"], index=0)
        univ_min_max = st.checkbox("Universal min/max across frames", value=False)
        log_scale = st.checkbox("Log intensity scaling", value=False)
        region = st.selectbox("Region", options=["all", "1", "2", "3", "4", "custom"], index=0)
        show_colorbar = st.checkbox("Show colorbar", value=True)
        show_labels = st.checkbox("Show frame # / exposure / gain", value=True)
        fps = st.slider("FPS (preview & video)", 1, 60, 15)
        export_fmt = st.selectbox("Download format", ["MP4", "MOV", "TIFF"], index=0)

    uploaded = st.file_uploader("Upload a .sif movie", type=["sif"], accept_multiple_files=False)
    if not uploaded:
        st.info("Upload a .sif file to begin.")
        return

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

        idxs = list(range(0, T, 1))  # no stride per user request
        norm = _compute_norm(idxs, raw_frames, meta_raw, log_scale, region) if univ_min_max else None

        for i in idxs:
            cps, md = _to_cps(raw_frames[i], meta_raw)
            cps = _crop_region(cps, region)
            u8 = _cps_to_uint8(cps, norm)
            rgb_or_gray = _apply_colormap(u8, colormap)

            # Build colorbar strip if requested and global norm is enabled
            strip = _make_colorbar_strip(norm, colormap, height=rgb_or_gray.shape[0],
                                         vmin_val=float(getattr(norm, 'vmin', np.nan)) if norm else None,
                                         vmax_val=float(getattr(norm, 'vmax', np.nan)) if norm else None) if show_colorbar else None
            framed = _concat_right(rgb_or_gray, strip)

            # Overlay text (frame #, exposure, gain)
            if show_labels:
                exp_ms = md.exposure_ms
                gain = md.gain
                label = f"Frame {i+1} | Exp {exp_ms:g} ms | Gain {gain:g}" if exp_ms is not None and gain is not None else f"Frame {i+1}"
                framed = _overlay_labels(framed, label)

            frames_u8.append(framed)

        # Preview: MP4 if FFmpeg else GIF
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

        # Downloads
        try:
            if export_fmt in ("MP4", "MOV"):
                if not _has_ffmpeg:
                    st.error("FFmpeg backend not available. Install `imageio-ffmpeg` to enable MP4/MOV export.")
                    # Also offer TIFF immediately
                    dl_bytes = _encode_tiff_stack(frames_u8)
                    today = date.today().strftime("%Y%m%d")
                    st.download_button(
                        label="Download TIFF",
                        data=dl_bytes,
                        file_name=f"sif_movie_{today}.tiff",
                        mime="image/tiff",
                    )
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
            else:
                dl_bytes = _encode_tiff_stack(frames_u8)
                today = date.today().strftime("%Y%m%d")
                st.download_button(
                    label="Download TIFF",
                    data=dl_bytes,
                    file_name=f"sif_movie_{today}.tiff",
                    mime="image/tiff",
                )
        except Exception as e:
            st.error(f"Export failed: {e}")

    finally:
        # Cleanup safely
        try:
            if raw_frames is not None:
                del raw_frames
        except Exception:
            pass
        try:
            del frames_u8
        except Exception:
            pass
        gc.collect()


if __name__ == "__main__":
    run()
