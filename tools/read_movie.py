# read_movie.py
# SIF Movie Exporter (MP4/MOV/TIFF) — Region, Labels, Bottom Colorbar, Flip-X, Compact Preview

from __future__ import annotations

import os
import gc
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

# PIL for text/legend overlays & resizing
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




def _overlay_labels(frame_rgb_or_gray: np.ndarray, text: str) -> np.ndarray:
    if Image is None:
        return frame_rgb_or_gray
    if frame_rgb_or_gray.ndim == 2:
        base = np.stack([frame_rgb_or_gray]*3, axis=-1)
    else:
        base = frame_rgb_or_gray
    H = base.shape[0]
    font_px = max(16, int(0.04 * H))  # ~4% of height
    font = _get_font(font_px)

    pil = Image.fromarray(base)
    draw = ImageDraw.Draw(pil)
    margin = max(4, font_px // 4)
    try:
        x1, y1, x2, y2 = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = x2 - x1, y2 - y1
    except Exception:
        text_w, text_h = font_px * 8, font_px
    pad = max(4, font_px // 4)
    box = [margin, margin, margin + text_w + 2*pad, margin + text_h + 2*pad]
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((margin + pad, margin + pad), text, fill=(255, 255, 255), font=font)
    return np.array(pil)


def _make_colorbar_with_ticks(height: int, cmap_name: str, vmin_val: float, vmax_val: float, log_scale: bool) -> np.ndarray:
    H = int(height)
    strip_w = 28
    panel_w = 110  # a bit wider for 5 labels in 1e+XX
    bar_h = int(round(0.75 * H))
    top_pad = (H - bar_h) // 2

    # Build the color strip at 1x height (bar only)
    grad = np.linspace(1, 0, bar_h, dtype=np.float32)[:, None]
    if cmap_name.lower() in ("gray", "grey"):
        gray = (grad * 255.0).astype(np.uint8)
        strip = np.repeat(gray, strip_w, axis=1)
        strip_rgb = np.stack([strip, strip, strip], axis=-1)
    else:
        cmap = mpl_cm.get_cmap(cmap_name)
        rgba = cmap(grad)
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        strip_rgb = np.repeat(rgb, strip_w, axis=1)

    # Compose full canvas (H x (strip+panel)), black
    cbar = np.zeros((H, strip_w + panel_w, 3), dtype=np.uint8)
    cbar[top_pad:top_pad+bar_h, :strip_w, :] = strip_rgb

    # ---- draw ticks/labels on a supersampled panel for crisp text ----
    if Image is None:
        return cbar

    # Create a supersampled panel
    panel_hi = Image.new("RGB", (panel_w * OVERLAY_SCALE, H * OVERLAY_SCALE), (0, 0, 0))
    draw = ImageDraw.Draw(panel_hi)
    font_px = max(12, int(0.032 * H) ) * OVERLAY_SCALE
    font = _get_font(font_px)

    # 5 ticks (linear or log)
    if log_scale and (vmin_val > 0) and np.isfinite(vmin_val) and np.isfinite(vmax_val) and (vmax_val > vmin_val):
        ticks_vals = np.geomspace(vmin_val, vmax_val, 5)
        def norm_fn(v): return (np.log(v) - np.log(vmin_val)) / (np.log(vmax_val) - np.log(vmin_val))
    else:
        ticks_vals = np.linspace(vmin_val, vmax_val, 5)
        def norm_fn(v): return (v - vmin_val) / (vmax_val - vmin_val) if vmax_val != vmin_val else 0.0

    def val_to_y_hi(v):
        t = float(np.clip(norm_fn(v), 0, 1))
        y_bar = int(round((1 - t) * (bar_h - 1)))  # 0..bar_h-1
        return (top_pad + y_bar) * OVERLAY_SCALE

    # Draw ticks on original strip (1x) and labels on hi-res panel
    # First, draw white tick marks along strip edge on the cbar
    pil_cbar = Image.fromarray(cbar)
    draw_cbar = ImageDraw.Draw(pil_cbar)
    tick_len = 6
    for v in ticks_vals:
        y = top_pad + int(round((1 - float(np.clip(norm_fn(v), 0, 1))) * (bar_h - 1)))
        x0 = strip_w - 1
        draw_cbar.line([(x0 - tick_len, y), (x0, y)], fill=(255, 255, 255), width=1)

    # Now labels in hi-res panel
    label_pad_x = 6 * OVERLAY_SCALE
    for v in ticks_vals:
        y_hi = val_to_y_hi(v)
        label = f"{v:.2e}"
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = (60 * OVERLAY_SCALE, 12 * OVERLAY_SCALE)
        draw.text((label_pad_x, max(0, y_hi - th // 2)), label, fill=(255, 255, 255), font=font)

    # Units, top-right, away from labels
    units = "cps"
    try:
        bbox_u = draw.textbbox((0, 0), units, font=font)
        uw, uh = bbox_u[2] - bbox_u[0], bbox_u[3] - bbox_u[1]
    except Exception:
        uw, uh = (28 * OVERLAY_SCALE, 12 * OVERLAY_SCALE)
    unit_x = panel_w * OVERLAY_SCALE - uw - 6 * OVERLAY_SCALE
    unit_y = max(6 * OVERLAY_SCALE, top_pad * OVERLAY_SCALE - uh - 6 * OVERLAY_SCALE)
    draw.text((unit_x, unit_y), units, fill=(255, 255, 255), font=font)

    # Downsample hi-res panel to 1x with LANCZOS and paste onto cbar
    panel_lo = panel_hi.resize((panel_w, H), resample=Image.LANCZOS)
    cbar[:, strip_w:strip_w+panel_w, :] = np.array(panel_lo)

    return np.array(pil_cbar)


    pil = Image.fromarray(cbar)
    draw = ImageDraw.Draw(pil)
    font = _get_font(12)

    # Tick values (5)
    if log_scale and (vmin_val > 0) and np.isfinite(vmin_val) and np.isfinite(vmax_val) and (vmax_val > vmin_val):
        ticks_vals = np.geomspace(vmin_val, vmax_val, 5)
        def norm_fn(v):
            return (np.log(v) - np.log(vmin_val)) / (np.log(vmax_val) - np.log(vmin_val))
    else:
        ticks_vals = np.linspace(vmin_val, vmax_val, 5)
        def norm_fn(v):
            return (v - vmin_val) / (vmax_val - vmin_val) if vmax_val != vmin_val else 0.0

    # Value → y coord within bar (0=top of bar, bar_h=bottom)
    def val_to_y(v):
        t = float(np.clip(norm_fn(v), 0, 1))
        return top_pad + int(round((1 - t) * (bar_h - 1)))

    # Draw ticks + labels
    tick_len = 6
    label_pad_x = 4
    label_positions = []
    for v in ticks_vals:
        y = val_to_y(v)
        x0 = strip_w - 1
        draw.line([(x0 - tick_len, y), (x0, y)], fill=(255, 255, 255), width=1)
        label = f"{v:.2e}"
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = (60, 12)
        draw.text((strip_w + label_pad_x, max(0, y - th // 2)),
                  label, fill=(255, 255, 255), font=font)
        label_positions.append((y, th))

    # Units label "cps" in white, top-right of panel, safely away from ticks
    units = "cps"
    try:
        bbox_u = draw.textbbox((0, 0), units, font=font)
        uw, uh = bbox_u[2] - bbox_u[0], bbox_u[3] - bbox_u[1]
    except Exception:
        uw, uh = (28, 12)
    unit_x = strip_w + panel_w - uw - 4
    unit_y = max(4, top_pad - uh - 4)  # just above the bar
    draw.text((unit_x, unit_y), units, fill=(255, 255, 255), font=font)

    return np.array(pil)



def _concat_right(img: np.ndarray, right: Optional[np.ndarray]) -> np.ndarray:
    if right is None:
        return img
    H = img.shape[0]
    if right.shape[0] != H:
        if Image is not None:
            pil = Image.fromarray(right)
            pil = pil.resize((right.shape[1], H), resample=Image.NEAREST)
            right = np.array(pil)
        else:
            right = np.resize(right, (H, right.shape[1], right.shape[-1]))
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if right.ndim == 2:
        right = np.stack([right]*3, axis=-1)
    return np.concatenate([img, right], axis=1)

def _concat_bottom(img: np.ndarray, bottom: Optional[np.ndarray]) -> np.ndarray:
    if bottom is None:
        return img
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if bottom.ndim == 2:
        bottom = np.stack([bottom]*3, axis=-1)
    H, W = img.shape[:2]
    if bottom.shape[1] != W:
        if Image is not None:
            pil = Image.fromarray(bottom)
            pil = pil.resize((W, bottom.shape[0]), resample=Image.NEAREST)
            bottom = np.array(pil)
        else:
            bottom = np.resize(bottom, (bottom.shape[0], W, bottom.shape[-1]))
    return np.concatenate([img, bottom], axis=0)


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


def _get_font(size: int):
    if ImageFont is None:
        return None
    for name in [
        "Arial.ttf", "arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def run():
    st.title("SIF Movie Exporter")
    OVERLAY_SCALE = 2  # 2x supersampling for higher res text


    with st.sidebar:
        st.header("Controls")
        colormap = st.selectbox("Colormap", ["gray", "magma", "viridis", "plasma", "hot", "hsv", "cividis", "inferno"], index=0)
        univ_min_max = st.checkbox("Universal min/max across frames", value=False)
        log_scale = st.checkbox("Log intensity scaling", value=False)
        region = st.selectbox("Region", options=["all", "1", "2", "3", "4"], index=0)
        show_colorbar = st.checkbox("Show colorbar", value=True)
        show_labels = st.checkbox("Show frame # / exposure / gain", value=True)
        fps = st.slider("FPS (preview & video)", 1, 60, 15)
        export_fmt = st.selectbox("Download format", ["MP4", "MOV", "TIFF"], index=0)

    uploaded = st.file_uploader("Upload a .sif movie", type=["sif"], accept_multiple_files=False)
    uploaded_name = uploaded.name  # e.g. "experiment1.sif"
    base = os.path.splitext(os.path.basename(uploaded_name))[0]
    today = date.today().strftime("%Y%m%d")
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

        idxs = list(range(0, T, 1))  # show all frames
        norm = _compute_norm(idxs, raw_frames, meta_raw, log_scale, region) if univ_min_max else None

        # Build full-res frames (with flip, labels, bottom colorbar)
        for i in idxs:
            cps, md = _to_cps(raw_frames[i], meta_raw)
            cps = _crop_region(cps, region)

            cps = np.flip(cps, axis=0) #flip x axis
            u8 = _cps_to_uint8(cps, norm)
            rgb_or_gray = _apply_colormap(u8, colormap)

            strip = None
            if show_colorbar:
                if norm is not None:
                    vmin_val = float(getattr(norm, 'vmin', np.nan))
                    vmax_val = float(getattr(norm, 'vmax', np.nan))
                else:
                    vmin_val = float(np.nanmin(cps))
                    vmax_val = float(np.nanmax(cps))
                strip = _make_colorbar_with_ticks(
                    height=rgb_or_gray.shape[0],
                    cmap_name=colormap,
                    vmin_val=vmin_val,
                    vmax_val=vmax_val,
                    log_scale=log_scale,
                )
            framed = _concat_right(rgb_or_gray, strip)


            if show_labels:
                if (md.exposure_ms is not None) and (md.gain is not None):
                    label = f"Frame {i+1} | Exp {md.exposure_ms:g} ms | Gain {md.gain:g}"
                else:
                    label = f"Frame {i+1}"
                framed = _overlay_labels(framed, label)

            frames_u8.append(framed)

        # Build smaller frames for preview
        frames_preview = []
        for f in frames_u8:
            if Image is not None:
                pil = Image.fromarray(f if f.ndim == 3 else np.stack([f]*3, axis=-1))
                w, h = pil.size
                frames_preview.append(np.array(pil))
            else:
                frames_preview.append(f)

        # Preview
        col_preview, col_empty = st.columns([1, 2])
        with col_preview:
            st.subheader("Preview")
            if _has_ffmpeg:
                try:
                    preview_bytes = _encode_video(frames_u8, fps=fps, format_ext="mp4")
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
                    st.image(gif_buf.getvalue(), caption="GIF preview (FFmpeg not available)", output_format="GIF")
                else:
                    st.info("Install `imageio` for GIF preview.")


        # Downloads
        try:
            if export_fmt in ("MP4", "MOV"):
                if not _has_ffmpeg:
                    st.error("FFmpeg backend not available. Install `imageio-ffmpeg` to enable MP4/MOV export.")
                    # also offer TIFF immediately
                    dl_bytes = _encode_tiff_stack(frames_u8)
                    today = date.today().strftime("%Y%m%d")
                    st.download_button(
                        label="Download TIFF",
                        data=dl_bytes,
                        file_name=f"{base}_{today}.tiff",
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
                        file_name=f"{base}_{today}.{ext}",
                        mime=mime,
                    )
            else:
                dl_bytes = _encode_tiff_stack(frames_u8)
                today = date.today().strftime("%Y%m%d")
                st.download_button(
                    label="Download TIFF",
                    data=dl_bytes,
                    file_name=f"{base}_{today}.tiff",
                    mime="image/tiff",
                )
        except Exception as e:
            st.error(f"Export failed: {e}")

    finally:
        # Safe cleanup (no UnboundLocalError)
        try:
            if raw_frames is not None:
                del raw_frames
        except Exception:
            pass
        try:
            frames_u8  # ensure defined
            del frames_u8
        except Exception:
            pass
        gc.collect()
