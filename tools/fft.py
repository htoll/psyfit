"""Interactive TEM FFT lattice analysis.

Workflow
────────
1. Upload an HRTEM (.dm3/.emd) atomic-resolution image and set / confirm the pixel size.
2. Draw a box over a well-ordered lattice region; its FFT is computed live.
3. Reciprocal-lattice spots are detected (sub-pixel) and converted to g = 1/d in 1/nm.
4. The spot set is *blindly indexed* against six candidate host matrices
   (α/β-NaYF₄, α/β-NaYbF₄, LiYF₄, LiYbF₄): every phase × low-index zone axis is
   searched for the orientation that reproduces the measured d-spacings and inter-spot
   angles. The best fit gives the phase, zone axis, an (hkl) label per spot, and a
   calibration cross-check.
5. A rotational (azimuthally-averaged) radial profile gives all lattice d-spacings at
   once, useful for noisy or polycrystalline ROIs.
6. Results (annotated FFT + spot/phase tables) can be exported.

The crystallographic engine lives in ``tools/crystallography.py`` (Streamlit-free, unit
tested by ``tools/verify_crystallography.py``).
"""
from __future__ import annotations

import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import streamlit.elements.image as st_image

# streamlit_drawable_canvas calls
#   st_image.image_to_url(image, width:int, clamp, channels, output_format, image_id)
# but Streamlit ≥1.5x renamed/retyped the 2nd arg from an int width to a `layout_config`
# object (accessed as layout_config.width). Install a compatibility adapter that resolves
# the genuine implementation and, when it expects a layout_config, wraps the int width in a
# minimal duck-typed object. Guarded so re-importing this module doesn't double-wrap.
if not getattr(st_image, "_canvas_compat_shim", False):
    try:
        from streamlit.elements.lib.image_utils import image_to_url as _real_itu
    except ImportError:
        _real_itu = getattr(st_image, "image_to_url", None)

    _itu_uses_layout_config = False
    if _real_itu is not None:
        try:
            import inspect as _inspect
            _p = list(_inspect.signature(_real_itu).parameters)
            _itu_uses_layout_config = len(_p) >= 2 and _p[1] == "layout_config"
        except (TypeError, ValueError):
            pass

    class _LayoutShim:
        __slots__ = ("width", "height")

        def __init__(self, width):
            self.width = width if isinstance(width, int) else None
            self.height = None

    def _canvas_image_to_url(image, width, clamp, channels, output_format,
                             image_id, allow_emoji=False):
        if _real_itu is None:
            return ""
        if _itu_uses_layout_config:
            return _real_itu(image, _LayoutShim(width), clamp, channels, output_format, image_id)
        return _real_itu(image, width, clamp, channels, output_format, image_id)

    st_image.image_to_url = _canvas_image_to_url
    st_image._canvas_compat_shim = True

from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from skimage.feature import peak_local_max
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Crystallography engine — works whether imported as tools.fft (app) or fft (standalone).
try:
    from tools import crystallography as xtal
except ImportError:  # running from within tools/
    import crystallography as xtal

# Optional import of ncempy for file reading
try:
    from ncempy.io import dm as ncem_dm
    from ncempy.io import emd as ncem_emd
except ImportError:
    ncem_dm = None
    ncem_emd = None

# Optional h5py — needed for Velox/Thermo Fisher .emd files (HDF5 with a Data/Image group),
# which ncempy's Berkeley-EMD reader cannot open.
try:
    import h5py
except ImportError:
    h5py = None


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class TEMImage:
    data: np.ndarray
    nm_per_px: float
    filename: str


# ═══════════════════════════════════════════════════════════════════════════
# File reading (.dm3 via ncempy; .emd via h5py — handles Velox & Berkeley layouts)
# ═══════════════════════════════════════════════════════════════════════════
def _extract_pixel_size_nm(meta_str: str) -> float:
    """Pull the pixel size (→ nm) from a Velox JSON metadata string. Velox stores
    PixelSize.width/height in metres (often as strings), so we average and ×1e9."""
    def _find(obj):
        if isinstance(obj, dict):
            if "PixelSize" in obj and isinstance(obj["PixelSize"], dict):
                ps = obj["PixelSize"]
                vals = [float(v) for v in (ps.get("width"), ps.get("height")) if v is not None]
                if vals:
                    return (sum(vals) / len(vals)) * 1e9
            for v in obj.values():
                r = _find(v)
                if r is not None:
                    return r
        elif isinstance(obj, list):
            for v in obj:
                r = _find(v)
                if r is not None:
                    return r
        return None

    try:
        val = _find(json.loads(meta_str))
        if val is not None:
            return float(val)
    except Exception:
        pass
    m = re.search(r'"PixelSize"\s*:\s*\{[^}]*"width"\s*:\s*"?([0-9.eE+-]+)"?\s*,\s*'
                  r'"height"\s*:\s*"?([0-9.eE+-]+)"?', meta_str)
    if m:
        try:
            return ((float(m.group(1)) + float(m.group(2))) / 2.0) * 1e9
        except Exception:
            pass
    return float("nan")


def _read_dm3(tmp_path: str) -> Tuple[np.ndarray, float]:
    if ncem_dm is None:
        raise RuntimeError("ncempy is not installed (needed for .dm3). pip install ncempy")
    with ncem_dm.fileDM(tmp_path, verbose=False) as rdr:
        im = rdr.getDataset(0)
        data = np.array(im["data"], dtype=np.float32)
        nm_per_px = np.nan
        if "pixelSize" in im and len(im["pixelSize"]) > 0:
            val = im["pixelSize"][0]
            nm_per_px = val * 1e9 if val < 1e-6 else val
        if np.isnan(nm_per_px):
            md = rdr.allTags
            for key, factor in [
                ("ImageList.1.ImageData.Calibrations.Dimension.0.Scale", 1e9),
                ("pixelSize.x", 1e9), ("xscale", 1e9),
                ("root.ImageList.1.ImageData.Calibrations.Dimension.0.Scale", 1),
            ]:
                try:
                    val = md
                    for k in key.split("."):
                        val = val[k]
                    if isinstance(val, (int, float)) and val > 0:
                        nm_per_px = float(val) * factor
                        break
                except Exception:
                    continue
    return data, nm_per_px


def _read_emd(tmp_path: str) -> Tuple[np.ndarray, float]:
    """Read a Velox/Thermo Fisher .emd (Data/Image group) or a generic HDF5 EMD via h5py,
    falling back to ncempy's Berkeley-EMD reader if h5py is unavailable."""
    if h5py is None:
        if ncem_emd is None:
            raise RuntimeError("Reading .emd needs h5py or ncempy. pip install h5py")
        with ncem_emd.fileEMD(tmp_path, readonly=True) as f:
            for group in f.list_groups():
                try:
                    ds = f.get_dataset(group)
                    if isinstance(ds, tuple) and len(ds) >= 1:
                        return np.array(ds[0], dtype=np.float32), np.nan
                except Exception:
                    continue
        raise ValueError("No readable dataset in EMD file.")

    data = None
    nm_per_px = np.nan
    with h5py.File(tmp_path, "r") as h5:
        # Velox layout: /Data/Image/<uid>/{Data, Metadata}. Data is (H, W, n_frames).
        if "Data" in h5 and "Image" in h5["Data"] and len(h5["Data"]["Image"].keys()):
            dg = h5["Data"]["Image"][list(h5["Data"]["Image"].keys())[0]]
            stack = dg["Data"][()]
            if stack.ndim == 3:
                data = stack[:, :, 0].astype(np.float32)
            elif stack.ndim == 2:
                data = stack.astype(np.float32)
            if "Metadata" in dg:
                try:
                    meta = dg["Metadata"][()].tobytes().decode("utf-8", errors="ignore")
                    nm_per_px = _extract_pixel_size_nm(meta)
                except Exception:
                    pass
        if data is None:
            # Generic fallback: use the largest 2-D(+) dataset in the file.
            datasets: List["h5py.Dataset"] = []
            h5.visititems(lambda name, obj: datasets.append(obj)
                          if isinstance(obj, h5py.Dataset) and obj.ndim >= 2 else None)
            if datasets:
                best = max(datasets, key=lambda d: int(np.prod(d.shape[:2])))
                arr = np.array(best, dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr[..., 0] if arr.shape[2] in (1, 3, 4) else arr[0]
                data = arr
    if data is None:
        raise ValueError("No image dataset found in EMD file.")
    return data, nm_per_px


@st.cache_data(show_spinner=False)
def get_file_content(file_bytes: bytes, filename: str) -> TEMImage:
    """Read .dm3/.emd bytes and extract the pixel size where available."""
    suffix = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if suffix == ".dm3":
            data, nm_per_px = _read_dm3(tmp_path)
        elif suffix == ".emd":
            data, nm_per_px = _read_emd(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        if data is None:
            raise ValueError("Could not extract image data.")
        if data.ndim > 2:  # take the first frame of a stack
            data = data[0] if data.shape[0] < data.shape[-1] else data[..., 0]
        if np.isnan(nm_per_px) or nm_per_px == 0:
            nm_per_px = 1.0
        return TEMImage(data=data, nm_per_px=nm_per_px, filename=filename)
    finally:
        # On Windows the HDF5 reader can still hold the file handle briefly, so os.remove
        # raises WinError 32. Swallow it (as tem_analysis.py does) so a lock never masks a
        # successful read; the OS reclaims the temp file later.
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except PermissionError:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# FFT + spot detection
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def process_fft_image(roi_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the (Hanning-windowed) power spectrum and an ImageJ-style display image."""
    roi_clean = np.nan_to_num(roi_data.astype(np.float64))
    h, w = roi_clean.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    fft_shifted = np.fft.fftshift(np.fft.fft2(roi_clean * window))
    fft_mag = np.abs(fft_shifted)

    fft_log = np.log10(fft_mag + 1.0)
    fft_display = gaussian_filter(fft_log, sigma=1.0)

    cy, cx = h // 2, w // 2
    mask = np.ones_like(fft_display, dtype=bool)
    mask[cy - 2:cy + 3, cx - 2:cx + 3] = False
    v_min = np.percentile(fft_display[mask], 1)
    v_max = np.percentile(fft_display[mask], 99.9)
    if v_max == v_min:
        v_max = v_min + 1e-6
    fft_norm = np.clip((fft_display - v_min) / (v_max - v_min), 0, 1)
    fft_rgb = (plt.get_cmap("gray")(fft_norm)[:, :, :3] * 255).astype(np.uint8)
    return fft_mag, fft_rgb


def _refine_subpixel(fft_mag: np.ndarray, coords: np.ndarray, radius: int = 2) -> np.ndarray:
    """Intensity-weighted centroid refinement of each peak within ±radius."""
    h, w = fft_mag.shape
    refined = []
    for r, c in coords:
        r0, r1 = max(0, r - radius), min(h, r + radius + 1)
        c0, c1 = max(0, c - radius), min(w, c + radius + 1)
        patch = fft_mag[r0:r1, c0:c1]
        tot = patch.sum()
        if tot <= 0:
            refined.append((float(r), float(c)))
            continue
        rr, cc = np.mgrid[r0:r1, c0:c1]
        refined.append((float((rr * patch).sum() / tot), float((cc * patch).sum() / tot)))
    return np.array(refined)


def detect_spots(fft_mag: np.ndarray, nm_per_px: float, roi_shape: Tuple[int, int],
                 sensitivity: float, min_dist_px: int, dc_radius: int,
                 max_spots: int) -> pd.DataFrame:
    """Detect reciprocal-lattice spots and return a table with g-vectors (1/nm).

    Columns: row, col (sub-pixel, on the shifted FFT), gx, gy, g (1/nm), d_nm, azimuth_deg.
    ``gx``/``gy`` are relative to the DC term; the reciprocal sampling is
    1/(N·pixel_size) per FFT pixel and is applied per axis (handles non-square ROIs)."""
    h, w = fft_mag.shape
    cy, cx = h // 2, w // 2

    search = fft_mag.copy()
    search[cy - dc_radius:cy + dc_radius + 1, cx - dc_radius:cx + dc_radius + 1] = 0.0
    coords = peak_local_max(search, min_distance=min_dist_px,
                            threshold_rel=sensitivity, num_peaks=max_spots)
    if len(coords) == 0:
        return pd.DataFrame(columns=["row", "col", "gx", "gy", "g", "d_nm", "azimuth_deg"])

    coords = _refine_subpixel(fft_mag, coords)
    roi_h, roi_w = roi_shape
    dfy = 1.0 / (roi_h * nm_per_px)     # 1/nm per FFT row
    dfx = 1.0 / (roi_w * nm_per_px)     # 1/nm per FFT col

    rows = []
    for r, c in coords:
        gy = (r - cy) * dfy
        gx = (c - cx) * dfx
        g = float(np.hypot(gx, gy))
        if g <= 0:
            continue
        rows.append({
            "row": r, "col": c, "gx": gx, "gy": gy, "g": g,
            "d_nm": 1.0 / g, "azimuth_deg": float(np.degrees(np.arctan2(gy, gx))),
        })
    df = pd.DataFrame(rows).sort_values("g").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def radial_profile(fft_mag: np.ndarray, nm_per_px: float, roi_shape: Tuple[int, int],
                   dc_radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """Azimuthally-averaged power vs spatial frequency (1/nm).

    Uses an isotropic frequency axis (mean of the two per-axis samplings) so a single
    radial coordinate is meaningful even for mildly non-square ROIs."""
    h, w = fft_mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - cy, xx - cx)
    r_int = r.astype(int)

    prof = np.bincount(r_int.ravel(), weights=fft_mag.ravel())
    counts = np.bincount(r_int.ravel())
    counts[counts == 0] = 1
    prof = prof / counts

    roi_h, roi_w = roi_shape
    df_iso = 0.5 * (1.0 / (roi_h * nm_per_px) + 1.0 / (roi_w * nm_per_px))
    freq = np.arange(len(prof)) * df_iso
    prof[:max(dc_radius, 1)] = 0.0      # suppress the DC beam
    return freq, prof


def profile_peaks(freq: np.ndarray, prof: np.ndarray,
                  prominence_frac: float = 0.05) -> pd.DataFrame:
    """Peaks in the radial profile → d-spacings (nm)."""
    if prof.size == 0 or prof.max() <= 0:
        return pd.DataFrame(columns=["freq", "d_nm", "intensity"])
    idx, props = find_peaks(prof, prominence=prominence_frac * prof.max())
    idx = idx[freq[idx] > 0]
    rows = [{"freq": float(freq[i]), "d_nm": float(1.0 / freq[i]),
             "intensity": float(prof[i])} for i in idx]
    return pd.DataFrame(rows).sort_values("intensity", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Overlay / export helpers
# ═══════════════════════════════════════════════════════════════════════════
def _hkl_label(phase: xtal.Phase, hkl) -> str:
    """(hkl) string; 4-index Miller-Bravais (hkil) for hexagonal phases."""
    if phase.system == "hexagonal":
        parts = list(xtal.to_miller_bravais(hkl))
    else:
        parts = list(hkl)
    return "(" + " ".join(f"{p}" for p in parts) + ")"


def annotated_fft_figure(fft_rgb: np.ndarray, spots: pd.DataFrame,
                         solution: Optional[xtal.Solution]) -> "plt.Figure":
    """Matplotlib figure of the FFT with detected spots and (hkl) labels, for PNG export."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    ax.imshow(fft_rgb)
    cy, cx = fft_rgb.shape[0] / 2, fft_rgb.shape[1] / 2
    ax.plot(cx, cy, "rx", ms=8)
    matched = {m.spot_index: m for m in solution.matches} if solution else {}
    for i, s in spots.iterrows():
        col = "lime" if i in matched else "orange"
        ax.plot(s["col"], s["row"], "o", mfc="none", mec=col, ms=12, mew=1.5)
        if i in matched:
            ax.text(s["col"] + 4, s["row"] - 4,
                    _hkl_label(solution.phase, matched[i].hkl),
                    color="lime", fontsize=8, weight="bold")
    if solution:
        ax.set_title(f"{solution.phase.label}  zone {list(solution.uvw)}  "
                     f"(cov {solution.coverage:.0%})", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════
def run():
    st.title("TEM FFT Lattice Analysis — Phase ID & Indexing")

    if "last_file_id" not in st.session_state:
        st.session_state.last_file_id = None

    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload HRTEM image", type=["dm3", "emd"])
        st.divider()
        manual_scale_container = st.container()
        st.divider()
        st.write("**Spot detection**")
        sensitivity = st.slider("Sensitivity (rel. threshold)", 0.01, 1.0, 0.10, 0.01)
        min_dist_px = st.slider("Min spot spacing (px)", 1, 50, 8)
        dc_radius = st.slider("DC mask radius (px)", 1, 40, 8)
        max_spots = st.slider("Max spots", 4, 60, 24)
        st.divider()
        st.write("**Indexing**")
        tol_pct = st.slider("Match tolerance (%)", 2, 15, 6)
        max_zone = st.select_slider("Max zone-axis index", options=[1, 2, 3], value=2)
        phase_labels = {p.label: k for k, p in xtal.PHASES.items()}
        pinned = st.multiselect(
            "Restrict to phases (blind = leave empty)", list(phase_labels.keys()))

    if not uploaded_file:
        st.info("Upload a .dm3 or .emd HRTEM image to begin.")
        _show_reference_table()
        return

    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.last_file_id != file_id:
        st.session_state.last_file_id = file_id
        uploaded_file.seek(0)

    try:
        tem_img_raw = get_file_content(uploaded_file.getvalue(), uploaded_file.name)
        with manual_scale_container:
            val_to_show = float(tem_img_raw.nm_per_px)
            if val_to_show == 1.0:
                st.warning("⚠️ Pixel size not found — enter it:")
            actual_scale = st.number_input("Pixel size (nm/px)", value=val_to_show,
                                           format="%.5f", min_value=0.0)
            tem_img = TEMImage(tem_img_raw.data, actual_scale, tem_img_raw.filename)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # --- Display image prep ---
    p2, p98 = np.percentile(tem_img.data, (2, 98))
    img_norm = np.clip((tem_img.data - p2) / (p98 - p2 + 1e-9), 0, 1) * 255
    img_rgb = np.stack((img_norm.astype(np.uint8),) * 3, axis=-1)

    CANVAS_WIDTH = 350
    CANVAS_HEIGHT = int(CANVAS_WIDTH * (img_rgb.shape[0] / img_rgb.shape[1]))
    pil_display = Image.fromarray(img_rgb).resize(
        (CANVAS_WIDTH, CANVAS_HEIGHT), resample=Image.Resampling.LANCZOS)

    c_left, c_mid, c_right = st.columns([1, 1, 1])

    with c_left:
        st.caption("Draw a box over an ordered lattice region")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.2)", stroke_width=2, stroke_color="#FFFFFF",
            background_image=pil_display, update_streamlit=True,
            height=CANVAS_HEIGHT, width=CANVAS_WIDTH, drawing_mode="rect", key="roi_canvas",
        )

    # --- Extract ROI ---
    roi_data = None
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        obj = canvas_result.json_data["objects"][-1]
        sx = tem_img.data.shape[1] / CANVAS_WIDTH
        sy = tem_img.data.shape[0] / CANVAS_HEIGHT
        x, y = int(obj["left"] * sx), int(obj["top"] * sy)
        w, h = int(obj["width"] * sx), int(obj["height"] * sy)
        y0, y1 = max(0, y), min(y + h, tem_img.data.shape[0])
        x0, x1 = max(0, x), min(x + w, tem_img.data.shape[1])
        if y1 - y0 > 8 and x1 - x0 > 8:
            roi_data = tem_img.data[y0:y1, x0:x1]

    if roi_data is None:
        with c_mid:
            st.info("Draw a box on the left image.")
        with c_right:
            _show_reference_table()
        return

    if tem_img.nm_per_px <= 0:
        with c_mid:
            st.error("Set a positive pixel size to compute d-spacings.")
        return

    # --- FFT + spots ---
    fft_mag, fft_rgb = process_fft_image(roi_data)
    spots = detect_spots(fft_mag, tem_img.nm_per_px, roi_data.shape,
                         sensitivity, min_dist_px, dc_radius, max_spots)

    # --- Blind index ---
    phases = None
    if pinned:
        phases = [xtal.PHASES[phase_labels[l]] for l in pinned]
    solutions: List[xtal.Solution] = []
    if len(spots) >= 3:
        measured = spots[["gx", "gy"]].to_numpy()
        solutions = xtal.index_pattern(
            measured, phases=phases, tol_frac=tol_pct / 100.0,
            max_zone_index=int(max_zone), top_n=5)
    top = solutions[0] if solutions else None
    matched = {m.spot_index: m for m in top.matches} if top else {}

    # --- FFT overlay (plotly) ---
    with c_mid:
        cy, cx = fft_mag.shape[0] // 2, fft_mag.shape[1] // 2
        fig = px.imshow(fft_rgb)
        if len(spots):
            unm = spots[~spots.index.isin(matched)]
            fig.add_trace(go.Scatter(
                x=unm["col"], y=unm["row"], mode="markers",
                marker=dict(color="orange", size=11, symbol="circle-open", line=dict(width=2)),
                name="spot", hovertext=[f"d={d:.3f} nm" for d in unm["d_nm"]]))
            if matched:
                mdf = spots[spots.index.isin(matched)]
                labels = [_hkl_label(top.phase, matched[i].hkl) for i in mdf.index]
                fig.add_trace(go.Scatter(
                    x=mdf["col"], y=mdf["row"], mode="markers+text",
                    marker=dict(color="lime", size=12, symbol="circle-open", line=dict(width=2)),
                    text=labels, textposition="top center",
                    textfont=dict(color="lime", size=10), name="indexed"))
        fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers",
                                 marker=dict(color="red", symbol="x", size=9), showlegend=False))
        fig.update_layout(width=350, height=350, margin=dict(l=0, r=0, t=0, b=0),
                          xaxis=dict(showgrid=False, showticklabels=False),
                          yaxis=dict(showgrid=False, showticklabels=False), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{len(spots)} spots detected"
                   + (f" · {len(matched)} indexed" if matched else ""))

    # --- Results ---
    with c_right:
        if top is None:
            st.warning("No phase solution.")
            st.caption("Try more spots (raise sensitivity), a larger ROI, or a wider tolerance.")
        else:
            st.success(f"**{top.phase.label}**")
            zone = list(top.uvw)
            if top.phase.system == "hexagonal":
                u, v, w = top.uvw
                zone = [u, v, -(u + v), w]
            st.metric("Zone axis", str(zone))
            st.metric("Coverage", f"{top.n_matched}/{top.n_spots} spots")
            st.metric("Mean d error", f"{top.mean_rel_err * 100:.1f}%")
            cal = top.scale
            cal_msg = "scale ≈ 1 ✓" if abs(cal - 1) < 0.03 else "check calibration"
            st.caption(f"Fit scale {cal:.3f} ({cal_msg}) · SG {top.phase.space_group}")
            st.caption(f"Implied pixel size: {tem_img.nm_per_px * cal:.5f} nm/px")
            st.caption(f"Ref: {top.phase.reference}")

    # --- Detail tables & extras (full width) ---
    st.divider()
    d1, d2 = st.columns([3, 2])

    with d1:
        st.subheader("Detected spots")
        disp = spots.copy()
        disp["hkl"] = [
            _hkl_label(top.phase, matched[i].hkl) if (top and i in matched) else ""
            for i in disp.index]
        disp = disp[["d_nm", "g", "azimuth_deg", "hkl"]].rename(
            columns={"d_nm": "d (nm)", "g": "|g| (1/nm)", "azimuth_deg": "azimuth (°)"})
        st.dataframe(disp.style.format({"d (nm)": "{:.4f}", "|g| (1/nm)": "{:.4f}",
                                        "azimuth (°)": "{:.1f}"}), height=260)

        if len(spots) >= 2:
            st.subheader("Inter-spot angles (from DC)")
            st.dataframe(_angle_table(spots, matched, top), height=180)

    with d2:
        st.subheader("Phase ranking")
        st.dataframe(_ranking_table(solutions), height=200)

        st.subheader("Radial profile")
        freq, prof = radial_profile(fft_mag, tem_img.nm_per_px, roi_data.shape, dc_radius)
        pk = profile_peaks(freq, prof)
        rfig = go.Figure()
        rfig.add_trace(go.Scatter(x=freq, y=prof, mode="lines", line=dict(color="#4cc9f0")))
        if len(pk):
            rfig.add_trace(go.Scatter(
                x=pk["freq"], y=pk["intensity"], mode="markers+text",
                marker=dict(color="#f72585", size=8),
                text=[f"{d:.3f} nm" for d in pk["d_nm"]], textposition="top center",
                textfont=dict(size=9)))
        rfig.update_layout(height=220, margin=dict(l=0, r=0, t=6, b=0), showlegend=False,
                           xaxis_title="spatial frequency (1/nm)", yaxis_title="⟨|F|⟩")
        st.plotly_chart(rfig, use_container_width=True)

    # --- Export ---
    st.divider()
    _export_row(fft_rgb, spots, matched, top, solutions, pk, tem_img.filename)


# ═══════════════════════════════════════════════════════════════════════════
# UI helper tables
# ═══════════════════════════════════════════════════════════════════════════
def _angle_table(spots: pd.DataFrame, matched: dict,
                 top: Optional[xtal.Solution]) -> pd.DataFrame:
    """Pairwise angles between the strongest few spots (measured; and expected when both
    spots are indexed)."""
    idx = list(spots.sort_values("g").index[:6])
    rows = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            va = np.array([spots.loc[i, "gx"], spots.loc[i, "gy"]])
            vb = np.array([spots.loc[j, "gx"], spots.loc[j, "gy"]])
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na == 0 or nb == 0:
                continue
            ang = np.degrees(np.arccos(np.clip(np.dot(va, vb) / (na * nb), -1, 1)))
            exp = ""
            if top and i in matched and j in matched:
                exp = f"{xtal.interplanar_angle_deg(top.phase, matched[i].hkl, matched[j].hkl):.1f}"
            rows.append({"spot A": i, "spot B": j, "measured (°)": round(ang, 1),
                         "expected (°)": exp})
    return pd.DataFrame(rows)


def _ranking_table(solutions: List[xtal.Solution]) -> pd.DataFrame:
    rows = []
    for s in solutions:
        zone = list(s.uvw)
        if s.phase.system == "hexagonal":
            u, v, w = s.uvw
            zone = [u, v, -(u + v), w]
        rows.append({
            "phase": s.phase.label, "zone": str(zone),
            "matched": f"{s.n_matched}/{s.n_spots}",
            "d err %": round(s.mean_rel_err * 100, 1),
            "scale": round(s.scale, 3), "score": round(s.score, 3)})
    return pd.DataFrame(rows)


def _show_reference_table():
    st.subheader("Candidate host matrices (literature)")
    rows = []
    for p in xtal.PHASES.values():
        rows.append({"phase": p.label, "system": p.system, "a (Å)": p.a,
                     "c (Å)": ("—" if p.system == "cubic" else p.c),
                     "space group": p.space_group, "reference": p.reference})
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def _export_row(fft_rgb, spots, matched, top, solutions, pk, filename):
    stem = os.path.splitext(filename)[0] + "_fft"
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = annotated_fft_figure(fft_rgb, spots, top)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        st.download_button("Download annotated FFT (PNG)", buf.getvalue(),
                           f"{stem}.png", "image/png")
    with c2:
        out = spots.copy()
        out["hkl"] = [
            _hkl_label(top.phase, matched[i].hkl) if (top and i in matched) else ""
            for i in out.index]
        out["phase"] = top.phase.label if top else ""
        st.download_button("Download spots (CSV)", out.to_csv(index=False),
                           f"{stem}_spots.csv", "text/csv")
    with c3:
        st.download_button("Download ranking (CSV)", _ranking_table(solutions).to_csv(index=False),
                           f"{stem}_ranking.csv", "text/csv")


if __name__ == "__main__":
    run()
