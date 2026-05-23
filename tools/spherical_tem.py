"""Interactive TEM particle analysis tool.

Main features
─────────────
* Upload .dm3 or .emd (Velox HDF5) images.
* RAPID TUNING: Auto-extracts high-contrast ROI for instant parameter feedback.
* Deep Learning Segmentation: Exclusively uses Cellpose for robust shape detection.
* Size-distribution histograms with Gaussian fits, SD, and n.
"""

from __future__ import annotations

import io
import math
import os
import re
import json
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import gaussian_filter, laplace, median_filter, uniform_filter
from scipy.stats import norm
from skimage import exposure
from skimage.measure import regionprops

# Forced Cellpose Import
from cellpose import models

# Optional — emd support requires h5py
try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    h5py = None
    _HAS_H5PY = False

# Optional — dm3 support
try:
    import dm3_lib as pyDM3reader
    _HAS_DM3 = True
except ImportError:
    pyDM3reader = None
    _HAS_DM3 = False


# ═══════════════════════════════════════════════════════════════════════════
# Data structures & Cache Clearing
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class TEMImage:
    data: np.ndarray
    nm_per_px: float

def _clear_cache():
    st.cache_data.clear()
    if "full_run_complete" in st.session_state:
        st.session_state.full_run_complete = False

# ═══════════════════════════════════════════════════════════════════════════
# Cellpose Model Caching
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_cellpose_model(model_type: str = 'cyto3'):
    # Request GPU (if your Mac has Apple Silicon, this tries to use the Metal GPU)
    # It safely falls back to CPU if no GPU is found.
    return models.CellposeModel(gpu=True, model_type=model_type)

# ═══════════════════════════════════════════════════════════════════════════
# Unit conversion & File Reading 
# ═══════════════════════════════════════════════════════════════════════════
def _unit_to_nm(pixel_size: float, unit_str: str) -> float:
    u = str(unit_str).strip().replace("\x00", "").lower()
    conversions = {"m": 1e9, "mm": 1e6, "µm": 1e3, "um": 1e3, "micron": 1e3, "nm": 1.0, "å": 0.1, "a": 0.1, "angstrom": 0.1, "pm": 1e-3}
    return pixel_size * conversions.get(u, 1.0)

def _find_dimension_tags(tags: dict) -> list:
    found: list = []
    if isinstance(tags, dict):
        for key, value in tags.items():
            if key == "Dimension" and isinstance(value, dict): found.append(value)
            elif isinstance(value, dict): found.extend(_find_dimension_tags(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict): found.extend(_find_dimension_tags(item))
    return found

def try_read_dm3(file_bytes: bytes) -> TEMImage:
    if not _HAS_DM3: raise RuntimeError("dm3_lib is not installed.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dm3")
    try:
        tmp.write(file_bytes)
        tmp.close()
        dm3_file = pyDM3reader.DM3(tmp.name)
        data = np.array(dm3_file.imagedata, dtype=np.float64)
        pixel_size, pixel_unit = 0.0, ""
        
        if hasattr(dm3_file, "pxsize"):
            ps = dm3_file.pxsize
            if isinstance(ps, (list, tuple)) and len(ps) >= 2: pixel_size, pixel_unit = float(ps[0]), str(ps[1])
            elif isinstance(ps, (int, float)): pixel_size = float(ps)
        if pixel_size == 0 and hasattr(dm3_file, "tags"):
            for grp in _find_dimension_tags(dm3_file.tags):
                if "0" in grp and isinstance(grp["0"], dict) and float(grp["0"].get("Scale", 0)) > 0:
                    pixel_size, pixel_unit = float(grp["0"]["Scale"]), str(grp["0"].get("Units", ""))
                    break
        nm_per_px = _unit_to_nm(pixel_size, pixel_unit) if pixel_size > 0 else float("nan")
        return TEMImage(data, nm_per_px)
    finally:
        os.unlink(tmp.name)

def extract_pixel_size_nm(meta_str: str, default: float = float("nan")) -> float:
    def _find_pixelsize_in_obj(obj):
        if isinstance(obj, dict):
            if "PixelSize" in obj:
                ps = obj["PixelSize"]
                if isinstance(ps, dict):
                    vals = [float(v) for v in (ps.get("width"), ps.get("height")) if v is not None]
                    if vals: return (sum(vals) / len(vals)) * 1e9
            for v in obj.values():
                res = _find_pixelsize_in_obj(v)
                if res is not None: return res
        elif isinstance(obj, list):
            for v in obj:
                res = _find_pixelsize_in_obj(v)
                if res is not None: return res
        return None

    try:
        meta = json.loads(meta_str)
        val = _find_pixelsize_in_obj(meta)
        if val is not None: return val
    except Exception: pass

    for pattern in [
        r'"PixelSize"\s*:\s*\{[^}]*"width"\s*:\s*"([^"]+)"\s*,\s*"height"\s*:\s*"([^"]+)"[^}]*\}',
        r'"PixelSize"\s*:\s*\{[^}]*"width"\s*:\s*([0-9\.eE+-]+)\s*,\s*"height"\s*:\s*([0-9\.eE+-]+)[^}]*\}'
    ]:
        m = re.search(pattern, meta_str, re.IGNORECASE)
        if m:
            try: return ((float(m.group(1)) + float(m.group(2))) / 2.0) * 1e9
            except Exception: pass
    return default

def _search_hdf5_datasets(group, collected: list, depth: int = 0) -> None:
    if depth > 20: return
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset) and item.ndim >= 2: collected.append(item)
        elif isinstance(item, h5py.Group): _search_hdf5_datasets(item, collected, depth + 1)

def try_read_emd(file_bytes: bytes) -> TEMImage:
    if not _HAS_H5PY: raise RuntimeError("h5py is not installed.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".emd")
    try:
        tmp.write(file_bytes)
        tmp.close()
        with h5py.File(tmp.name, "r") as h5:
            if "Data" in h5 and "Image" in h5["Data"]:
                img_group = h5["Data"]["Image"]
                if img_group.keys():
                    dg = img_group[list(img_group.keys())[0]]
                    stack = dg["Data"][()]
                    data = stack[:, :, 0].astype(np.float64) if stack.ndim == 3 else stack.astype(np.float64) if stack.ndim == 2 else stack[0].astype(np.float64)
                    nm_px = float("nan")
                    if "Metadata" in dg: nm_px = extract_pixel_size_nm(dg["Metadata"][()].tobytes().decode("utf-8", errors="ignore"))
                    return TEMImage(data, nm_px)
            
            datasets = []
            _search_hdf5_datasets(h5, datasets)
            if not datasets: return TEMImage(np.zeros((1, 1), np.float64), float("nan"))
            best = max(datasets, key=lambda d: int(np.prod(d.shape[:2])))
            data = np.array(best, dtype=np.float64)
            if data.ndim == 3: data = np.mean(data[..., :3], axis=2) if data.shape[2] in (3, 4) else data[0]
            if data.ndim > 2: data = data.reshape(data.shape[0], data.shape[1])
            
            nm_px = float("nan")
            for attr in ("pixelSize", "PixelSize", "pixel_size"):
                if attr in best.attrs:
                    nm_px = _unit_to_nm(float(best.attrs[attr]), str(best.attrs.get(f"{attr}Unit", "m")))
                    break
            return TEMImage(data, nm_px)
    finally:
        os.unlink(tmp.name)

@st.cache_data(show_spinner=False)
def read_tem_file(file_bytes: bytes, filename: str) -> TEMImage:
    return try_read_dm3(file_bytes) if os.path.splitext(filename)[1].lower() == ".dm3" else try_read_emd(file_bytes)

# ═══════════════════════════════════════════════════════════════════════════
# ROI Selection
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def extract_high_contrast_roi(data: np.ndarray, nm_per_px: float, roi_size_nm: float) -> np.ndarray:
    roi_size_px = int(roi_size_nm) if np.isnan(nm_per_px) or nm_per_px <= 0 else int(roi_size_nm / nm_per_px)
    if roi_size_px >= min(data.shape[0], data.shape[1]): return data

    data_float = data.astype(np.float32)
    mean_sq = uniform_filter(data_float**2, size=roi_size_px, mode='reflect')
    mean = uniform_filter(data_float, size=roi_size_px, mode='reflect')
    variance = mean_sq - (mean**2)
    
    y, x = np.unravel_index(np.argmax(variance), variance.shape)
    half = roi_size_px // 2
    r_start = max(0, min(y - half, data.shape[0] - roi_size_px))
    c_start = max(0, min(x - half, data.shape[1] - roi_size_px))
    
    return data[r_start:r_start+roi_size_px, c_start:c_start+roi_size_px]

# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing Logic
# ═══════════════════════════════════════════════════════════════════════════
def preprocess_image(data: np.ndarray, smoothing: float, contrast_clip: float) -> np.ndarray:
    img = data.astype(np.float64)
    lap = laplace(img)
    noise_est = np.median(np.abs(lap)) * 1.4826
    if noise_est > 0:
        snr = (img.max() - img.min()) / (noise_est + 1e-12)
        if snr < 5: img = median_filter(img, size=5)
        elif snr < 15: img = median_filter(img, size=3)
    
    img = gaussian_filter(img, sigma=smoothing)
    lo, hi = np.percentile(img, [0.5, 99.5])
    img = np.clip((img - lo) / (hi - lo), 0, 1) if hi - lo > 0 else np.zeros_like(img)
    return exposure.equalize_adapthist(img, clip_limit=contrast_clip)

# ═══════════════════════════════════════════════════════════════════════════
# Shape classification
# ═══════════════════════════════════════════════════════════════════════════
SHAPE_CHOICES = ("Sphere", "Hexagonal Prism", "Cube", "Octahedron", "Tic Tac")
_COLORS = {
    "circle": "lime", "hexagon": "cyan", "rectangle": "yellow", 
    "square": "orange", "diamond": "magenta", "ellipse": "violet", "unknown": "gray"
}

def classify_projection(prop, target_shape: str) -> str:
    area, perim = float(prop.area), float(getattr(prop, "perimeter", 0.0)) or 1e-6
    circ = 4.0 * np.pi * area / perim ** 2
    solidity, extent = float(getattr(prop, "solidity", 0.0)), float(getattr(prop, "extent", 0.0))
    maj, minor = float(getattr(prop, "major_axis_length", 0.0)) or 1e-6, float(getattr(prop, "minor_axis_length", 0.0)) or 1e-6
    aspect = maj / minor

    if target_shape == "Sphere" and circ > 0.60: return "circle"
    elif target_shape == "Hexagonal Prism":
        if circ > 0.60 and solidity > 0.80: return "hexagon"
        if extent > 0.65 and solidity > 0.75 and aspect > 1.15: return "rectangle"
    elif target_shape == "Cube" and extent > 0.68 and solidity > 0.82: return "square"
    elif target_shape == "Octahedron" and solidity > 0.72: return "diamond"
    elif target_shape == "Tic Tac" and solidity > 0.85 and 1.1 <= aspect <= 4.0: return "ellipse"
    
    return "unknown"

def _hexagon_vertices(cx: float, cy: float, radius: float) -> np.ndarray:
    angles = np.pi / 3.0 * np.arange(6)
    return np.column_stack((cx + radius * np.cos(angles), cy + radius * np.sin(angles)))

def _diamond_vertices(cx: float, cy: float, half_maj: float, half_min: float, angle: float) -> np.ndarray:
    ca, sa = np.cos(angle), np.sin(angle)
    return np.array([[cx + half_maj * ca, cy + half_maj * sa], [cx - half_min * sa, cy + half_min * ca], [cx - half_maj * ca, cy - half_maj * sa], [cx + half_min * sa, cy - half_min * ca]])

def _rotated_rect_vertices(cx: float, cy: float, half_maj: float, half_min: float, angle: float) -> np.ndarray:
    ca, sa = np.cos(angle), np.sin(angle)
    corners = np.array([[half_maj, half_min], [-half_maj, half_min], [-half_maj, -half_min], [half_maj, -half_min]])
    xs = cx + corners[:, 0] * ca - corners[:, 1] * sa
    ys = cy + corners[:, 0] * sa + corners[:, 1] * ca
    return np.column_stack((xs, ys))

# ═══════════════════════════════════════════════════════════════════════════
# Cellpose Segmentation + Measurement
# ═══════════════════════════════════════════════════════════════════════════
def segment_and_measure(
    data: np.ndarray, nm_per_px: float, shape_type: str, min_size_value: float,
    measurement_unit: str, smoothing_sigma: float, contrast_clip: float,
    model_type: str, flow_threshold: float, cellprob_threshold: float
) -> Dict[str, Any]:
    
    preprocessed = preprocess_image(data, smoothing_sigma, contrast_clip)
    
    # 1. Invert image (Cellpose expects bright objects on dark background)
    inverted_img = preprocessed.max() - preprocessed
    
    min_diam_px = min_size_value / nm_per_px if measurement_unit == "nm" else min_size_value
    expected_diam_px = max(4.0, min_diam_px)
    
    # 2. Load model and predict
    model = load_cellpose_model(model_type)
    labels_ws, flows, styles = model.eval(
            inverted_img, 
            diameter=expected_diam_px, 
            channels=[0, 0],
            flow_threshold=flow_threshold, 
            cellprob_threshold=cellprob_threshold,
            resample=False,  # <--- SPEEDUP 1: Skip expensive upscaling
        )

    sf = nm_per_px if measurement_unit == "nm" else 1.0
    img_h, img_w = data.shape
    
    measurements = {
        "diameters": [], "hex_widths": [], "hex_heights": [], 
        "side_lengths": [], "oct_major": [], "oct_minor": [],
        "tic_tac_major": [], "tic_tac_minor": []
    }
    draw_shapes = []

    for prop in regionprops(labels_ws):
        minr, minc, maxr, maxc = prop.bbox
        if minr <= 2.0 or minc <= 2.0 or maxr >= img_h - 2.0 or maxc >= img_w - 2.0: continue

        cls = classify_projection(prop, shape_type)
        if cls == "unknown": continue

        cy, cx = prop.centroid
        orient = (np.pi / 2.0) - float(getattr(prop, "orientation", 0.0))
        color = _COLORS.get(cls, "white")
        maj_px, min_px = float(getattr(prop, "major_axis_length", 0.0)) or 0.0, float(getattr(prop, "minor_axis_length", 0.0)) or 0.0

        if cls == "circle":
            d_px = (maj_px + min_px) / 2.0
            d_val = d_px * sf
            if d_val >= min_size_value:
                measurements["diameters"].append(d_val)
                draw_shapes.append({"type": "circle", "cx": cx, "cy": cy, "r": d_px/2.0, "color": color})
        elif cls == "hexagon":
            d_px = (maj_px + min_px) / 2.0
            d_val = d_px * sf
            if d_val >= min_size_value:
                measurements["hex_widths"].append(d_val)
                draw_shapes.append({"type": "polygon", "vertices": _hexagon_vertices(cx, cy, d_px/2.0), "color": color})
        elif cls == "rectangle":
            if maj_px * sf >= min_size_value:
                measurements["hex_heights"].append(maj_px * sf)
                draw_shapes.append({"type": "polygon", "vertices": _rotated_rect_vertices(cx, cy, maj_px/2.0, min_px/2.0, orient), "color": color})
        elif cls == "square":
            side_px = ((maj_px + min_px) / 2.0) * 0.8660
            if side_px * sf >= min_size_value:
                measurements["side_lengths"].append(side_px * sf)
                draw_shapes.append({"type": "polygon", "vertices": _rotated_rect_vertices(cx, cy, side_px/2.0, side_px/2.0, orient), "color": color})
        elif cls == "diamond":
            maj_val, min_val = maj_px * 1.2247 * sf, min_px * 1.2247 * sf
            if maj_val >= min_size_value:
                measurements["oct_major"].append(maj_val)
                measurements["oct_minor"].append(min_val)
                draw_shapes.append({"type": "polygon", "vertices": _diamond_vertices(cx, cy, (maj_px*1.2247)/2.0, (min_px*1.2247)/2.0, orient), "color": color})
        elif cls == "ellipse":
            maj_val, min_val = maj_px * sf, min_px * sf
            if maj_val >= min_size_value:
                measurements["tic_tac_major"].append(maj_val)
                measurements["tic_tac_minor"].append(min_val)
                draw_shapes.append({
                    "type": "ellipse", "cx": cx, "cy": cy, 
                    "w": maj_px, "h": min_px, "angle": np.degrees(orient), "color": color
                })

    rand_cmap = np.random.RandomState(42).rand(max(int(labels_ws.max()) + 1, 1), 3)
    rand_cmap[0] = [1.0, 1.0, 1.0]
    
    out = {
        "data": data, "draw_shapes": draw_shapes, "rgb_ws": (rand_cmap[labels_ws] * 255).astype(np.uint8),
        "unit": measurement_unit, "nm_per_px": float(nm_per_px) if measurement_unit == "nm" else float("nan")
    }
    for k, v in measurements.items(): out[k] = np.array(v, dtype=np.float64)
    return out

# ═══════════════════════════════════════════════════════════════════════════
# Drawing Helper
# ═══════════════════════════════════════════════════════════════════════════
def plot_annotated_image(ax, image_data, draw_shapes):
    zmin, zmax = np.percentile(image_data, [0.1, 99.9])
    ax.imshow(image_data, cmap="gray", vmin=zmin, vmax=zmax)
    for s in draw_shapes:
        if s["type"] == "circle":
            ax.add_patch(patches.Circle((s["cx"], s["cy"]), s["r"], edgecolor=s["color"], facecolor=s["color"], alpha=0.4, lw=2))
        elif s["type"] == "polygon":
            ax.add_patch(patches.Polygon(s["vertices"], closed=True, edgecolor=s["color"], facecolor=s["color"], alpha=0.4, lw=2))
        elif s["type"] == "ellipse":
            ax.add_patch(patches.Ellipse((s["cx"], s["cy"]), s["w"], s["h"], angle=s["angle"], edgecolor=s["color"], facecolor=s["color"], alpha=0.4, lw=2))
    ax.axis("off")

# ═══════════════════════════════════════════════════════════════════════════
# Caching Full Batch Run
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_batch_analysis(file_data: List[Tuple[bytes, str]], shape_type: str, params: dict) -> List[Dict[str, Any]]:
    results = []
    for f_bytes, f_name in file_data:
        tem = read_tem_file(f_bytes, f_name)
        unit_full = "nm" if not np.isnan(tem.nm_per_px) else "px"
        seg = segment_and_measure(
            data=tem.data, nm_per_px=tem.nm_per_px, shape_type=shape_type, 
            min_size_value=params["min_feature"], measurement_unit=unit_full, 
            smoothing_sigma=params["smoothing_sigma"], contrast_clip=params["contrast_clip"],
            model_type=params["model_type"], flow_threshold=params["flow_threshold"],
            cellprob_threshold=params["cellprob_threshold"]
        )
        seg["name"] = f_name
        results.append(seg)
    return results

# ═══════════════════════════════════════════════════════════════════════════
# Histograms
# ═══════════════════════════════════════════════════════════════════════════
def histogram_with_fit(values: np.ndarray, title: str, unit_label: str) -> Tuple[plt.Figure, float, float, int]:
    n_fit = len(values)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    mu, std = float("nan"), float("nan")

    if n_fit > 0:
        vis_min, vis_max = np.percentile(values, [0.5, 99.5])
        if vis_min >= vis_max: vis_min, vis_max = values.min(), values.max()
        
        counts, bins, _ = ax.hist(values, bins=max(10, int((vis_max - vis_min) / ((vis_max - vis_min) / 40.0)) if vis_max > vis_min else 10), color='steelblue', edgecolor='black', alpha=0.6)
        bw = float(bins[1] - bins[0]) if len(bins) > 1 else 1.0
        
        if n_fit >= 3:
            gmm = GaussianMixture(n_components=1, random_state=42).fit(values.reshape(-1, 1))
            mu, std = float(gmm.means_[0][0]), float(np.sqrt(gmm.covariances_[0][0][0]))
            x_fit = np.linspace(vis_min, vis_max, 300).reshape(-1, 1)
            pdf = np.exp(gmm.score_samples(x_fit))
            ax.plot(x_fit, pdf * n_fit * bw, 'k', linewidth=1.5, label='Fit')
            ax.set_title(f"{title} (n={n_fit})\nDominant μ={mu:.2f} ± {std:.2f} {unit_label}", fontsize=14)
        else:
            ax.set_title(f"{title} (n={n_fit})", fontsize=14)
        ax.set_xlim(vis_min, vis_max)
    else:
        ax.set_title(f"{title} (n=0)", fontsize=14)

    ax.set_xlabel(f"Size ({unit_label})", color='black', weight='bold')
    ax.set_ylabel("Count", color='black', weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig, float(mu), float(std), n_fit

def _common_prefix(names: List[str]) -> str:
    if not names: return "analysis"
    shortest = min(names, key=len)
    for i, ch in enumerate(shortest):
        if any(name[i] != ch for name in names): return shortest[:i].rstrip(" _-.") or "analysis"
    return shortest.rstrip(" _-.") or "analysis"

def _download_row(fig: plt.Figure, df: pd.DataFrame, stem: str) -> None:
    c1, c2 = st.columns(2)
    with c1:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        st.download_button("Download PNG", buf.getvalue(), f"{stem}.png", "image/png")
    with c2:
        st.download_button("Download CSV", df.to_csv(index=False), f"{stem}.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# Main Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════
def run() -> None:
    st.set_page_config(page_title="TEM Particle Analysis", layout="wide")
    st.title("TEM Particle Characterization (Cellpose)")
    
    if "full_run_complete" not in st.session_state:
        st.session_state.full_run_complete = False

    with st.sidebar:
        st.header("1. Upload and Setup")
        accepted_types = ["dm3"] if _HAS_DM3 else []
        if _HAS_H5PY: accepted_types.append("emd")
        
        files = st.file_uploader("Upload TEM image(s)", accept_multiple_files=True, type=accepted_types, on_change=_clear_cache)
        shape_type = st.selectbox("Particle shape", list(SHAPE_CHOICES), index=0)
        
        if not files:
            st.warning("Upload files to begin.")
            return

        tune_file_name = st.selectbox("Select image for tuning:", [f.name for f in files])
        tune_file = next(f for f in files if f.name == tune_file_name)

    tab_tune, tab_results = st.tabs(["Rapid Tuning (ROI)", "Full Batch Results"])
    
    tem_tune = read_tem_file(tune_file.getvalue(), tune_file.name)
    calibrated = not np.isnan(tem_tune.nm_per_px)
    unit = "nm" if calibrated else "px"

    with tab_tune:
        st.subheader("Step 2: Tune Deep Learning Parameters")
        
        c_roi, c_size, c_flow, c_prob = st.columns(4)
        with c_roi:
            roi_size = st.number_input(f"ROI Size ({unit})", min_value=50, max_value=2000, value=200, step=50)
        with c_size:
            min_feature = st.number_input(f"Min feature size ({unit})", min_value=0.0, value=5.0, step=0.5)
        with c_flow:
            flow_threshold = st.slider("Flow Threshold", 0.0, 1.0, 0.4, 0.1, help="Increase to find more ROIs. Decrease to strictly separate objects.")
        with c_prob:
            cellprob_threshold = st.slider("Probability Threshold", -6.0, 6.0, 0.0, 0.5, help="Decrease to find more pixels per object. Increase to restrict object boundaries.")

        roi_data = extract_high_contrast_roi(tem_tune.data, tem_tune.nm_per_px, roi_size)

        with st.expander("Advanced Settings", expanded=False):
            adv_c1, adv_c2, adv_c3 = st.columns(3)
            with adv_c1:
                smoothing_sigma = st.slider("Image Smoothing (Sigma)", 0.5, 5.0, 1.5, 0.1, help="Blurs image to reduce background noise before feeding to Cellpose.")
            with adv_c2:
                contrast_clip = st.slider("Contrast Enhancement", 0.01, 0.10, 0.01, 0.01, help="Enhances contrast before feeding to Cellpose.")
            with adv_c3:
                model_type = st.selectbox("Cellpose Model Type", ["cyto", "cyto2", "cyto3"], index=0, help="Change the internal neural network model used.")

        seg_roi = segment_and_measure(
            data=roi_data, nm_per_px=tem_tune.nm_per_px, shape_type=shape_type, 
            min_size_value=min_feature, measurement_unit=unit, 
            smoothing_sigma=smoothing_sigma, contrast_clip=contrast_clip,
            model_type=model_type, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold
        )

        plot_c1, plot_c2 = st.columns(2) 
        with plot_c1:
            st.caption("Fitted Shapes (ROI)")
            fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
            plot_annotated_image(ax, seg_roi["data"], seg_roi["draw_shapes"])
            fig.tight_layout(pad=0)
            st.pyplot(fig)
            plt.close(fig)
        with plot_c2:
            st.caption("Cellpose Mask (ROI)")
            st.image(seg_roi["rgb_ws"], use_container_width=True)

        if st.button("Apply to All Images and Run Full Analysis", type="primary"):
            st.session_state.full_run_complete = True
            st.session_state.run_params = {
                "min_feature": min_feature,
                "flow_threshold": flow_threshold,
                "cellprob_threshold": cellprob_threshold,
                "smoothing_sigma": smoothing_sigma,
                "contrast_clip": contrast_clip,
                "model_type": model_type
            }
            st.success("Parameters locked in. Please open the 'Full Batch Results' tab to view your processed data.")

    with tab_results:
        if not st.session_state.get("full_run_complete", False):
            st.info("Find optimal parameters in the 'Rapid Tuning' tab, then click 'Apply to All' to generate full histograms.")
        else:
            with st.spinner("Processing full images with Cellpose..."):
                file_data = [(f.getvalue(), f.name) for f in files]
                results = run_batch_analysis(file_data, shape_type, st.session_state.run_params)
            
            st.success(f"Successfully processed {len(results)} image(s).")
            
            st.markdown("### Processed Image Verification")
            selected_full = st.selectbox("Select image to display", [r["name"] for r in results])
            sel_res = next(r for r in results if r["name"] == selected_full)
            
            img_c1, img_c2 = st.columns(2)
            with img_c1:
                st.caption(f"Annotated Full Image ({sel_res['unit']})")
                fig_full, ax_full = plt.subplots(figsize=(8, 8), dpi=150)
                plot_annotated_image(ax_full, sel_res["data"], sel_res["draw_shapes"])
                fig_full.tight_layout(pad=0)
                st.pyplot(fig_full)
                plt.close(fig_full)
            with img_c2:
                st.caption("Cellpose Mask Full Image")
                st.image(sel_res["rgb_ws"], use_container_width=True)

            st.markdown("---")
            st.markdown("### Histograms")
            
            prefix = _common_prefix([r["name"] for r in results])
            unit_full = results[0]["unit"] if results else "nm"

            if shape_type == "Sphere":
                all_d = np.concatenate([r["diameters"] for r in results])
                if all_d.size > 0:
                    fig, mu, std, n = histogram_with_fit(all_d, f"{prefix} — Diameter", unit_full)
                    st.pyplot(fig, use_container_width=True)
                    _download_row(fig, pd.DataFrame({"diameter": all_d}), f"{prefix}_diameter")
                else: st.info("No spherical particles detected.")
                
            elif shape_type == "Hexagonal Prism":
                all_w, all_h = np.concatenate([r["hex_widths"] for r in results]), np.concatenate([r["hex_heights"] for r in results])
                if all_w.size > 0:
                    fig_w, mu, std, n = histogram_with_fit(all_w, f"{prefix} — Hex width (face-on)", unit_full)
                    st.pyplot(fig_w, use_container_width=True)
                    _download_row(fig_w, pd.DataFrame({"hex_width": all_w}), f"{prefix}_hex_width")
                if all_h.size > 0:
                    fig_h, mu, std, n = histogram_with_fit(all_h, f"{prefix} — Height (side-on)", unit_full)
                    st.pyplot(fig_h, use_container_width=True)
                    _download_row(fig_h, pd.DataFrame({"hex_height": all_h}), f"{prefix}_hex_height")

            elif shape_type == "Cube":
                all_s = np.concatenate([r["side_lengths"] for r in results])
                if all_s.size > 0:
                    fig, mu, std, n = histogram_with_fit(all_s, f"{prefix} — Cube side length", unit_full)
                    st.pyplot(fig, use_container_width=True)
                    _download_row(fig, pd.DataFrame({"side_length": all_s}), f"{prefix}_side_length")
                else: st.info("No cubic particles detected.")

            elif shape_type == "Octahedron":
                all_maj, all_min = np.concatenate([r["oct_major"] for r in results]), np.concatenate([r["oct_minor"] for r in results])
                if all_maj.size > 0 or all_min.size > 0:
                    all_axes = np.concatenate([all_maj, all_min])
                    fig, mu, std, n = histogram_with_fit(all_axes, f"{prefix} — Octahedron Combined Axes", unit_full)
                    st.pyplot(fig, use_container_width=True)
                    _download_row(fig, pd.DataFrame({"combined_axes": all_axes}), f"{prefix}_octahedron")
                else: st.info("No octahedral particles detected.")
                
            elif shape_type == "Tic Tac":
                all_maj, all_min = np.concatenate([r["tic_tac_major"] for r in results]), np.concatenate([r["tic_tac_minor"] for r in results])
                if all_maj.size > 0:
                    fig_maj, mu, std, n = histogram_with_fit(all_maj, f"{prefix} — Tic Tac Major Axis", unit_full)
                    st.pyplot(fig_maj, use_container_width=True)
                    _download_row(fig_maj, pd.DataFrame({"tic_tac_major": all_maj}), f"{prefix}_tictac_major")
                if all_min.size > 0:
                    fig_min, mu, std, n = histogram_with_fit(all_min, f"{prefix} — Tic Tac Minor Axis", unit_full)
                    st.pyplot(fig_min, use_container_width=True)
                    _download_row(fig_min, pd.DataFrame({"tic_tac_minor": all_min}), f"{prefix}_tictac_minor")
                if all_maj.size == 0 and all_min.size == 0:
                    st.info("No Tic Tac particles detected.")
            
if __name__ == "__main__":
    run()