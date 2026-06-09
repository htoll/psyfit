"""Interactive TEM particle analysis tool (Pure Computer Vision).

Main features
─────────────
* Upload .dm3 or .emd (Velox HDF5) images.
* RAPID TUNING: Auto-extracts high-contrast ROI for instant parameter feedback.
* CV Segmentation: Uses Otsu thresholding, Distance Transforms, and Watershed for ultra-fast shape detection.
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
from scipy.ndimage import gaussian_filter, laplace, median_filter, uniform_filter, distance_transform_edt, label
from scipy.stats import norm

from skimage import exposure
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import disk, opening
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

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
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass # Windows or Google Drive is holding the file; let the OS clean it up later.

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
    solidity = float(getattr(prop, "solidity", 0.0))
    maj = float(getattr(prop, "major_axis_length", 0.0)) or 1e-6
    minor = float(getattr(prop, "minor_axis_length", 0.0)) or 1e-6
    aspect = maj / minor

    if target_shape == "Sphere" and circ > 0.60: return "circle"
    elif target_shape == "Hexagonal Prism":
        # Removed 'extent' check. Relying purely on aspect ratio and solidity!
        if solidity > 0.75 and aspect > 1.15: return "rectangle"
        elif circ > 0.60 and solidity > 0.80: return "hexagon"
    elif target_shape == "Cube" and solidity > 0.82: return "square"
    elif target_shape == "Octahedron" and solidity > 0.72: return "diamond"
    elif target_shape == "Tic Tac" and solidity > 0.85 and 1.1 <= aspect <= 4.0: return "ellipse"
    
    return "unknown"

def _hexagon_vertices(cx: float, cy: float, radius: float, angle: float = 0.0) -> np.ndarray:
    # Add the rotation angle to the base angles
    angles = (np.pi / 3.0 * np.arange(6)) + angle
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
# Fast Computer Vision Segmentation
# ═══════════════════════════════════════════════════════════════════════════
def segment_and_measure(
    data: np.ndarray, nm_per_px: float, shape_type: str, min_size_value: float,
    measurement_unit: str, smoothing_sigma: float, contrast_clip: float,
    thresh_offset: float, min_peak_distance: int
) -> Dict[str, Any]:
    
    # 1. Preprocess the image
    preprocessed = preprocess_image(data, smoothing_sigma, contrast_clip)
    
    # TEM particles are usually dark on a lighter background. 
    # We invert it so particles are bright (closer to 1.0)
    inverted_img = preprocessed.max() - preprocessed
    
    # 2. Otsu Thresholding to create a binary mask
    # We add a slight offset multiplier to allow the user to make the mask stricter or looser
    global_thresh = threshold_otsu(inverted_img) * thresh_offset
    binary_mask = inverted_img > global_thresh
    
    # 3. Morphological opening to remove tiny noise specs (salt noise)
    binary_mask = opening(binary_mask, disk(2))
    
    # 4. Euclidean Distance Transform & Watersheding
    # This separates clumped particles by finding their distinct centers
    # 4. Euclidean Distance Transform & Watersheding
    distance = distance_transform_edt(binary_mask)
    
    # --- NEW: Smooth the distance map to fix fragmented rectangles ---
    # Rectangles create flat "ridges" that cause multiple peaks.
    # Blurring it slightly forces a single peak in the absolute center.
    distance_smoothed = gaussian_filter(distance, sigma=2.0)
    
    # Find peaks on the *smoothed* distance map
    coords = peak_local_max(distance_smoothed, min_distance=min_peak_distance, labels=binary_mask)
    
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = label(mask)
    
    # Apply watershed
    labels_ws = watershed(-distance, markers, mask=binary_mask)

    # 5. Extract Measurements
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
        # Skip objects touching the very edge
        if minr <= 2.0 or minc <= 2.0 or maxr >= img_h - 2.0 or maxc >= img_w - 2.0: continue

        cls = classify_projection(prop, shape_type)
        if cls == "unknown": continue

        cy, cx = prop.centroid
        orient = (np.pi / 2.0) - float(getattr(prop, "orientation", 0.0))
        color = _COLORS.get(cls, "white")
        maj_px = float(getattr(prop, "major_axis_length", 0.0)) or 0.0
        min_px = float(getattr(prop, "minor_axis_length", 0.0)) or 0.0

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
                # Pass the 'orient' variable as the 4th argument here!
                draw_shapes.append({
                    "type": "polygon", 
                    "vertices": _hexagon_vertices(cx, cy, d_px/2.0, orient), 
                    "color": color
                })
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

    # Generate random colored mask for visualization
    rand_cmap = np.random.RandomState(42).rand(max(int(labels_ws.max()) + 1, 1), 3)
    rand_cmap[0] = [1.0, 1.0, 1.0] # Background is white
    
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
            thresh_offset=params["thresh_offset"], min_peak_distance=params["min_peak_distance"]
        )
        seg["name"] = f_name
        results.append(seg)
    return results

# ═══════════════════════════════════════════════════════════════════════════
# Histograms
# ═══════════════════════════════════════════════════════════════════════════
def histogram_with_fit(values: np.ndarray, title: str, unit_label: str, fit_min: float, fit_max: float) -> Tuple[plt.Figure, float, float, int]:
    # Filter values based on user-defined range
    filtered_vals = values[(values >= fit_min) & (values <= fit_max)]
    n_fit = len(filtered_vals)
    
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    mu, std = 0.0, 0.0

    if n_fit > 1:
        # Dynamic binning based on data range and size
        bins = max(10, int(min(50, n_fit / 3)))
        counts, bin_edges, _ = ax.hist(filtered_vals, bins=bins, color='steelblue', edgecolor='black', alpha=0.6)
        bw = float(bin_edges[1] - bin_edges[0])
        
        # Single Gaussian fit parameters
        mu, std = np.mean(filtered_vals), np.std(filtered_vals)
        
        if std > 0:
            x_fit = np.linspace(fit_min, fit_max, 300)
            pdf = norm.pdf(x_fit, mu, std)
            ax.plot(x_fit, pdf * n_fit * bw, 'k', linewidth=2.0, label='Gaussian Fit')

        ax.set_title(f"{title} (n={n_fit})\nμ = {mu:.2f} ± {std:.2f} {unit_label}", fontsize=14)
    else:
        if n_fit == 1:
            ax.hist(filtered_vals, bins=1, color='steelblue', edgecolor='black', alpha=0.6)
            mu = filtered_vals[0]
        ax.set_title(f"{title} (n={n_fit})", fontsize=14)

    # Lock the plot to the user's defined bounds
    if fit_min < fit_max:
        ax.set_xlim(fit_min, fit_max)

    ax.set_xlabel(f"Size ({unit_label})", color='black', weight='bold')
    ax.set_ylabel("Count", color='black', weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    
    return fig, mu, std, n_fit

def get_min_max_ui(key_prefix: str, label: str, data: np.ndarray) -> Tuple[float, float]:
    """Generates UI columns for min/max filtering."""
    if data.size == 0:
        return 0.0, 1.0
    d_min, d_max = float(np.min(data)), float(np.max(data))
    if d_min == d_max:
        d_max = d_min + 1.0  # Prevent min == max collapse
        
    c1, c2 = st.columns(2)
    with c1:
        fit_min = st.number_input(f"Min {label}", value=d_min, key=f"{key_prefix}_min")
    with c2:
        fit_max = st.number_input(f"Max {label}", value=d_max, key=f"{key_prefix}_max")
    return fit_min, fit_max

def _common_prefix(names: List[str]) -> str:
    if not names: return "Analysis"
    
    # If there's only one file, use its full root name
    if len(names) == 1:
        return os.path.splitext(names[0])[0]
        
    shortest = min(names, key=len)
    prefix = shortest
    for i, ch in enumerate(shortest):
        if any(name[i] != ch for name in names): 
            prefix = shortest[:i]
            break

    prefix = prefix.rstrip(" _-.")
    
    # If the common prefix is too short or doesn't exist, use the first file's name as a base
    if len(prefix) < 3:
        base = os.path.splitext(names[0])[0]
        return f"{base} (+{len(names)-1} files)"
        
    return prefix

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
    #st.set_page_config(page_title="TEM Particle Analysis", layout="wide")
    st.title("TEM Particle Characterization (Fast CV)")
    
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
        st.subheader("Step 2: Tune Watershed Parameters")
        
        c_roi, c_size, c_thresh, c_peak = st.columns(4)
        with c_roi:
            roi_size = st.number_input(f"ROI Size ({unit})", min_value=50, max_value=2000, value=200, step=50)
        with c_size:
            min_feature = st.number_input(f"Min feature size ({unit})", min_value=0.0, value=5.0, step=0.5)
        with c_thresh:
            thresh_offset = st.slider("Threshold Sensitivity", 0.5, 1.5, 1.0, 0.05, help="Low = Picks up faint particles. High = Stricter boundaries.")
        with c_peak:
            min_peak_distance = st.number_input("Min peak distance (px)", min_value=1, value=5, help="Increase if particles are being overly fragmented. Decrease if touching particles aren't splitting.")

        roi_data = extract_high_contrast_roi(tem_tune.data, tem_tune.nm_per_px, roi_size)

        with st.expander("Advanced Settings", expanded=False):
            adv_c1, adv_c2 = st.columns(2)
            with adv_c1:
                smoothing_sigma = st.slider("Image Smoothing (Sigma)", 0.5, 5.0, 1.5, 0.1, help="Blurs image to reduce background noise.")
            with adv_c2:
                contrast_clip = st.slider("Contrast Enhancement", 0.01, 0.10, 0.01, 0.01, help="Enhances contrast to separate particles from the background.")

        seg_roi = segment_and_measure(
            data=roi_data, nm_per_px=tem_tune.nm_per_px, shape_type=shape_type, 
            min_size_value=min_feature, measurement_unit=unit, 
            smoothing_sigma=smoothing_sigma, contrast_clip=contrast_clip,
            thresh_offset=thresh_offset, min_peak_distance=min_peak_distance
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
            st.caption("Watershed Mask (ROI)")
            st.image(seg_roi["rgb_ws"], use_container_width=True)

        if st.button("Apply to All Images and Run Full Analysis", type="primary"):
            st.session_state.full_run_complete = True
            st.session_state.run_params = {
                "min_feature": min_feature,
                "thresh_offset": thresh_offset,
                "min_peak_distance": min_peak_distance,
                "smoothing_sigma": smoothing_sigma,
                "contrast_clip": contrast_clip,
            }
            st.success("Parameters locked in. Please open the 'Full Batch Results' tab to view your processed data.")

    with tab_results:
        if not st.session_state.get("full_run_complete", False):
            st.info("Find optimal parameters in the 'Rapid Tuning' tab, then click 'Apply to All' to generate full histograms.")
        else:
            with st.spinner("Processing full images with Watershed..."):
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
                st.caption("Watershed Mask Full Image")
                st.image(sel_res["rgb_ws"], use_container_width=True)

            st.markdown("---")
            st.markdown("### Histograms & Effective Radius ($r_{eff}$)")
            
            prefix = _common_prefix([r["name"] for r in results])
            unit_full = results[0]["unit"] if results else "nm"

            def calc_reff(volume: float) -> float:
                """Helper to convert volume to effective radius."""
                return ((3.0 * volume) / (4.0 * np.pi)) ** (1.0 / 3.0)
            if shape_type == "Sphere":
                all_d = np.concatenate([r["diameters"] for r in results])
                if all_d.size > 0:
                    fit_min, fit_max = get_min_max_ui("sph_d", f"Diameter ({unit_full})", all_d)
                    fig, mu, std, n = histogram_with_fit(all_d, f"{prefix} — Diameter", unit_full, fit_min, fit_max)
                    st.pyplot(fig, use_container_width=True)
                    _download_row(fig, pd.DataFrame({"diameter": all_d[(all_d >= fit_min) & (all_d <= fit_max)]}), f"{prefix}_diameter")
                    
                    if n > 0:
                        st.metric(label=f"Effective Radius (r_eff)", value=f"{(mu / 2.0):.2f} {unit_full}")
                else: 
                    st.info("No spherical particles detected.")
                
            elif shape_type == "Hexagonal Prism":
                all_w = np.concatenate([r["hex_widths"] for r in results])
                all_h = np.concatenate([r["hex_heights"] for r in results])
                
                mu_w, mu_h = 0.0, 0.0
                n_w, n_h = 0, 0
                
                if all_w.size > 0:
                    w_min, w_max = get_min_max_ui("hex_w", f"Width ({unit_full})", all_w)
                    fig_w, mu_w, std_w, n_w = histogram_with_fit(all_w, f"{prefix} — Hex width (face-on)", unit_full, w_min, w_max)
                    st.pyplot(fig_w, use_container_width=True)
                    _download_row(fig_w, pd.DataFrame({"hex_width": all_w}), f"{prefix}_hex_width")
                    
                st.markdown("---")
                    
                if all_h.size > 0:
                    h_min, h_max = get_min_max_ui("hex_h", f"Height ({unit_full})", all_h)
                    fig_h, mu_h, std_h, n_h = histogram_with_fit(all_h, f"{prefix} — Height (side-on)", unit_full, h_min, h_max)
                    st.pyplot(fig_h, use_container_width=True)
                    _download_row(fig_h, pd.DataFrame({"hex_height": all_h}), f"{prefix}_hex_height")
                    
                if n_w > 0 and n_h > 0:
                    v_hex = (np.sqrt(3) / 2.0) * (mu_w ** 2) * mu_h
                    st.metric(label=f"Effective Radius (r_eff)", value=f"{calc_reff(v_hex):.2f} {unit_full}", help=f"Calculated from fitted width μ ({mu_w:.2f}) and height μ ({mu_h:.2f})")
                else:
                    st.warning("Both face-on (width) and side-on (height) measurements are needed to calculate r_eff.")

            elif shape_type == "Cube":
                all_s = np.concatenate([r["side_lengths"] for r in results])
                if all_s.size > 0:
                    s_min, s_max = get_min_max_ui("cube_s", f"Side Length ({unit_full})", all_s)
                    fig, mu, std, n = histogram_with_fit(all_s, f"{prefix} — Cube side length", unit_full, s_min, s_max)
                    st.pyplot(fig, use_container_width=True)
                    _download_row(fig, pd.DataFrame({"side_length": all_s}), f"{prefix}_side_length")
                    
                    if n > 0:
                        st.metric(label=f"Effective Radius (r_eff)", value=f"{calc_reff(mu ** 3):.2f} {unit_full}")
                else: 
                    st.info("No cubic particles detected.")

            elif shape_type == "Octahedron":
                all_maj = np.concatenate([r["oct_major"] for r in results])
                all_min = np.concatenate([r["oct_minor"] for r in results])
                
                mu_maj, mu_min = 0.0, 0.0
                n_maj, n_min = 0, 0
                
                if all_maj.size > 0:
                    st.markdown("#### Major Axis")
                    maj_min, maj_max = get_min_max_ui("oct_maj", f"Major Axis ({unit_full})", all_maj)
                    fig_maj, mu_maj, std_maj, n_maj = histogram_with_fit(all_maj, f"{prefix} — Octahedron Major", unit_full, maj_min, maj_max)
                    st.pyplot(fig_maj, use_container_width=True)
                    _download_row(fig_maj, pd.DataFrame({"oct_major": all_maj}), f"{prefix}_oct_major")
                    
                st.markdown("---")
                    
                if all_min.size > 0:
                    st.markdown("#### Minor Axis")
                    min_min, min_max = get_min_max_ui("oct_min", f"Minor Axis ({unit_full})", all_min)
                    fig_min, mu_min, std_min, n_min = histogram_with_fit(all_min, f"{prefix} — Octahedron Minor", unit_full, min_min, min_max)
                    st.pyplot(fig_min, use_container_width=True)
                    _download_row(fig_min, pd.DataFrame({"oct_minor": all_min}), f"{prefix}_oct_minor")
                    
                if n_maj > 0 and n_min > 0:
                    v_oct = (np.sqrt(2) / 3.0) * mu_maj * (mu_min ** 2)
                    st.metric(label=f"Effective Radius (r_eff)", value=f"{calc_reff(v_oct):.2f} {unit_full}", help=f"Calculated from fitted major μ ({mu_maj:.2f}) and minor μ ({mu_min:.2f})")
                else: 
                    st.info("Missing dimensions required to calculate r_eff.")
                
            elif shape_type == "Tic Tac":
                all_maj = np.concatenate([r["tic_tac_major"] for r in results])
                all_min = np.concatenate([r["tic_tac_minor"] for r in results])
                
                mu_maj, mu_min = 0.0, 0.0
                n_maj, n_min = 0, 0
                
                if all_maj.size > 0:
                    maj_min, maj_max = get_min_max_ui("tic_maj", f"Major Axis ({unit_full})", all_maj)
                    fig_maj, mu_maj, std_maj, n_maj = histogram_with_fit(all_maj, f"{prefix} — Tic Tac Major", unit_full, maj_min, maj_max)
                    st.pyplot(fig_maj, use_container_width=True)
                    _download_row(fig_maj, pd.DataFrame({"tic_tac_major": all_maj}), f"{prefix}_tictac_major")
                    
                st.markdown("---")
                    
                if all_min.size > 0:
                    min_min, min_max = get_min_max_ui("tic_min", f"Minor Axis ({unit_full})", all_min)
                    fig_min, mu_min, std_min, n_min = histogram_with_fit(all_min, f"{prefix} — Tic Tac Minor", unit_full, min_min, min_max)
                    st.pyplot(fig_min, use_container_width=True)
                    _download_row(fig_min, pd.DataFrame({"tic_tac_minor": all_min}), f"{prefix}_tictac_minor")
                    
                if n_maj > 0 and n_min > 0:
                    a = mu_maj / 2.0
                    b = mu_min / 2.0
                    v_tictac = (4.0 / 3.0) * np.pi * a * (b ** 2)
                    st.metric(label=f"Effective Radius (r_eff)", value=f"{calc_reff(v_tictac):.2f} {unit_full}", help=f"Calculated from fitted major μ ({mu_maj:.2f}) and minor μ ({mu_min:.2f})")
                else:
                    st.warning("Both major and minor axes are needed to calculate r_eff.")

if __name__ == "__main__":
    run()
