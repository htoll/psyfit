"""Interactive TEM particle analysis tool.

This Streamlit app reads `.dm3` and `.emd` files and measures particle sizes.
It supports spherical, hexagonal-prism, cubic and octahedral nanoparticles and
provides visual checks of the segmentation quality.

Main features
─────────────
* Upload `.dm3` **or** `.emd` (Velox HDF5) images.
* Robust adaptive pre-processing before watershed segmentation.
* Shape-aware 2-D projection classification.
* Fast static overlay of detected shapes on the original grayscale image.
* Size-distribution histograms with Gaussian fits, SD, and n.
* Combined CSV / PNG export.
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
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    laplace,
    median_filter,
)
from scipy.stats import norm
from skimage import exposure
from skimage.feature import peak_local_max
from skimage.filters import threshold_li, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_holes,
    remove_small_objects,
)
from skimage.segmentation import watershed
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

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
# Data structures
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class TEMImage:
    """Container for a single TEM image with calibration."""
    data: np.ndarray
    nm_per_px: float  # NaN when calibration is missing

def _clear_cache():
        st.cache_data.clear()

# ═══════════════════════════════════════════════════════════════════════════
# Unit conversion
# ═══════════════════════════════════════════════════════════════════════════
def _unit_to_nm(pixel_size: float, unit_str: str) -> float:
    """Convert *pixel_size* in arbitrary *unit_str* to nanometres."""
    u = str(unit_str).strip().replace("\x00", "").lower()
    conversions = {
        "m": 1e9, "mm": 1e6,
        "µm": 1e3, "um": 1e3, "micron": 1e3, "microns": 1e3,
        "nm": 1.0,
        "å": 0.1, "a": 0.1, "angstrom": 0.1,
        "pm": 1e-3,
    }
    factor = conversions.get(u, None)
    if factor is not None:
        return pixel_size * factor
    return pixel_size  # assume nm if unknown

# ═══════════════════════════════════════════════════════════════════════════
# File reading — DM3
# ═══════════════════════════════════════════════════════════════════════════
def _find_dimension_tags(tags: dict) -> list:
    found: list = []
    if isinstance(tags, dict):
        for key, value in tags.items():
            if key == "Dimension" and isinstance(value, dict):
                found.append(value)
            elif isinstance(value, dict):
                found.extend(_find_dimension_tags(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        found.extend(_find_dimension_tags(item))
    return found


def try_read_dm3(file_bytes: bytes) -> TEMImage:
    if not _HAS_DM3:
        raise RuntimeError("dm3_lib is not installed.  pip install dm3_lib")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dm3")
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()

        dm3_file = pyDM3reader.DM3(tmp.name)
        data = np.array(dm3_file.imagedata, dtype=np.float64)

        pixel_size: float = 0.0
        pixel_unit: str = ""

        # 1. Check direct pxsize attribute
        if hasattr(dm3_file, "pxsize"):
            try:
                ps = dm3_file.pxsize
                if isinstance(ps, (list, tuple)) and len(ps) >= 2:
                    pixel_size, pixel_unit = float(ps[0]), str(ps[1])
                elif isinstance(ps, (int, float)):
                    pixel_size = float(ps)
            except Exception:
                pass

        # 2. Check dimension tags for Scale and Units
        if pixel_size == 0 and hasattr(dm3_file, "tags"):
            for grp in _find_dimension_tags(dm3_file.tags):
                if "0" in grp and isinstance(grp["0"], dict):
                    dim = grp["0"]
                    scale = float(dim.get("Scale", 0))
                    units = str(dim.get("Units", ""))
                    if scale > 0:
                        pixel_size, pixel_unit = scale, units
                        break

        # 3. Check legacy "Pixel Size (um)" tag
        if pixel_size == 0 and hasattr(dm3_file, "tags"):
            rt = dm3_file.tags
            if isinstance(rt, dict) and "ImageTags" in rt:
                it = rt["ImageTags"]
                if isinstance(it, dict) and "Pixel Size (um)" in it:
                    pixel_size = float(it["Pixel Size (um)"])
                    pixel_unit = "µm"

        nm_per_px = _unit_to_nm(pixel_size, pixel_unit) if pixel_size > 0 else float("nan")
        return TEMImage(data=data, nm_per_px=nm_per_px)
        
    except Exception as exc:
        st.warning(f"DM3 read error: {exc}")
        return TEMImage(data=np.zeros((1, 1), dtype=np.float64), nm_per_px=float("nan"))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
# ═══════════════════════════════════════════════════════════════════════════
# File reading — DM3
# ═══════════════════════════════════════════════════════════════════════════
def _find_dimension_tags(tags: dict) -> list:
    found: list = []
    if isinstance(tags, dict):
        for key, value in tags.items():
            if key == "Dimension" and isinstance(value, dict):
                found.append(value)
            elif isinstance(value, dict):
                found.extend(_find_dimension_tags(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        found.extend(_find_dimension_tags(item))
    return found

def try_read_emd(file_bytes: bytes) -> TEMImage:
    if not _HAS_H5PY:
        raise RuntimeError("h5py is not installed.  pip install h5py")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".emd")
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()

        with h5py.File(tmp.name, "r") as h5:
            datasets = []
            
            # Recursively collect all datasets
            def collect_datasets(name, node):
                if isinstance(node, h5py.Dataset):
                    datasets.append((name, node))
            
            h5.visititems(collect_datasets)
            
            # Filter for likely image data: 
            # 1. Must have at least 2 dimensions
            # 2. Both spatial dimensions must be > 1 to avoid picking up 1D spectra shaped as (N, 1)
            valid_ds = [x for x in datasets if x[1].ndim >= 2 and x[1].shape[0] > 1 and x[1].shape[1] > 1]
            
            if not valid_ds:
                valid_ds = [x for x in datasets if x[1].ndim >= 2]
                
            if not valid_ds:
                st.warning("No 2-D datasets found in EMD file.")
                return TEMImage(np.zeros((1, 1), np.float64), float("nan"))

            # Rank them: Prioritize Velox standard image structures, then by total pixel count
            def rank_ds(item):
                name, node = item
                is_velox_img = "Data/Image" in name and name.endswith("/Data")
                size = int(np.prod(node.shape[:2]))
                return (1 if is_velox_img else 0, size)

            best_name, best_ds = max(valid_ds, key=rank_ds)
            data = np.array(best_ds, dtype=np.float64)
            
            # Handle 3D+ arrays (Image Stacks, Data Cubes, or RGB)
            if data.ndim >= 3:
                if data.shape[-1] in (3, 4):  # RGB/RGBA image
                    data = np.mean(data[..., :3], axis=-1)
                else:
                    # Multi-frame stack: take the first frame
                    while data.ndim > 2:
                        data = data[0]

            # --- METADATA EXTRACTION ---
            nm_per_px = float("nan")
            
            # Strategy 1: Velox Metadata JSON sidecar
            meta_name = best_name.rsplit("/", 1)[0] + "/Metadata"
            if meta_name in h5:
                try:
                    meta_arr = np.array(h5[meta_name])
                    if meta_arr.dtype.kind in ('S', 'O'): # Array of strings
                        meta_str = "".join([x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else str(x) for x in meta_arr.flatten()])
                    else: # Array of bytes/ints
                        meta_str = meta_arr.tobytes().decode('utf-8', errors='ignore')
                    
                    # Regex search to safely grab the width from the embedded JSON text
                    match = re.search(r'"PixelSize"\s*:\s*\{\s*"width"\s*:\s*([0-9\.eE+-]+)', meta_str, re.IGNORECASE)
                    if match:
                        nm_per_px = float(match.group(1)) * 1e9  # Velox strictly stores pixel size in meters
                except Exception:
                    pass
                    

                
        return TEMImage(data=data, nm_per_px=nm_per_px)
        
    except Exception as exc:
        st.warning(f"EMD read error: {exc}")
        return TEMImage(np.zeros((1, 1), np.float64), float("nan"))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

# ═══════════════════════════════════════════════════════════════════════════
# File reading — EMD (Velox / HDF5)
# ═══════════════════════════════════════════════════════════════════════════
def extract_pixel_size_nm(meta_str: str, default: float = float("nan")) -> float:
    """Parse pixel size (nm/pixel) from EMD metadata JSON/string."""
    def _find_pixelsize_in_obj(obj):
        if isinstance(obj, dict):
            if "PixelSize" in obj:
                ps = obj["PixelSize"]
                if isinstance(ps, dict):
                    w = ps.get("width")
                    h = ps.get("height")
                    vals = []
                    for v in (w, h):
                        if v is None: continue
                        try:
                            vals.append(float(v))
                        except Exception:
                            pass
                    if vals:
                        v_m = sum(vals) / len(vals)
                        return v_m * 1e9 # meters to nm
            for v in obj.values():
                res = _find_pixelsize_in_obj(v)
                if res is not None:
                    return res
        elif isinstance(obj, list):
            for v in obj:
                res = _find_pixelsize_in_obj(v)
                if res is not None:
                    return res
        return None

    # 1. Try to parse as JSON
    try:
        meta = json.loads(meta_str)
        val = _find_pixelsize_in_obj(meta)
        if val is not None:
            return val
    except Exception:
        pass

    # 2. Regex fallback (for string-formatted scientific notation)
    ps_pattern = re.compile(
        r'"PixelSize"\s*:\s*\{[^}]*"width"\s*:\s*"([^"]+)"\s*,\s*"height"\s*:\s*"([^"]+)"[^}]*\}',
        re.IGNORECASE
    )
    m = ps_pattern.search(meta_str)
    if m:
        try:
            w, h = float(m.group(1)), float(m.group(2))
            return ((w + h) / 2.0) * 1e9
        except Exception:
            pass

    # 3. Regex fallback (for raw float scientific notation)
    ps_pattern2 = re.compile(
        r'"PixelSize"\s*:\s*\{[^}]*"width"\s*:\s*([0-9\.eE+-]+)\s*,\s*"height"\s*:\s*([0-9\.eE+-]+)[^}]*\}',
        re.IGNORECASE
    )
    m2 = ps_pattern2.search(meta_str)
    if m2:
        try:
            w, h = float(m2.group(1)), float(m2.group(2))
            return ((w + h) / 2.0) * 1e9
        except Exception:
            pass

    return default


def _search_hdf5_datasets(group, collected: list, depth: int = 0) -> None:
    """Fallback recursive search for non-Velox EMDs."""
    if depth > 20: return
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            if item.ndim >= 2:
                collected.append(item)
        elif isinstance(item, h5py.Group):
            _search_hdf5_datasets(item, collected, depth + 1)


def try_read_emd(file_bytes: bytes) -> TEMImage:
    if not _HAS_H5PY:
        raise RuntimeError("h5py is not installed.  pip install h5py")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".emd")
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()

        with h5py.File(tmp.name, "r") as h5:
            # --- Primary Strategy: Velox Format (From Labmate) ---
            if "Data" in h5 and "Image" in h5["Data"]:
                img_group = h5["Data"]["Image"]
                keys = list(img_group.keys())
                if keys:
                    first_key = keys[0]
                    data_group = img_group[first_key]
                    
                    # 1. Extract Data
                    img_stack = data_group["Data"][()]
                    if img_stack.ndim == 3:
                        data = img_stack[:, :, 0].astype(np.float64)
                    elif img_stack.ndim == 2:
                        data = img_stack.astype(np.float64)
                    else:
                        data = img_stack[0].astype(np.float64)
                        
                    # 2. Extract Metadata
                    nm_per_px = float("nan")
                    if "Metadata" in data_group:
                        meta_raw = data_group["Metadata"][()]
                        meta_str = meta_raw.tobytes().decode("utf-8", errors="ignore")
                        nm_per_px = extract_pixel_size_nm(meta_str, default=float("nan"))
                        
                    return TEMImage(data=data, nm_per_px=nm_per_px)

            # --- Secondary Strategy: Fallback generic search (for Berkeley EMDs) ---
            datasets: list = []
            _search_hdf5_datasets(h5, datasets)
            if not datasets:
                st.warning("No 2-D datasets found in EMD file.")
                return TEMImage(np.zeros((1, 1), np.float64), float("nan"))

            best = max(datasets, key=lambda d: int(np.prod(d.shape[:2])))
            data = np.array(best, dtype=np.float64)
            if data.ndim == 3:
                data = np.mean(data[..., :3], axis=2) if data.shape[2] in (3, 4) else data[0]
            if data.ndim > 2:
                data = data.reshape(data.shape[0], data.shape[1])

            # Try to find standard HDF5 attributes
            nm_per_px = float("nan")
            for attr_name in ("pixelSize", "PixelSize", "pixel_size"):
                if attr_name in best.attrs:
                    val = float(best.attrs[attr_name])
                    unit_str = str(best.attrs.get(f"{attr_name}Unit", "m"))
                    nm_per_px = _unit_to_nm(val, unit_str)
                    break

            return TEMImage(data=data, nm_per_px=nm_per_px)
            
    except Exception as exc:
        st.warning(f"EMD read error: {exc}")
        return TEMImage(np.zeros((1, 1), np.float64), float("nan"))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def read_tem_file(file_bytes: bytes, filename: str) -> TEMImage:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".dm3":
        return try_read_dm3(file_bytes)
    elif ext == ".emd":
        return try_read_emd(file_bytes)
    else:
        st.error(f"Unsupported file type: {ext}")
        return TEMImage(np.zeros((1, 1), np.float64), float("nan"))


# ═══════════════════════════════════════════════════════════════════════════
# Threshold helpers
# ═══════════════════════════════════════════════════════════════════════════
def robust_percentile_cut(data: np.ndarray, p: float = 99.5) -> np.ndarray:
    flat = data.ravel()
    return flat[flat <= np.percentile(flat, p)]


def histogram_for_intensity(
    data: np.ndarray, nbins: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    vals = robust_percentile_cut(data, 99.5)
    if nbins is None:
        nbins = max(10, int(round(math.sqrt(len(vals)) / 2)))
    counts, edges = np.histogram(vals, bins=nbins)
    centers = edges[:-1] + np.diff(edges) / 2
    return centers, counts


def kmeans_threshold(data: np.ndarray, sample: int = 50_000) -> float:
    flat = data.ravel().astype(np.float64)
    if flat.size > sample:
        flat = flat[np.random.choice(flat.size, sample, replace=False)]
    km = KMeans(n_clusters=2, n_init="auto", random_state=42)
    labs = km.fit_predict(flat.reshape(-1, 1))
    return float(min(flat[labs == 0].max(), flat[labs == 1].max()))


def gmm_threshold(data: np.ndarray, nbins: Optional[int] = None,
                   sample: int = 50_000) -> float:
    flat = data.ravel().astype(np.float64)
    if flat.size > sample:
        flat = flat[np.random.choice(flat.size, sample, replace=False)]
    if flat.size < 2:
        return float(np.median(data)) if data.size else 0.0
    gm = GaussianMixture(n_components=3, random_state=42)
    gm.fit(flat.reshape(-1, 1))
    mu = np.sort(gm.means_.ravel())
    centers, counts = histogram_for_intensity(flat, nbins)
    mask = (centers >= mu[0]) & (centers <= mu[1])
    if not np.any(mask):
        return float((mu[0] + mu[1]) / 2)
    return float(centers[mask][np.argmin(counts[mask])])


def auto_threshold(data: np.ndarray) -> float:
    candidates: List[float] = []
    try:
        candidates.append(gmm_threshold(data))
    except Exception:
        pass
    try:
        candidates.append(float(threshold_otsu(data)))
    except Exception:
        pass
    try:
        candidates.append(float(threshold_li(data)))
    except Exception:
        pass
    return candidates[0] if candidates else float(np.median(data))


# ═══════════════════════════════════════════════════════════════════════════
# Robust pre-processing
# ═══════════════════════════════════════════════════════════════════════════
def preprocess_image(data: np.ndarray) -> np.ndarray:
    """Adaptive pre-processing for varying noise / contrast levels."""
    img = data.astype(np.float64)

    lap = laplace(img)
    noise_est = np.median(np.abs(lap)) * 1.4826

    if noise_est > 0:
        snr = (img.max() - img.min()) / (noise_est + 1e-12)
        if snr < 5:
            img = median_filter(img, size=5)
        elif snr < 15:
            img = median_filter(img, size=3)

    img = gaussian_filter(img, sigma=1.0)

    lo, hi = np.percentile(img, [0.5, 99.5])
    if hi - lo > 0:
        img = np.clip((img - lo) / (hi - lo), 0, 1)
    else:
        img = np.zeros_like(img)
    img = exposure.equalize_adapthist(img, clip_limit=0.01)

    return img


# ═══════════════════════════════════════════════════════════════════════════
# Shape classification
# ═══════════════════════════════════════════════════════════════════════════
SHAPE_CHOICES = ("Sphere", "Hexagonal Prism", "Cube", "Octahedron")

_COLORS: Dict[str, str] = {
    "circle":    "lime",
    "hexagon":   "cyan",
    "rectangle": "yellow",
    "square":    "orange",
    "diamond":   "magenta",
    "unknown":   "gray",
}

def classify_projection(prop, target_shape: str) -> str:
    """Classify the 2-D projection of a *regionprops* object."""
    area = float(prop.area)
    perim = float(getattr(prop, "perimeter", 0.0)) or 1e-6
    circ = 4.0 * np.pi * area / perim ** 2
    solidity = float(getattr(prop, "solidity", 0.0))
    extent = float(getattr(prop, "extent", 0.0))
    maj = float(getattr(prop, "major_axis_length", 0.0)) or 1e-6
    minor = float(getattr(prop, "minor_axis_length", 0.0)) or 1e-6
    aspect = maj / minor

    if target_shape == "Sphere":
        if circ > 0.60 and aspect < 1.7:
            return "circle"

    elif target_shape == "Hexagonal Prism":
        # Hexagons: often rounded, so relax circularity and solidity slightly
        if circ > 0.65 and solidity > 0.85 and aspect < 1.45:
            return "hexagon"
        # Rectangles (side-on): elongated, boxy, but corners might be rounded (lower extent)
        if extent > 0.65 and solidity > 0.75 and aspect > 1.15:
            return "rectangle"
        # Relaxed fallbacks for slightly irregular projections
        if circ > 0.60 and solidity > 0.80 and aspect < 1.50:
            return "hexagon"

    elif target_shape == "Cube":
        if extent > 0.68 and solidity > 0.82 and aspect < 1.45:
            return "square"

    elif target_shape == "Octahedron":
        if solidity > 0.78 and aspect < 1.55 and extent < 0.78:
            return "diamond"
        if solidity > 0.72 and aspect < 1.65:
            return "diamond"

    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Geometry helpers for Plotting (NumPy coordinates)
# ═══════════════════════════════════════════════════════════════════════════
def _hexagon_vertices(cx: float, cy: float, radius: float) -> np.ndarray:
    angles = np.pi / 3.0 * np.arange(6)
    return np.column_stack((cx + radius * np.cos(angles), cy + radius * np.sin(angles)))

def _diamond_vertices(cx: float, cy: float, half_maj: float, half_min: float, angle: float) -> np.ndarray:
    ca, sa = np.cos(angle), np.sin(angle)
    return np.array([
        [cx + half_maj * ca, cy + half_maj * sa],
        [cx - half_min * sa, cy + half_min * ca],
        [cx - half_maj * ca, cy - half_maj * sa],
        [cx + half_min * sa, cy - half_min * ca],
    ])

def _rotated_rect_vertices(cx: float, cy: float, half_maj: float, half_min: float, angle: float) -> np.ndarray:
    ca, sa = np.cos(angle), np.sin(angle)
    corners = np.array([
        [half_maj, half_min],
        [-half_maj, half_min],
        [-half_maj, -half_min],
        [half_maj, -half_min]
    ])
    xs = cx + corners[:, 0] * ca - corners[:, 1] * sa
    ys = cy + corners[:, 0] * sa + corners[:, 1] * ca
    return np.column_stack((xs, ys))


# ═══════════════════════════════════════════════════════════════════════════
# Segmentation + measurement
# ═══════════════════════════════════════════════════════════════════════════
def segment_and_measure(
    data: np.ndarray,
    nm_per_px: float,
    shape_type: str,
    min_size_value: float,
    measurement_unit: str,
    min_area_px: int = 5,
    thresh_offset: float = 1.0, 
) -> Dict[str, Any]:
    """Segment particles with robust watershed and classify shapes.

    Returns a dict with measurement arrays, shape geometries, and metadata.
    """
    preprocessed = preprocess_image(data)

    try:
        adapted_thresh = gmm_threshold(preprocessed)
    except Exception:
        try:
            adapted_thresh = float(threshold_otsu(preprocessed))
        except Exception:
            adapted_thresh = float(np.median(preprocessed))

    adapted_thresh = adapted_thresh * 1.5 #boost adapt threshold by 50%
    im_bi = preprocessed < (adapted_thresh * thresh_offset)
    im_bi = remove_small_objects(im_bi, min_size=min_area_px)
    im_bi = remove_small_holes(im_bi, area_threshold=max(min_area_px*4, 64))

    # Calculate expected minimum radius in pixels
    if measurement_unit == "nm" and np.isfinite(nm_per_px) and nm_per_px > 0:
        min_diam_px = min_size_value / nm_per_px
    else:
        min_diam_px = min_size_value

    expected_r_px = max(2.0, min_diam_px / 2.0)

    # 1. Adaptive Closing (~15% of expected radius)
    close_rad = max(1, int(expected_r_px * 0.15))
    im_bi = binary_closing(im_bi, disk(close_rad))
    im_bi = binary_opening(im_bi, disk(max(1, close_rad - 1)))

    # 2. Adaptive Smoothing (~10% of expected radius)
    dist = distance_transform_edt(im_bi)
    sm_sigma = max(0.5, expected_r_px * 0.25)
    dist_sm = gaussian_filter(dist, sigma=sm_sigma)
    
    # 3. Adaptive Peak Distance
    min_dist = max(2, int(expected_r_px * 0.45))
    coords = peak_local_max(dist_sm, min_distance=min_dist, labels=im_bi)
    
    mask_peaks = np.zeros(dist.shape, dtype=bool)
    if coords.size > 0:
        mask_peaks[tuple(coords.T)] = True
        
    markers = label(mask_peaks)
    labels_ws = watershed(-dist_sm, markers=markers, mask=im_bi)
    im_bi[labels_ws == 0] = 0

    if measurement_unit == "nm" and np.isfinite(nm_per_px) and nm_per_px > 0:
        sf = nm_per_px
        excl_px = 2.0
    else:
        sf = 1.0
        excl_px = 2.0
        
    img_h, img_w = data.shape

    measurements: Dict[str, List[float]] = {
        "diameters":   [],  
        "hex_widths":  [], 
        "hex_heights": [],
        "side_lengths": [], 
        "oct_major":   [],
        "oct_minor":   [],
    }

    draw_shapes = []

    for prop in regionprops(labels_ws):
        minr, minc, maxr, maxc = prop.bbox

        if (minr <= excl_px or minc <= excl_px
                or maxr >= img_h - excl_px or maxc >= img_w - excl_px):
            continue

        cls = classify_projection(prop, shape_type)
        if cls == "unknown":
            continue

        cy, cx = prop.centroid
        
        # Map skimage orientation (0 = vertical) to standard Cartesian geometry (0 = horizontal)
        orient = (np.pi / 2.0) - float(getattr(prop, "orientation", 0.0))
        color = _COLORS.get(cls, "white")

        if cls == "circle":
            maj_px = float(getattr(prop, "major_axis_length", 0.0)) or 0.0
            min_px = float(getattr(prop, "minor_axis_length", 0.0)) or 0.0
            d_px = (maj_px + min_px) / 2.0
            d_val = d_px * sf
            if d_val < min_size_value or (measurement_unit == "nm" and d_val > 250.0):
                continue
            measurements["diameters"].append(d_val)
            r = d_px / 2.0
            draw_shapes.append({"type": "circle", "cx": cx, "cy": cy, "r": r, "color": color})

        elif cls == "hexagon":
            maj_px = float(getattr(prop, "major_axis_length", 0.0)) or 0.0
            min_px = float(getattr(prop, "minor_axis_length", 0.0)) or 0.0
            d_px = (maj_px + min_px) / 2.0
            d_val = d_px * sf
            if d_val < min_size_value or (measurement_unit == "nm" and d_val > 250.0):
                continue
            measurements["hex_widths"].append(d_val)
            draw_shapes.append({"type": "polygon", "vertices": _hexagon_vertices(cx, cy, d_px / 2.0), "color": color})

        elif cls == "rectangle":
            maj_px = float(getattr(prop, "major_axis_length", 0.0)) or 0.0
            min_px = float(getattr(prop, "minor_axis_length", 0.0)) or 0.0
            length_val = maj_px * sf
            width_val = min_px * sf
            if length_val < min_size_value or (measurement_unit == "nm" and length_val > 250.0):
                continue
            measurements["hex_heights"].append(length_val)
            draw_shapes.append({"type": "polygon", "vertices": _rotated_rect_vertices(cx, cy, maj_px / 2.0, min_px / 2.0, orient), "color": color})

        elif cls == "square":
            # Scale by sqrt(3)/2 approx 0.866 to map standard ellipse axes to true square side length
            corr_factor_sq = 0.8660
            maj_px = (float(getattr(prop, "major_axis_length", 0.0)) or 0.0) * corr_factor_sq
            min_px = (float(getattr(prop, "minor_axis_length", 0.0)) or 0.0) * corr_factor_sq
            side_px = (maj_px + min_px) / 2.0
            side_val = side_px * sf
            if side_val < min_size_value or (measurement_unit == "nm" and side_val > 250.0):
                continue
            measurements["side_lengths"].append(side_val)
            half = side_px / 2.0
            draw_shapes.append({"type": "polygon", "vertices": _rotated_rect_vertices(cx, cy, half, half, orient), "color": color})

        elif cls == "diamond":
            # Scale by sqrt(1.5) approx 1.2247 to map standard ellipse axes perfectly to rhombus diagonals (sharp tips)
            corr_factor = 1.2247
            maj_px = (float(getattr(prop, "major_axis_length", 0.0)) or 0.0) * corr_factor
            min_px = (float(getattr(prop, "minor_axis_length", 0.0)) or 0.0) * corr_factor
            maj_val = maj_px * sf
            min_val = min_px * sf
            if maj_val < min_size_value or (measurement_unit == "nm" and maj_val > 250.0):
                continue
            measurements["oct_major"].append(maj_val)
            measurements["oct_minor"].append(min_val)
            draw_shapes.append({"type": "polygon", "vertices": _diamond_vertices(cx, cy, maj_px / 2.0, min_px / 2.0, orient), "color": color})

    n_labels = int(labels_ws.max()) + 1
    rng = np.random.RandomState(42)
    rand_cmap = rng.rand(max(n_labels, 1), 3)
    rand_cmap[0] = [1.0, 1.0, 1.0]
    rgb_ws = (rand_cmap[labels_ws] * 255).astype(np.uint8)

    out: Dict[str, Any] = {
        "data":           data,
        "draw_shapes":    draw_shapes,
        "rgb_ws":         rgb_ws,
        "adapted_thresh": float(adapted_thresh),
        "unit":           measurement_unit,
        "nm_per_px":      float(nm_per_px) if measurement_unit == "nm" else float("nan"),
    }
    for key, vals in measurements.items():
        out[key] = np.array(vals, dtype=np.float64)

    return out
# ═══════════════════════════════════════════════════════════════════════════
# Histogram with Gaussian Mixture Model (GMM) fit
# ═══════════════════════════════════════════════════════════════════════════
def histogram_with_fit(
    values: np.ndarray,
    title: str,
    unit_label: str,
    r_eff_factor: Optional[float] = None,
    bimodal_axes: bool = False,
    user_xlim: Optional[Tuple[float, float]] = None,
    gmm_components: int = 2,  # <--- NEW ARGUMENT
) -> Tuple[plt.Figure, float, float, int]:
    
    n_total = len(values)

    if user_xlim is not None and user_xlim[0] < user_xlim[1]:
        vis_min, vis_max = user_xlim[0], user_xlim[1]
        fit_vals = values[(values >= vis_min) & (values <= vis_max)]
    else:
        vis_min, vis_max = np.percentile(values, [0.5, 99.5])
        if vis_min >= vis_max:
            vis_min, vis_max = values.min(), values.max()
        fit_vals = values
        
    n_fit = len(fit_vals)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    mu, std = float("nan"), float("nan")

    if n_total > 0:
        visible_span = vis_max - vis_min
        if visible_span > 0:
            bin_width = visible_span / 40.0
            total_span = values.max() - values.min()
            n_bins = max(10, int(total_span / bin_width))
            n_bins = min(500, n_bins)
        else:
            n_bins = 10
            
        counts, bins, _ = ax.hist(
            values, 
            bins=n_bins, 
            color='steelblue', 
            edgecolor='black', 
            linewidth=0.5, 
            alpha=0.6, 
            density=False
        )
        bw = float(bins[1] - bins[0]) if len(bins) > 1 else 1.0
        
        if n_fit >= max(3, gmm_components):
            X = fit_vals.reshape(-1, 1)
            # Use the user's selected component count (capped by available data points if n_fit is very small)
            n_comp = min(gmm_components, n_fit)
            gmm = GaussianMixture(n_components=n_comp, random_state=42)
            gmm.fit(X)
            
            x_fit = np.linspace(vis_min, vis_max, 300).reshape(-1, 1)
            pdf = np.exp(gmm.score_samples(x_fit))
            p = pdf * n_fit * bw
            ax.plot(x_fit, p, 'k', linewidth=1.5, label='Combined Fit')
            
            if n_comp > 1:
                for i in range(n_comp):
                    w = gmm.weights_[i]
                    m = gmm.means_[i][0]
                    s = np.sqrt(gmm.covariances_[i][0][0])
                    comp_pdf = norm.pdf(x_fit.flatten(), m, s)
                    comp_p = w * comp_pdf * n_fit * bw
                    ax.plot(x_fit, comp_p, '--', color='crimson', linewidth=1.2, alpha=0.8)
            
            if bimodal_axes and n_comp == 2:
                order = np.argsort(gmm.means_.flatten())
                mu_min = float(gmm.means_[order[0]][0])
                std_min = float(np.sqrt(gmm.covariances_[order[0]][0][0]))
                mu_maj = float(gmm.means_[order[1]][0])
                std_maj = float(np.sqrt(gmm.covariances_[order[1]][0][0]))
                
                title_str = f"{title} (n={n_fit})"
                if r_eff_factor is not None and np.isfinite(mu_maj) and mu_maj > 0:
                    r_eff = mu_maj * r_eff_factor
                    title_str += f" | r_eff = {r_eff:.1f} {unit_label}"
                    
                ax.set_title(f"{title_str}\nMinor μ={mu_min:.2f} ± {std_min:.2f} | Major μ={mu_maj:.2f} ± {std_maj:.2f}", fontsize=13, pad=2)
                mu, std = mu_maj, std_maj  
                
            else:
                dom_idx = np.argmax(gmm.weights_)
                mu = float(gmm.means_[dom_idx][0])
                std = float(np.sqrt(gmm.covariances_[dom_idx][0][0]))
                
                title_str = f"{title} (n={n_fit})"
                if r_eff_factor is not None and np.isfinite(mu) and mu > 0:
                    r_eff = mu * r_eff_factor
                    title_str += f" | r_eff = {r_eff:.1f} {unit_label}"
                    
                ax.set_title(f"{title_str}\nDominant μ={mu:.2f} ± {std:.2f} {unit_label}", fontsize=14, pad=2)
        else:
            title_str = f"{title} (n={n_fit})"
            ax.set_title(title_str, fontsize=14, pad=2)
            
        ax.set_xlim(vis_min, vis_max)
            
    else:
        ax.set_title(f"{title} (n=0)", fontsize=14, pad=2)

    ax.set_xlabel(f"Size ({unit_label})", color='black', weight='bold')
    ax.set_ylabel("Count", color='black', weight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig, float(mu), float(std), n_fit

# ═══════════════════════════════════════════════════════════════════════════
# Histogram section builders 
# ═══════════════════════════════════════════════════════════════════════════
def _show_sphere_histograms(results: List[Dict], unit: str, user_xlim: Optional[Tuple[float, float]] = None, gmm_components: int = 2) -> None:
    all_d = np.concatenate([r["diameters"] for r in results])
    if all_d.size == 0:
        st.info("No spherical particles detected.")
        return

    prefix = _common_prefix([r["name"] for r in results])
    fig, mu, std, n = histogram_with_fit(all_d, f"{prefix} — Diameter", unit, user_xlim=user_xlim, gmm_components=gmm_components)
    st.pyplot(fig, use_container_width=True)
    _download_row(fig, pd.DataFrame({"diameter": all_d}), f"{prefix}_diameter")


def _show_hex_histograms(results: List[Dict], unit: str, user_xlim: Optional[Tuple[float, float]] = None, gmm_components: int = 2) -> None:
    all_w = np.concatenate([r["hex_widths"] for r in results])
    all_h = np.concatenate([r["hex_heights"] for r in results])
    prefix = _common_prefix([r["name"] for r in results])

    if all_w.size == 0 and all_h.size == 0:
        st.info("No hexagonal-prism particles detected.")
        return

    if all_w.size > 0:
        r_eff_hex = (9.0 * np.sqrt(3) / (32.0 * np.pi))**(1/3)
        fig_w, mu_w, std_w, n_w = histogram_with_fit(all_w, f"{prefix} — Hex width (face-on)", unit, r_eff_factor=r_eff_hex, user_xlim=user_xlim, gmm_components=gmm_components)
        st.pyplot(fig_w, use_container_width=True)
        _download_row(fig_w, pd.DataFrame({"hex_width": all_w}), f"{prefix}_hex_width")
    else:
        st.info("No face-on hexagonal projections detected.")

    if all_h.size > 0:
        fig_h, mu_h, std_h, n_h = histogram_with_fit(all_h, f"{prefix} — Height (side-on)", unit, user_xlim=user_xlim, gmm_components=gmm_components)
        st.pyplot(fig_h, use_container_width=True)
        _download_row(fig_h, pd.DataFrame({"hex_height": all_h}), f"{prefix}_hex_height")
    else:
        st.info("No side-on rectangular projections detected.")


def _show_cube_histograms(results: List[Dict], unit: str, user_xlim: Optional[Tuple[float, float]] = None, gmm_components: int = 2) -> None:
    all_s = np.concatenate([r["side_lengths"] for r in results])
    if all_s.size == 0:
        st.info("No cubic particles detected.")
        return

    prefix = _common_prefix([r["name"] for r in results])
    r_eff_cube = (0.75 / np.pi)**(1/3) 
    fig, mu, std, n = histogram_with_fit(all_s, f"{prefix} — Cube side length", unit, r_eff_factor=r_eff_cube, user_xlim=user_xlim, gmm_components=gmm_components)
    st.pyplot(fig, use_container_width=True)
    _download_row(fig, pd.DataFrame({"side_length": all_s}), f"{prefix}_side_length")


def _show_octahedron_histograms(results: List[Dict], unit: str, user_xlim: Optional[Tuple[float, float]] = None, gmm_components: int = 2) -> None:
    all_maj = np.concatenate([r["oct_major"] for r in results])
    all_min = np.concatenate([r["oct_minor"] for r in results])
    prefix = _common_prefix([r["name"] for r in results])

    if all_maj.size == 0 and all_min.size == 0:
        st.info("No octahedral particles detected.")
        return

    all_axes = np.concatenate([all_maj, all_min])
    r_eff_oct = (1.0 / (8.0 * np.pi))**(1/3)
    
    fig_axes, mu_maj, std_maj, n_axes = histogram_with_fit(
        all_axes, f"{prefix} — Octahedron Combined Axes", unit, 
        r_eff_factor=r_eff_oct, bimodal_axes=True, user_xlim=user_xlim, gmm_components=gmm_components
    )
    st.pyplot(fig_axes, use_container_width=True)

    df = pd.DataFrame({"combined_axes": all_axes})
    _download_row(fig_axes, df, f"{prefix}_octahedron_combined")

def _download_row(fig: plt.Figure, df: pd.DataFrame, stem: str) -> None:
    c1, c2 = st.columns(2)
    with c1:
        hist_buffer = io.BytesIO()
        fig.savefig(hist_buffer, format="png", bbox_inches="tight", dpi=300)
        st.download_button("⬇ PNG", hist_buffer.getvalue(), f"{stem}.png", "image/png")
    with c2:
        st.download_button("⬇ CSV", df.to_csv(index=False), f"{stem}.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# Streamlit helpers
# ═══════════════════════════════════════════════════════════════════════════
def _common_prefix(names: List[str]) -> str:
    if not names:
        return "analysis"
    shortest = min(names, key=len)
    for i, ch in enumerate(shortest):
        if any(name[i] != ch for name in names):
            return shortest[:i].rstrip(" _-.") or "analysis"
    return shortest.rstrip(" _-.") or "analysis"



# ═══════════════════════════════════════════════════════════════════════════
# Cached analysis wrapper
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _analyze_one(file_bytes: bytes, filename: str, shape_type: str,
                 min_feature_size: float, thresh_offset: float) -> Dict[str, Any]:
    tem = read_tem_file(file_bytes, filename)
    data = tem.data
    nm_per_px = tem.nm_per_px

    if data.size <= 1:
        return {"error": f"Image data is empty for {filename}."}

    if np.isfinite(nm_per_px) and nm_per_px > 0:
        measurement_unit = "nm"
        min_diam_px = min_feature_size / nm_per_px
    else:
        measurement_unit = "px"
        nm_per_px = float("nan")
        min_diam_px = min_feature_size

    min_area_px = max(1, int(np.pi * (min_diam_px / 2.0) ** 2))

    seg = segment_and_measure(
        data=data,
        nm_per_px=nm_per_px,
        shape_type=shape_type,
        min_size_value=float(min_feature_size),
        measurement_unit=measurement_unit,
        min_area_px=min_area_px,
        thresh_offset=thresh_offset,
    )
    seg["name"] = filename
    return seg





# ═══════════════════════════════════════════════════════════════════════════
# Main Streamlit entry point
# ═══════════════════════════════════════════════════════════════════════════
def run() -> None:
    st.set_page_config(page_title="TEM Particle Characterization",
                       layout="wide")
    st.title("TEM Particle Characterization")

    missing: List[str] = []
    if not _HAS_DM3:
        missing.append("`dm3_lib`  →  `pip install dm3_lib`")
    if not _HAS_H5PY:
        missing.append("`h5py`     →  `pip install h5py`")
    if missing:
        st.warning("Optional readers not installed:\n\n" +
                   "\n".join(f"- {m}" for m in missing))

    st.caption(
        "Upload one or more `.dm3` or `.emd` images, choose the particle "
        "shape, then view size distributions with Gaussian fits."
    )

    col_left, col_right = st.columns([1, 1])

    accepted_types = []
    if _HAS_DM3:
        accepted_types.append("dm3")
    if _HAS_H5PY:
        accepted_types.append("emd")
    if not accepted_types:
        st.error("No file readers available. Install dm3_lib and/or h5py.")
        return

    with col_left:
        files = st.file_uploader(
            "Upload TEM image(s)",
            accept_multiple_files=True,
            type=accepted_types,
            on_change=_clear_cache 
        )

    with col_right:
        shape_type = st.selectbox("Particle shape", list(SHAPE_CHOICES),
                                  index=0)
        min_feature = st.number_input(
            "Minimum feature size (nm)",
            min_value=0.0, max_value=250.0, value=5.0, step=0.5,
        )

        thresh_offset = st.slider("Threshold (Lower = Stricter Watershedding)", 0.50, 1.50, 1.00, 0.05)
        gmm_components = st.number_input("GMM Components (Histogram Fit)", min_value=1, max_value=5, value=2, step=1)
        use_custom_xlim = st.checkbox("Lock Histogram X-axis limits")
        if use_custom_xlim:
            c_min, c_max = st.columns(2)
            with c_min:
                xlim_min = st.number_input("X Min", value=0.0, step=1.0)
            with c_max:
                xlim_max = st.number_input("X Max", value=50.0, step=1.0)
            user_xlim = (xlim_min, xlim_max)
        else:
            user_xlim = None

    if not files:
        st.info(f"Upload one or more `{'` / `'.join(accepted_types)}` "
                "files to begin.")
        return

    results: List[Dict[str, Any]] = []
    missing_scale: List[str] = []

    with st.status("Processing images…", expanded=True) as status:
        for i, f in enumerate(files, 1):
            status.update(label=f"Processing {i}/{len(files)}: {f.name}")
            seg = _analyze_one(f.read(), f.name, shape_type, min_feature, thresh_offset)
            if "error" in seg:
                st.warning(seg["error"])
                continue
            if seg.get("unit", "px") == "px":
                missing_scale.append(f.name)
            results.append(seg)
        status.update(label="Processing complete!", state="complete",
                      expanded=False)

    if not results:
        st.warning("No images could be processed.")
        return

    if missing_scale:
        st.warning(
            "Pixel-size metadata was **not found** for the following files.  "
            "All measurements are reported in **pixels**:\n\n" +
            "\n".join(f"- {n}" for n in sorted(set(missing_scale)))
        )

    names = [r["name"] for r in results]
    sel_name = st.selectbox("Select image to display", names)
    sel = next(r for r in results if r["name"] == sel_name)

    c1, c2 = st.columns(2)
    with c1:
        thresh_str = f"{sel.get('adapted_thresh', 0):.4f}"
        #st.info(f"Adaptive threshold applied = **{thresh_str}**")
    with c2:
        nm_per_px_val = sel.get("nm_per_px", float("nan"))
        if np.isfinite(nm_per_px_val):
            st.success(f"Extracted metadata calibration = **{nm_per_px_val:.4f} nm/px**")
        else:
            st.warning("Extracted metadata calibration = **Uncalibrated (px)**")

    tab_ann, tab_ws = st.tabs(["🔬 Annotated", "🎨 Watershed"])
    with tab_ann:
        st.caption(f"Detected shapes overlaid ({sel['unit']})")
        
        # Fast Matplotlib rasterization avoiding Streamlit DOM limits
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
        zmin, zmax = np.percentile(sel["data"], [0.1, 99.9])
        ax.imshow(sel["data"], cmap="gray", vmin=zmin, vmax=zmax)
        
        for s in sel["draw_shapes"]:
            if s["type"] == "circle":
                ax.add_patch(patches.Circle((s["cx"], s["cy"]), s["r"], edgecolor=s["color"], facecolor='none', lw=1.5, alpha=0.5))
            elif s["type"] == "polygon":
                ax.add_patch(patches.Polygon(s["vertices"], closed=True, edgecolor=s["color"], facecolor='none', lw=1.5, alpha=0.5))
                
        ax.axis("off")
        fig.tight_layout(pad=0)
        st.pyplot(fig)
        plt.close(fig) # Prevent memory leak
        
    with tab_ws:
        st.caption("Watershed labels (randomly coloured)")
        st.image(sel["rgb_ws"], use_container_width=True)

    st.markdown("---")
    st.subheader("Size Distributions")

    units_present = {r.get("unit", "nm") for r in results}
    if len(units_present) > 1:
        st.error("Cannot combine histograms — uploaded images use mixed "
                 "measurement units (nm vs px).")
        return

    unit = units_present.pop()

    if shape_type == "Sphere":
        _show_sphere_histograms(results, unit, user_xlim, gmm_components)
    elif shape_type == "Hexagonal Prism":
        _show_hex_histograms(results, unit, user_xlim, gmm_components)
    elif shape_type == "Cube":
        _show_cube_histograms(results, unit, user_xlim, gmm_components)
    elif shape_type == "Octahedron":
        _show_octahedron_histograms(results, unit, user_xlim, gmm_components)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run()