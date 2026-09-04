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
import os
import textwrap
import re
import json
import tempfile
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import TextArea
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import gaussian_filter, laplace, median_filter, uniform_filter, distance_transform_edt, label
from sklearn.mixture import GaussianMixture
from skimage import exposure
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import disk, opening
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from utils import file_uploader_with_clear, install_canvas_image_compat

# Optional: emd support requires h5py
try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    h5py = None
    _HAS_H5PY = False

# Optional: dm3 support
try:
    import dm3_lib as pyDM3reader
    _HAS_DM3 = True
except ImportError:
    pyDM3reader = None
    _HAS_DM3 = False

# Optional: annotation-driven tuning needs a drawing canvas
try:
    from PIL import Image, ImageDraw
    from streamlit_drawable_canvas import st_canvas
    # Must run before the first st_canvas() call; see install_canvas_image_compat.
    install_canvas_image_compat()
    _HAS_CANVAS = True
except ImportError:
    Image = ImageDraw = st_canvas = None
    _HAS_CANVAS = False


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


def _is_streamlit_cloud() -> bool:
    """Best-effort detection of Streamlit Community Cloud.

    Community Cloud mounts the repo under /mount/src and runs on Linux; that
    path does not exist in local runs, so it is a reliable signal. The
    PSYFIT_FORCE_CLOUD env var overrides the heuristic ("1"/"true" forces the
    cloud banner on, "0"/"false" forces it off) for testing or if the mount
    path ever changes.
    """
    override = os.environ.get("PSYFIT_FORCE_CLOUD")
    if override is not None:
        return override.strip().lower() in ("1", "true", "yes")
    return os.path.isdir("/mount/src")


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

@st.cache_data(show_spinner=False)
def extract_representative_roi(data: np.ndarray, nm_per_px: float, roi_size_nm: float) -> np.ndarray:
    """Pick an ROI rich in *well-separated, representative* particles rather than the highest-
    contrast (usually clumpiest) region.

    Touching particles merge into one oversized connected component, so counting *normal-sized*
    components inside a window is a direct proxy for "how many cleanly separated particles" it
    holds. We slide the window (via a box filter over component centroids), reward normal-sized
    components and penalise oversized clumps, and centre the ROI on the best-scoring spot.
    Falls back to the high-contrast ROI if segmentation finds too little.
    """
    H, W = data.shape[0], data.shape[1]
    roi_px = int(roi_size_nm) if (np.isnan(nm_per_px) or nm_per_px <= 0) else int(roi_size_nm / nm_per_px)
    if roi_px >= min(H, W) or roi_px < 3:
        return data
    try:
        sm = gaussian_filter(data.astype(np.float64), 2.0)
        inv = sm.max() - sm                                  # particles (dark) → bright
        mask = opening(inv > threshold_otsu(inv), disk(2))
        lbl, _ = label(mask)
        props = regionprops(lbl)
        areas = np.array([p.area for p in props], dtype=float)
        if areas.size < 3:
            raise ValueError("too few particles to locate a representative ROI")
        med = float(np.median(areas))
        cents = np.array([p.centroid for p in props])        # (row, col)
        good = cents[(areas >= 0.3 * med) & (areas <= 2.5 * med)]   # isolated, typical particles
        clump = cents[areas > 2.5 * med]                            # merged/touching blobs
        good_map, clump_map = np.zeros((H, W)), np.zeros((H, W))
        if len(good):
            good_map[good[:, 0].astype(int), good[:, 1].astype(int)] = 1.0
        if len(clump):
            clump_map[clump[:, 0].astype(int), clump[:, 1].astype(int)] = 1.0
        # Local counts within the ROI window (mode='constant' zero-pads, so edge windows score low).
        score = (uniform_filter(good_map, size=roi_px, mode="constant")
                 - 2.0 * uniform_filter(clump_map, size=roi_px, mode="constant"))
        if not np.any(score > 0):
            raise ValueError("no window with well-separated particles")
        y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    except Exception:
        return extract_high_contrast_roi(data, nm_per_px, roi_size_nm)
    half = roi_px // 2
    r0 = max(0, min(y - half, H - roi_px))
    c0 = max(0, min(x - half, W - roi_px))
    return data[r0:r0 + roi_px, c0:c0 + roi_px]

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
# When face-on hexagons number fewer than this fraction of side-on rectangles,
# the face-on width sample is too small to trust, so the rectangle-GMM width
# estimate is auto-enabled (the user can still override it).
HEX_GMM_AUTO_RATIO = 0.2
# A tic-tac width-profile bin counts as "body" (rather than an end cap) while its width stays
# at least this fraction of the full body width; the body length is the span of such bins.
# Lower = more of the tapering cap gets absorbed into the body; higher = noisier junction.
BODY_WIDTH_FRAC = 0.9
_COLORS = {
    "circle": "lime", "hexagon": "cyan", "rectangle": "yellow",
    "square": "orange", "diamond": "magenta", "ellipse": "violet", "stadium": "violet", "unknown": "gray"
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
    elif target_shape == "Tic Tac" and solidity > 0.85 and 1.1 <= aspect <= 4.0: return "stadium"
    
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

def _stadium_vertices(cx: float, cy: float, L_body: float, W: float, d: float, angle: float, n_arc: int = 16) -> np.ndarray:
    """Outline of a stadium: a rectangle body (length ``L_body``, width ``W``) with two
    circular-arc end caps of sagitta ``d``, centred at (cx, cy) and rotated by ``angle``
    (radians) in pixel space. All lengths are in pixels."""
    half_L, half_W = L_body / 2.0, W / 2.0
    local: List[Tuple[float, float]] = [(-half_L, half_W), (half_L, half_W)]
    if d > 1e-6 and half_W > 1e-6:
        R = (d ** 2 + half_W ** 2) / (2.0 * d)               # arc radius from chord + sagitta
        phi = float(np.arcsin(min(half_W / R, 1.0)))         # half-angle subtended by the chord
        xc = half_L + d - R                                  # right-cap circle centre (on long axis)
        local += [(xc + R * np.cos(t), R * np.sin(t)) for t in np.linspace(phi, -phi, n_arc)]
        local.append((-half_L, -half_W))
        local += [(-(xc + R * np.cos(t)), R * np.sin(t)) for t in np.linspace(-phi, phi, n_arc)]
    else:
        local += [(half_L, -half_W), (-half_L, -half_W)]     # square ends (degenerate cap)
    ca, sa = np.cos(angle), np.sin(angle)
    verts = [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in local]
    return np.array(verts)

def _fit_stadium(prop) -> Optional[Dict[str, float]]:
    """Fit a stadium (rectangle body + two circular-arc end caps) to a labelled region via a
    PCA width-profile in the particle's principal frame (no image interpolation).

    ``L_body`` is measured *directly* as the span of the central plateau where the width sits
    at (near) its full value; the caps are the tapering ends, so ``d = (L_total − L_body)/2``.
    This deliberately replaces the older approach of back-solving the cap depth from the
    silhouette area: that inversion had two clamped regimes (``d→0`` when the outline fills its
    bounding box, ``d→W/2`` when the ends read fully round), and orientation/pixelation noise
    flipped otherwise-identical particles between them, splitting the body-length distribution
    into a spurious two peaks. A direct plateau measurement is monotonic in the true geometry,
    so uniform particles now land on one peak.

    Returns pixel-unit geometry ``{L_total, W, L_body, d, R, angle}`` or ``None`` when the
    region is too small/degenerate. ``angle`` is the long-axis orientation (radians) in
    (col=x, row=y) pixel space, used for drawing.
    """
    coords = prop.coords.astype(np.float64)                  # (N, 2): (row=y, col=x)
    if coords.shape[0] < 10:
        return None
    cy, cx = coords.mean(axis=0)
    xs, ys = coords[:, 1] - cx, coords[:, 0] - cy
    cov = np.cov(np.vstack((xs, ys)))
    if not np.all(np.isfinite(cov)):
        return None
    _, evecs = np.linalg.eigh(cov)                           # columns ascending by eigenvalue
    major, minor = evecs[:, -1], evecs[:, 0]                 # long / short principal axes (x, y)
    u = xs * major[0] + ys * major[1]                        # coordinate along the long axis
    v = xs * minor[0] + ys * minor[1]                        # coordinate across the particle
    L_total = float(u.max() - u.min())
    if L_total <= 1.0:
        return None

    # Width profile: bin the long axis (~1 px bins), width(bin) = span of v inside it.
    # The +eps in the denominator keeps the largest u from rounding into its own bin,
    # which would otherwise leave sporadic empty bins.
    n_bins = max(int(np.ceil(L_total)), 5)
    bin_w = L_total / n_bins
    bin_idx = np.clip(((u - u.min()) / (L_total + 1e-9) * n_bins).astype(int), 0, n_bins - 1)
    widths = np.zeros(n_bins)
    for b in range(n_bins):
        vb = v[bin_idx == b]
        if vb.size:
            widths[b] = vb.max() - vb.min()

    occupied = widths[widths > 0]
    if occupied.size == 0:
        return None
    # Full (body) width = the plateau level: the median of the near-widest bins, robust to a
    # single spuriously wide bin (which the old 95th-percentile would latch onto).
    plateau = widths[widths >= 0.9 * float(widths.max())]
    W = float(np.median(plateau)) if plateau.size else float(np.max(widths))
    if W <= 0:
        return None

    # Body length = span of the contiguous central bins that stay near the full width; the
    # tapering ends are the caps. First↔last body bin (inclusive) so the partly-filled junction
    # bins count with the body. BODY_WIDTH_FRAC sets how far below W still counts as body.
    body_bins = np.where(widths >= BODY_WIDTH_FRAC * W)[0]
    L_body = float((body_bins[-1] - body_bins[0] + 1) * bin_w) if body_bins.size else L_total
    L_body = min(max(L_body, 0.0), L_total)

    d = min(max((L_total - L_body) / 2.0, 0.0), W / 2.0)     # a cap can't exceed a semicircle
    L_body = L_total - 2.0 * d                               # keep body/caps consistent post-clamp
    R = (d ** 2 + (W / 2.0) ** 2) / (2.0 * d) if d > 1e-6 else float("inf")
    angle = float(np.arctan2(major[1], major[0]))
    return {"L_total": L_total, "W": W, "L_body": L_body, "d": d, "R": R, "angle": angle}

# ═══════════════════════════════════════════════════════════════════════════
# Fast Computer Vision Segmentation
# ═══════════════════════════════════════════════════════════════════════════
# The segmentation pipeline is split into four stages (preprocess → mask → watershed → measure)
# because each stage depends on only some of the tunable parameters. The annotation-driven
# auto-tuner (below) walks the parameter grid in that nesting order and reuses each stage's
# output across every setting that cannot change it, which is where most of its speed comes
# from. Keep the stages side-effect free so that caching stays valid.
def _min_peak_px(min_peak_distance: float, measurement_unit: str, nm_per_px: float) -> int:
    """``min_peak_distance`` arrives in measurement units (nm when calibrated); convert it to the
    pixel spacing ``peak_local_max`` expects."""
    if measurement_unit == "nm" and not np.isnan(nm_per_px) and nm_per_px > 0:
        return max(int(round(min_peak_distance / nm_per_px)), 1)
    return max(int(round(min_peak_distance)), 1)

def _binary_mask_from(preprocessed: np.ndarray, thresh_offset: float) -> np.ndarray:
    """Otsu-threshold the preprocessed image, then open it to drop tiny salt-noise specks.

    TEM particles are usually dark on a lighter background, so the image is inverted first and
    particles become bright. ``thresh_offset`` scales the Otsu level so the user can make the
    mask stricter (higher) or looser (lower).
    """
    inverted_img = preprocessed.max() - preprocessed
    binary_mask = inverted_img > threshold_otsu(inverted_img) * thresh_offset
    return opening(binary_mask, disk(2))

def _watershed_labels(binary_mask: np.ndarray, min_peak_px: int) -> np.ndarray:
    """Separate clumped particles with an EDT watershed seeded on distance-map peaks.

    Rectangles produce flat distance-map "ridges" that yield several peaks per particle and so
    fragment the fit, so the distance map is blurred before peak-finding to force a single peak
    at each particle's centre.
    """
    distance = distance_transform_edt(binary_mask)
    distance_smoothed = gaussian_filter(distance, sigma=2.0)
    coords = peak_local_max(distance_smoothed, min_distance=min_peak_px, labels=binary_mask)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = label(mask)
    return watershed(-distance, markers, mask=binary_mask)

def _measure_labels(
    labels_ws: np.ndarray, shape_type: str, min_size_value: float, sf: float,
    img_h: int, img_w: int,
) -> Tuple[Dict[str, list], list, List[Dict[str, Any]]]:
    """Classify and measure every watershed region, returning
    ``(measurements, draw_shapes, objects)``. ``sf`` scales pixels into the reported unit."""
    measurements = {
        "diameters": [], "hex_widths": [], "hex_heights": [], "hex_rect_widths": [],
        "side_lengths": [], "oct_major": [], "oct_minor": [],
        "tic_tac_major": [], "tic_tac_width": [], "tic_tac_body_length": [], "tic_tac_cap_depth": []
    }
    draw_shapes = []
    objects: List[Dict[str, Any]] = []   # per-shape centroid + measured dims (for FOV-crop scoring)

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
        n_before = len(objects)   # anything appended below belongs to this region (see loop end)

        if cls == "circle":
            d_px = (maj_px + min_px) / 2.0
            d_val = d_px * sf
            if d_val >= min_size_value:
                measurements["diameters"].append(d_val)
                draw_shapes.append({"type": "circle", "cx": cx, "cy": cy, "r": d_px/2.0, "color": color})
                objects.append({"cx": cx, "cy": cy, "dims": {"diameters": d_val}})
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
                objects.append({"cx": cx, "cy": cy, "dims": {"hex_widths": d_val}})
        elif cls == "rectangle":
            if maj_px * sf >= min_size_value:
                measurements["hex_heights"].append(maj_px * sf)
                # Minor axis of a side-on rectangle ≈ the hexagonal width; kept so
                # the width can be recovered from rectangles when face-on hexagons
                # are scarce (see the rectangle-GMM path in run()).
                measurements["hex_rect_widths"].append(min_px * sf)
                draw_shapes.append({"type": "polygon", "vertices": _rotated_rect_vertices(cx, cy, maj_px/2.0, min_px/2.0, orient), "color": color})
                objects.append({"cx": cx, "cy": cy,
                                "dims": {"hex_heights": maj_px * sf, "hex_rect_widths": min_px * sf}})
        elif cls == "square":
            side_px = ((maj_px + min_px) / 2.0) * 0.8660
            if side_px * sf >= min_size_value:
                measurements["side_lengths"].append(side_px * sf)
                draw_shapes.append({"type": "polygon", "vertices": _rotated_rect_vertices(cx, cy, side_px/2.0, side_px/2.0, orient), "color": color})
                objects.append({"cx": cx, "cy": cy, "dims": {"side_lengths": side_px * sf}})
        elif cls == "diamond":
            maj_val, min_val = maj_px * 1.2247 * sf, min_px * 1.2247 * sf
            if maj_val >= min_size_value:
                measurements["oct_major"].append(maj_val)
                measurements["oct_minor"].append(min_val)
                draw_shapes.append({"type": "polygon", "vertices": _diamond_vertices(cx, cy, (maj_px*1.2247)/2.0, (min_px*1.2247)/2.0, orient), "color": color})
                objects.append({"cx": cx, "cy": cy, "dims": {"oct_major": maj_val, "oct_minor": min_val}})
        elif cls == "stadium":
            fit = _fit_stadium(prop)
            if fit is None:
                continue
            L_total = fit["L_total"] * sf
            if L_total >= min_size_value:
                measurements["tic_tac_major"].append(L_total)
                measurements["tic_tac_width"].append(fit["W"] * sf)
                measurements["tic_tac_body_length"].append(fit["L_body"] * sf)
                measurements["tic_tac_cap_depth"].append(fit["d"] * sf)
                draw_shapes.append({
                    "type": "polygon",
                    "vertices": _stadium_vertices(cx, cy, fit["L_body"], fit["W"], fit["d"], fit["angle"]),
                    "color": color,
                })
                objects.append({"cx": cx, "cy": cy, "dims": {
                    "tic_tac_major": L_total, "tic_tac_width": fit["W"] * sf,
                    "tic_tac_body_length": fit["L_body"] * sf, "tic_tac_cap_depth": fit["d"] * sf}})

        # Tag whatever this region produced with its watershed label, so a hand-drawn
        # annotation can be matched to an object by looking its midpoint up in labels_ws.
        for o in objects[n_before:]:
            o["label"] = int(prop.label)

    return measurements, draw_shapes, objects

def segment_and_measure(
    data: np.ndarray, nm_per_px: float, shape_type: str, min_size_value: float,
    measurement_unit: str, smoothing_sigma: float, contrast_clip: float,
    thresh_offset: float, min_peak_distance: float
) -> Dict[str, Any]:
    preprocessed = preprocess_image(data, smoothing_sigma, contrast_clip)
    binary_mask = _binary_mask_from(preprocessed, thresh_offset)
    labels_ws = _watershed_labels(
        binary_mask, _min_peak_px(min_peak_distance, measurement_unit, nm_per_px))

    sf = nm_per_px if measurement_unit == "nm" else 1.0
    img_h, img_w = data.shape
    measurements, draw_shapes, objects = _measure_labels(
        labels_ws, shape_type, min_size_value, sf, img_h, img_w)

    # Generate random colored mask for visualization
    rand_cmap = np.random.RandomState(42).rand(max(int(labels_ws.max()) + 1, 1), 3)
    rand_cmap[0] = [1.0, 1.0, 1.0] # Background is white

    out = {
        "data": data, "draw_shapes": draw_shapes, "objects": objects,
        "labels_ws": labels_ws,
        "rgb_ws": (rand_cmap[labels_ws] * 255).astype(np.uint8),
        "unit": measurement_unit, "nm_per_px": float(nm_per_px) if measurement_unit == "nm" else float("nan")
    }
    for k, v in measurements.items(): out[k] = np.array(v, dtype=np.float64)
    return out

# ═══════════════════════════════════════════════════════════════════════════
# Drawing Helper
# ═══════════════════════════════════════════════════════════════════════════
def plot_annotated_image(ax, image_data, draw_shapes, nm_per_px: float = float("nan"),
                         unit: str = "px", show_mag: bool = False, mask_alpha: float = 0.4):
    """Grayscale image with the fitted-shape overlays. When the image is calibrated a scale
    bar (sized 'nicely' relative to the field of view) is drawn lower-left; ``show_mag`` adds
    the estimated magnification below it, which is only valid for a full image, not a cropped ROI.
    ``mask_alpha`` controls the fill opacity of the fitted-shape overlays."""
    zmin, zmax = np.percentile(image_data, [0.1, 99.9])
    ax.imshow(image_data, cmap="gray", vmin=zmin, vmax=zmax)
    for s in draw_shapes:
        if s["type"] == "circle":
            ax.add_patch(patches.Circle((s["cx"], s["cy"]), s["r"], edgecolor=s["color"], facecolor=s["color"], alpha=mask_alpha, lw=2))
        elif s["type"] == "polygon":
            ax.add_patch(patches.Polygon(s["vertices"], closed=True, edgecolor=s["color"], facecolor=s["color"], alpha=mask_alpha, lw=2))
        elif s["type"] == "ellipse":
            ax.add_patch(patches.Ellipse((s["cx"], s["cy"]), s["w"], s["h"], angle=s["angle"], edgecolor=s["color"], facecolor=s["color"], alpha=mask_alpha, lw=2))
    if np.isfinite(nm_per_px) and nm_per_px > 0:
        h_px, w_px = image_data.shape[:2]
        mag_kx = _estimate_mag_kx(w_px * nm_per_px) if show_mag else float("nan")
        _add_scale_bar(ax, w_px, h_px, nm_per_px, unit, mag_kx=mag_kx, font_size=11.0)
    ax.axis("off")

# ═══════════════════════════════════════════════════════════════════════════
# Caching Full Batch Run
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_batch_analysis(file_data: List[Tuple[bytes, str]], shape_type: str, params: dict) -> List[Dict[str, Any]]:
    # Measurement schema v3: Tic Tac results now carry stadium geometry:
    # "tic_tac_width" (body width W), "tic_tac_body_length" (straight plateau L_body)
    # and "tic_tac_cap_depth" (arc sagitta d), replacing the old "tic_tac_minor".
    # "hex_rect_widths" (rectangle minor axis) remains for the hexagonal-prism
    # rectangle-GMM width estimate. Schema v4 adds "objects": a per-shape list of
    # {cx, cy, dims{key: value}} used to crop the summary FOV to the densest region of
    # in-range particles. v5: tic-tac "tic_tac_body_length" is now measured directly from the
    # width-profile plateau (not back-solved from area), fixing spurious bimodality. This
    # comment is part of the cached function's source, so editing it busts any stale cache.
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
def _fit_gaussians(
    values: np.ndarray, n_components: int, mu_ranges: Optional[List[Tuple[float, float]]],
) -> Tuple[list, list, list, str]:
    """Return ``(mus, stds, weights, method)`` sorted by mean.

    When ``mu_ranges`` (one (lo, hi) window per component) is supplied for a multi-component
    fit, each component's mean is *constrained* to its window by fitting an independent
    Gaussian to the data inside it, guaranteeing each μ lands in its range. Otherwise a
    standard ``n_components`` GaussianMixture is fit.
    """
    n_fit = len(values)
    if (n_components >= 2 and mu_ranges and len(mu_ranges) == n_components):
        comps = []
        for lo, hi in mu_ranges:
            sub = values[(values >= min(lo, hi)) & (values <= max(lo, hi))]
            if sub.size >= 2:
                comps.append((float(np.mean(sub)), float(np.std(sub)) or 1e-6, sub.size / n_fit))
        if len(comps) == n_components:
            comps.sort(key=lambda c: c[0])
            return ([c[0] for c in comps], [c[1] for c in comps], [c[2] for c in comps],
                    f"{n_components}-component GMM (range-constrained)")
    if n_fit >= 3 * n_components:
        gmm = GaussianMixture(n_components=n_components, random_state=42).fit(values.reshape(-1, 1))
        order = np.argsort(gmm.means_.flatten())
        mus = gmm.means_.flatten()[order].tolist()
        stds = np.sqrt(gmm.covariances_.flatten())[order].tolist()
        weights = gmm.weights_.flatten()[order].tolist()
        method = "Gaussian fit" if n_components == 1 else f"{n_components}-component GMM"
        return mus, stds, weights, method
    return [], [], [], ""

# When a fit min/max crops the data, the visible x-axis is padded by this fraction of the
# (max − min) fit span on each side, so a narrow distribution isn't awkwardly zoomed in.
# Bins in the padded region are still drawn (from the real data), they just aren't fit.
HIST_X_BUFFER_FRAC = 1.0

def _plot_hist_on_ax(
    ax: plt.Axes, values: np.ndarray, title: str, unit_label: str, n_components: int = 1,
    fit_min: Optional[float] = None, fit_max: Optional[float] = None,
    mu_ranges: Optional[List[Tuple[float, float]]] = None,
    x_buffer_frac: float = HIST_X_BUFFER_FRAC,
) -> Tuple[list, list, int]:
    """Draw a histogram + Gaussian/GMM fit onto ``ax`` and return ``(mus, stds, n_fit)``.

    ``fit_min``/``fit_max`` crop the data used for the fit to a window; the visible x-axis is
    then padded by ``x_buffer_frac`` of that span on each side so the plot isn't over-zoomed
    (real bins in the padding are still shown, they just don't feed the fit). ``mu_ranges``
    optionally constrains each GMM component's mean to a range (see ``_fit_gaussians``). The
    title carries only the axis descriptor, how it was fit, and the μ ± σ values.
    """
    all_values = np.asarray(values, dtype=float)
    cropped = fit_min is not None and fit_max is not None and fit_max > fit_min
    fit_values = all_values[(all_values >= fit_min) & (all_values <= fit_max)] if cropped else all_values

    n_fit = len(fit_values)
    mus, stds = [], []

    if n_fit > 0:
        if cropped:
            # Pad the visible x-axis by a fraction of the fit span on each side so the plot isn't
            # over-zoomed. The padding is always added (even past the data), so real bins inside it
            # still show; beyond the data it's just breathing room. Only the low edge is clamped
            # at 0 since a size axis can't go negative.
            pad = x_buffer_frac * (float(fit_max) - float(fit_min))
            vis_min = max(float(fit_min) - pad, 0.0)
            vis_max = float(fit_max) + pad
        else:
            vis_min, vis_max = np.percentile(fit_values, [0.5, 99.5])
            if vis_min >= vis_max: vis_min, vis_max = fit_values.min(), fit_values.max()

        # Histogram every value inside the (padded) window so bins outside the fit range show.
        vis_values = all_values[(all_values >= vis_min) & (all_values <= vis_max)]
        _, bins, _ = ax.hist(vis_values, bins=40, range=(vis_min, vis_max), color='steelblue', edgecolor='black', alpha=0.6)
        bw = float(bins[1] - bins[0]) if len(bins) > 1 else 1.0

        mus, stds, weights, method = _fit_gaussians(fit_values, n_components, mu_ranges)
        if mus:
            x = np.linspace(vis_min, vis_max, 300)
            total = np.zeros_like(x)
            for i in range(len(mus)):
                comp = weights[i] * np.exp(-0.5 * ((x - mus[i]) / stds[i]) ** 2) / (stds[i] * np.sqrt(2 * np.pi))
                total += comp
                if len(mus) > 1:
                    ax.plot(x, comp * n_fit * bw, '--', alpha=0.8)
            ax.plot(x, total * n_fit * bw, 'k', linewidth=1.5)
            stats = " | ".join(f"μ{'' if len(mus) == 1 else i + 1}={mus[i]:.2f}, σ={stds[i]:.2f}" for i in range(len(mus)))
            ax.set_title(f"{title} · {method} (n={n_fit})\n{stats} {unit_label}", fontsize=12)
        else:
            ax.set_title(f"{title} (n={n_fit})", fontsize=12)
        ax.set_xlim(vis_min, vis_max)
    else:
        ax.set_title(f"{title} (n=0)", fontsize=12)

    ax.set_xlabel(f"Size ({unit_label})", color='black', weight='bold')
    ax.set_ylabel("Count", color='black', weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return mus, stds, n_fit

def histogram_with_fit(
    values: np.ndarray, title: str, unit_label: str, n_components: int = 1,
    fit_min: Optional[float] = None, fit_max: Optional[float] = None,
    mu_ranges: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[plt.Figure, list, list, int]:
    """Standalone histogram + Gaussian/GMM fit figure (thin wrapper over ``_plot_hist_on_ax``)."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    mus, stds, n_fit = _plot_hist_on_ax(ax, values, title, unit_label, n_components, fit_min, fit_max, mu_ranges)
    fig.tight_layout()
    return fig, mus, stds, n_fit

def get_mu_ranges_ui(key_prefix: str, label: str, data: np.ndarray, unit: str,
                     n_components: int = 2) -> Optional[List[Tuple[float, float]]]:
    """Optional per-peak min/max inputs constraining where each GMM mean may fall.
    Returns a list of (lo, hi) windows, or ``None`` when the user leaves it unconstrained."""
    if data.size == 0 or n_components < 2:
        return None
    if not st.checkbox(f"Constrain {label} GMM peak ranges", value=False, key=f"{key_prefix}_con",
                       help="Fit each Gaussian only to data inside its window, so each μ stays in range."):
        return None
    dmin, dmax = float(np.min(data)), float(np.max(data))
    med = float(np.median(data))
    step = max((dmax - dmin) / 100.0, 1e-6)
    edges = np.linspace(dmin, dmax, n_components + 1)
    ranges: List[Tuple[float, float]] = []
    cols = st.columns(n_components * 2)
    for i in range(n_components):
        lo_default = float(edges[i]) if n_components > 2 else (dmin if i == 0 else med)
        hi_default = float(edges[i + 1]) if n_components > 2 else (med if i == 0 else dmax)
        with cols[2 * i]:
            lo = st.number_input(f"Peak {i+1} min ({unit})", value=lo_default, step=step, key=f"{key_prefix}_lo{i}")
        with cols[2 * i + 1]:
            hi = st.number_input(f"Peak {i+1} max ({unit})", value=hi_default, step=step, key=f"{key_prefix}_hi{i}")
        ranges.append((lo, hi))
    return ranges

def get_min_max_ui(key_prefix: str, label: str, data: np.ndarray) -> Tuple[float, float]:
    """Two number inputs letting the user crop the histogram x-range to a chosen window.
    Defaults to the data's full range so the histogram is unchanged until the user edits it."""
    if data.size == 0:
        return 0.0, 1.0
    d_min, d_max = float(np.min(data)), float(np.max(data))
    if d_min >= d_max:
        d_max = d_min + 1.0
    step = max((d_max - d_min) / 100.0, 1e-6)
    c1, c2 = st.columns(2)
    with c1:
        fit_min = st.number_input(f"Min {label}", value=d_min, step=step, key=f"{key_prefix}_min")
    with c2:
        fit_max = st.number_input(f"Max {label}", value=d_max, step=step, key=f"{key_prefix}_max")
    return fit_min, fit_max

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
# Effective radius + uncertainty
# ═══════════════════════════════════════════════════════════════════════════
# Below this many measurements of a given projection, the sample is flagged as too
# small to trust and the geometric assumption used instead is stated explicitly.
MIN_SHAPE_COUNT = 5

def calc_reff(volume: float) -> float:
    """Radius of the sphere with the same volume as the reported particle."""
    if not np.isfinite(volume) or volume <= 0:
        return float("nan")
    return ((3.0 * volume) / (4.0 * np.pi)) ** (1.0 / 3.0)

def reff_with_sd(vol_fn, dims: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Effective radius and its 1σ uncertainty for a volume model.

    ``vol_fn`` maps an array of dimension means to a volume; ``dims`` is a list of
    ``(mean, sd)`` pairs (the population spread of each measured dimension). The SD of
    ``r_eff`` is obtained by linearised propagation through ``vol_fn``, general enough
    for products (prisms) and sums (the tic-tac body + caps) alike:
        σ_r = |dr/dV| · sqrt(Σ_i (∂V/∂x_i · σ_i)²),   dr/dV = r / (3V).
    """
    means = np.array([m for m, _ in dims], dtype=float)
    sds = np.array([s for _, s in dims], dtype=float)
    if not np.all(np.isfinite(means)):
        return float("nan"), float("nan")
    V = float(vol_fn(means))
    r = calc_reff(V)
    if not np.isfinite(r):
        return float("nan"), float("nan")
    dr_dV = r / (3.0 * V)
    var = 0.0
    for i in range(len(means)):
        h = max(abs(means[i]) * 1e-4, 1e-6)
        bumped = means.copy()
        bumped[i] += h
        dV_dxi = (float(vol_fn(bumped)) - V) / h
        sd_i = sds[i] if np.isfinite(sds[i]) else 0.0
        var += (dr_dV * dV_dxi * sd_i) ** 2
    return r, float(np.sqrt(var))

def _mean_sd(values: np.ndarray) -> Tuple[float, float]:
    """Population mean and standard deviation, safe for tiny/empty samples."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v))


# ═══════════════════════════════════════════════════════════════════════════
# Surface area + uncertainty
# ═══════════════════════════════════════════════════════════════════════════
def measure_with_sd(fn, dims: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Evaluate ``fn(means)`` and its 1σ uncertainty by the same linearised propagation used
    for r_eff: σ_f = sqrt(Σ_i (∂f/∂x_i · σ_i)²), with each ∂f/∂x_i taken numerically. ``dims``
    is a list of ``(mean, sd)`` pairs in the argument order ``fn`` expects. General enough for
    the surface-area models below (which mix products, squares and the ellipsoid formula)."""
    means = np.array([m for m, _ in dims], dtype=float)
    sds = np.array([s for _, s in dims], dtype=float)
    if not np.all(np.isfinite(means)):
        return float("nan"), float("nan")
    f0 = float(fn(means))
    if not np.isfinite(f0):
        return float("nan"), float("nan")
    var = 0.0
    for i in range(len(means)):
        h = max(abs(means[i]) * 1e-4, 1e-6)
        bumped = means.copy()
        bumped[i] += h
        df_dxi = (float(fn(bumped)) - f0) / h
        sd_i = sds[i] if np.isfinite(sds[i]) else 0.0
        var += (df_dxi * sd_i) ** 2
    return f0, float(np.sqrt(var))

def _hexagon_perimeter(ax_x: float, ax_y: float) -> float:
    """Perimeter of the (possibly anisotropic) hexagon whose vertices sit at
    ``(ax_x·cosθ, ax_y·sinθ)`` for θ = 0,60,...,300°, the same cross-section the wireframe
    draws. Reduces to ``6·(W/2) = 3W`` for a regular hexagon of vertex-to-vertex width ``W``
    (``ax_x = ax_y = W/2``)."""
    ang = np.pi / 3.0 * np.arange(6)
    pts = np.column_stack((ax_x * np.cos(ang), ax_y * np.sin(ang)))
    seg = np.diff(np.vstack((pts, pts[:1])), axis=0)
    return float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))

def _ellipsoid_surface_area(a: float, b: float, c: float) -> float:
    """Knud Thomsen approximation to the surface of an ellipsoid with semi-axes ``a,b,c``
    (no exact closed form exists; relative error < ~1.06%). ``p = 1.6075``."""
    if a <= 0 or b <= 0:
        return 0.0
    p = 1.6075
    ap, bp, cp = a ** p, b ** p, c ** p
    return 4.0 * np.pi * (((ap * bp + ap * cp + bp * cp) / 3.0) ** (1.0 / p))


# ═══════════════════════════════════════════════════════════════════════════
# Reported geometry: 2-D projections + an orthographically-projected 3-D wireframe
# ═══════════════════════════════════════════════════════════════════════════
# 30 kx on this microscope images an 848.37 × 565.58 nm field of view, measured from
# HWT08_047L_Er02Yb05_5mmolNaOH_90min_1001.dm3, a 4008 × 2672 px frame at 0.21167 nm/px
# whose "Indicated Magnification" tag reads 30000. The summary figure defaults to
# whichever uploaded image is closest to this magnification.
MAG_30KX_FOV_NM = (848.37, 565.58)

def _circle_poly(R: float, n: int = 64) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((R * np.cos(t), R * np.sin(t)))

def projection_polys(shape_type: str, geom: Dict[str, float]) -> List[np.ndarray]:
    """Outline(s) of the 2-D projection(s) used to fit the shape, centred at the origin
    in physical units (nm). These are exactly the silhouettes the segmentation measures."""
    polys: List[np.ndarray] = []
    if shape_type == "Sphere":
        polys.append(_circle_poly(geom["R"]))
    elif shape_type == "Cube":
        h = geom["s"] / 2.0
        polys.append(np.array([[-h, -h], [h, -h], [h, h], [-h, h]]))
    elif shape_type == "Hexagonal Prism":
        W, H = geom["W"], geom["H"]
        polys.append(_hexagon_vertices(0.0, 0.0, W / 2.0, 0.0))                          # face-on hexagon
        polys.append(np.array([[-W / 2, -H / 2], [W / 2, -H / 2], [W / 2, H / 2], [-W / 2, H / 2]]))  # side-on rectangle
    elif shape_type == "Octahedron":
        maj, mn = geom["major"], geom["minor"]
        polys.append(np.array([[0, maj / 2], [mn / 2, 0], [0, -maj / 2], [-mn / 2, 0]]))
    elif shape_type == "Tic Tac":
        polys.append(_stadium_vertices(0.0, 0.0, geom["L_body"], geom["W"], geom["d"], 0.0))
    return polys

def _iso(pts3d: np.ndarray, az: float = 35.0, el: float = 20.0) -> np.ndarray:
    """Orthographic projection of 3-D points (nm) to a 2-D screen (nm), preserving scale.
    Same azimuth/elevation as the old 3-D view so the wireframe reads as a solid."""
    a, e = np.radians(az), np.radians(el)
    x, y, z = pts3d[:, 0], pts3d[:, 1], pts3d[:, 2]
    sx = x * np.cos(a) - y * np.sin(a)
    t = x * np.sin(a) + y * np.cos(a)
    sy = -t * np.sin(e) + z * np.cos(e)
    return np.column_stack((sx, sy))

def _wire_edges_3d(shape_type: str, geom: Dict[str, float]) -> List[np.ndarray]:
    """3-D wireframe of the reported geometry as a list of polylines (Nx3 nm, origin-centred)."""
    segs: List[np.ndarray] = []
    if shape_type == "Sphere":
        R = geom["R"]; u = np.linspace(0, 2 * np.pi, 60)
        for lat in np.linspace(-np.pi / 2, np.pi / 2, 7):
            segs.append(np.column_stack((R * np.cos(lat) * np.cos(u), R * np.cos(lat) * np.sin(u), R * np.sin(lat) * np.ones_like(u))))
        w = np.linspace(-np.pi / 2, np.pi / 2, 60)
        for lon in np.linspace(0, 2 * np.pi, 13):
            segs.append(np.column_stack((R * np.cos(w) * np.cos(lon), R * np.cos(w) * np.sin(lon), R * np.sin(w))))
    elif shape_type == "Cube":
        h = geom["s"] / 2.0
        pts = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)])
        for i in range(8):
            for j in range(i + 1, 8):
                if int(np.sum(np.abs(pts[i] - pts[j]) > 1e-9)) == 1:
                    segs.append(np.vstack((pts[i], pts[j])))
    elif shape_type in ("Hexagonal Prism", "Tic Tac"):
        if shape_type == "Hexagonal Prism":
            ax_x = ax_y = geom["W"] / 2.0; z0, z1 = -geom["H"] / 2.0, geom["H"] / 2.0; d = 0.0
        else:
            ax_x, ax_y = geom["W"] / 2.0, geom["T"] / 2.0; z0, z1 = -geom["L_body"] / 2.0, geom["L_body"] / 2.0; d = geom["d"]
        ang = np.pi / 3.0 * np.arange(6)
        hx, hy = ax_x * np.cos(ang), ax_y * np.sin(ang)
        for z in (z0, z1):
            segs.append(np.column_stack((np.append(hx, hx[0]), np.append(hy, hy[0]), z * np.ones(7))))
        for k in range(6):
            segs.append(np.array([[hx[k], hy[k], z0], [hx[k], hy[k], z1]]))
        if d > 0:                                             # half-ellipsoid caps for the tic-tac
            # Keep it legible: 6 ribs aligned with the hexagon vertices (continuing the prism
            # edges up to the pole) plus 2 latitude hoops, rather than a dense converging fan.
            psi = np.linspace(0, np.pi / 2.0, 14)
            tt = np.linspace(0, 2 * np.pi, 48)
            for sign, ztip in ((1.0, z1), (-1.0, z0)):
                for beta in ang:                              # ribs over the 6 hexagon vertices
                    segs.append(np.column_stack((ax_x * np.sin(psi) * np.cos(beta),
                                                 ax_y * np.sin(psi) * np.sin(beta),
                                                 ztip + sign * d * np.cos(psi))))
                for frac in (1.0 / 3.0, 2.0 / 3.0):           # two hoops around the dome
                    ps = (np.pi / 2.0) * frac
                    zc = ztip + sign * d * np.cos(ps)
                    segs.append(np.column_stack((ax_x * np.sin(ps) * np.cos(tt),
                                                 ax_y * np.sin(ps) * np.sin(tt),
                                                 np.full(tt.shape, zc))))
    elif shape_type == "Octahedron":
        a, c = geom["minor"] / 2.0, geom["major"] / 2.0
        eq = [(a, 0, 0), (0, a, 0), (-a, 0, 0), (0, -a, 0)]
        for ap in ((0, 0, c), (0, 0, -c)):
            for e in eq:
                segs.append(np.array([ap, e], dtype=float))
        segs.append(np.array(eq + [eq[0]], dtype=float))
    return segs

# Miller-index label drawn above each 2-D projection, in projection_polys() order.
_PROJ_MILLER = {"Hexagonal Prism": ["(0001)", "(1000)"]}

def _fit_drawables(shape_type: str, geom: Dict[str, float]) -> List[dict]:
    """Everything drawn in the 'Average Shape Dimensions Fit' group, laid out left→right:
    each 2-D projection first, then the projected 3-D wireframe last. Polylines are in nm;
    projections may carry a Miller-index ``label`` drawn above them."""
    items: List[dict] = []
    labels = _PROJ_MILLER.get(shape_type, [])
    for idx, poly in enumerate(projection_polys(shape_type, geom)):
        items.append({"polylines": [np.asarray(poly, dtype=float)], "kind": "proj", "closed": True,
                      "label": labels[idx] if idx < len(labels) else None})
    edges = _wire_edges_3d(shape_type, geom)
    if edges:
        items.append({"polylines": [_iso(e) for e in edges], "kind": "wire", "closed": False, "label": None})
    return items

def _layout_fits(items: List[dict], scale: float) -> Tuple[list, float, float]:
    """Lay the fit drawables out left-to-right on a common bottom baseline. Returns
    ``(placed, total_width, max_height)`` in target units, where ``scale`` = target units
    per nm and each placed poly has x growing right, y growing up from the baseline."""
    if not items:
        return [], 0.0, 0.0
    bbs, widths, heights = [], [], []
    for it in items:
        allpts = np.vstack(it["polylines"])
        mn, mx = allpts.min(0), allpts.max(0)
        bbs.append((mn, mx)); widths.append((mx[0] - mn[0]) * scale); heights.append((mx[1] - mn[1]) * scale)
    gap = 0.3 * max(widths)
    placed, xcur = [], 0.0
    for it, (mn, _mx), wpx in zip(items, bbs, widths):
        drawn = []
        for pl in it["polylines"]:
            q = (np.asarray(pl, dtype=float) - mn) * scale     # origin at bbox min, scaled
            q[:, 0] += xcur                                    # horizontal offset for this item
            drawn.append(q)
        placed.append({"polys": drawn, "kind": it["kind"], "closed": it["closed"], "label": it.get("label")})
        xcur += wpx + gap
    return placed, xcur - gap, max(heights)

def _nice_bar_nm(fov_nm: float) -> float:
    """Largest 'nice' round length not exceeding ~18% of the field of view, so the bar scales
    sensibly at any magnification (an 848 nm 30 kx FOV → 100 nm; a 50 nm FOV → 5 nm)."""
    target = fov_nm * 0.18
    for b in (1000, 500, 200, 100, 50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1):
        if b <= target:
            return float(b)
    return round(target, 2)

def _estimate_mag_kx(fov_width_nm: float) -> float:
    """Estimated microscope magnification (in kx) for an image, assuming magnification is
    linear in inverse field-of-view and pinned to the known 30 kx reference FOV width.
    A wider FOV ⇒ lower magnification, so mag ∝ 1/FOV_width."""
    if not (np.isfinite(fov_width_nm) and fov_width_nm > 0):
        return float("nan")
    return 30.0 * MAG_30KX_FOV_NM[0] / fov_width_nm

def _add_scale_bar(ax, w_px: float, h_px: float, nm_per_px: float, unit: str,
                   mag_kx: float = float("nan"), font_size: float = 13.0) -> None:
    """Draw a 'nice'-length white scale bar (lower-left) on an auto-fitting grey panel. When
    ``mag_kx`` is finite, the estimated magnification is added on a smaller second line inside
    the same grey frame (the frame auto-expands to accommodate it)."""
    bar_nm = _nice_bar_nm(w_px * nm_per_px)
    bar_px = bar_nm / nm_per_px
    sb = AnchoredSizeBar(
        ax.transData, bar_px, f"{bar_nm:g} {unit}", loc="lower left",
        pad=0.4, borderpad=0.6, sep=5, color="white", frameon=True,
        size_vertical=max(0.012 * h_px, 3), fontproperties=FontProperties(size=font_size, weight="bold"),
    )
    sb.patch.set(facecolor="0.15", edgecolor="none", alpha=0.5)   # grey box auto-fits its contents
    if np.isfinite(mag_kx) and mag_kx > 0:
        mag_txt = f"~{mag_kx:.0f} kx" if mag_kx >= 10 else f"~{mag_kx:.1f} kx"
        mag_area = TextArea(mag_txt, textprops=dict(color="white", weight="bold",
                                                    size=max(font_size - 4.0, 7.0)))
        sb._box._children.append(mag_area)   # extra smaller line stacked under the label
    ax.add_artist(sb)

_FIT_TITLE = "Average Shape \nDimensions Fit:"
# IBM colorblind-safe categorical palette ("ibm_dark"): each fit drawable gets its own colour.
IBM_DARK = ["#648FFF", "#DC267F", "#FFB000", "#FE6100", "#785EF0"]

def _translucent_panel(ax, x0, y0, x1, y1, w_px, h_px, alpha: float = 0.5) -> None:
    """A grey semi-transparent background rectangle, clamped inside the image bounds."""
    x0, x1 = sorted((max(0.0, min(x0, w_px)), max(0.0, min(x1, w_px))))
    y0, y1 = sorted((max(0.0, min(y0, h_px)), max(0.0, min(y1, h_px))))
    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   facecolor="0.15", edgecolor="none", alpha=alpha, zorder=5))

def draw_tem_with_fits(ax, tem: dict, shape_type: str, geom: Dict[str, float], unit: str,
                       px_per_in: float = 120.0) -> None:
    """Grayscale TEM image with (lower-right) the reported 2-D projections then 3-D wireframe
    drawn to scale under an 'Average Shape Dimensions Fit' heading on a grey panel, and
    (lower-left) a white scale bar on an auto-fitting grey panel. ``px_per_in`` is the image's
    data-pixels-per-inch on the page, used to size the heading box to the (point-sized) text."""
    img = np.asarray(tem["data"], dtype=float)
    nmpp = float(tem.get("nm_per_px", float("nan")))
    zmin, zmax = np.percentile(img, [0.5, 99.5])
    ax.imshow(img, cmap="gray", vmin=zmin, vmax=zmax)
    h_px, w_px = img.shape[:2]
    ax.set_xlim(0, w_px); ax.set_ylim(h_px, 0)
    ax.axis("off")
    # Wrap a long file name so the title stays within the panel instead of clipping/overrunning.
    ax.set_title("\n".join(textwrap.wrap(f"TEM: {tem.get('name', '')}", width=42)), fontsize=10)

    if not (np.isfinite(nmpp) and nmpp > 0):
        ax.text(0.5, 0.04, "Uncalibrated image, cannot render fits to scale",
                transform=ax.transAxes, ha="center", color="yellow", fontsize=9)
        return

    # ---- Lower-right: reported geometry (2-D projections then wireframe) to scale ----
    placed, total_w, max_h = _layout_fits(_fit_drawables(shape_type, geom), 1.0 / nmpp)
    if placed:
        title_fs = 10
        lines = _FIT_TITLE.split("\n")
        # Text is sized in points; convert to data-px via px_per_in so the box fits it snugly.
        title_w = max(len(s) for s in lines) * (title_fs / 72.0) * 0.56 * px_per_in
        title_h = len(lines) * (title_fs / 72.0) * 1.22 * px_per_in
        pad_x, pad_y = 0.008 * w_px, 0.006 * h_px
        # Reserve a strip above the shapes for the Miller-index labels (if any).
        label_fs = 9
        label_h = ((label_fs / 72.0) * 1.3 * px_per_in) if any(it.get("label") for it in placed) else 0.0
        content_w = max(total_w, title_w)
        box_right = w_px - 0.02 * w_px
        box_left = max(0.02 * w_px, box_right - content_w - 2 * pad_x)
        baseline = h_px - 0.05 * h_px                            # bottom of the shapes
        title_base = baseline - max_h - label_h - 0.008 * h_px   # heading baseline (above labels)
        _translucent_panel(ax, box_left, title_base - title_h - pad_y,
                            box_right, baseline + pad_y, w_px, h_px)
        shapes_left = box_left + pad_x + max(0.0, ((box_right - box_left) - 2 * pad_x - total_w) / 2.0)
        # Line weight tracks how large the shapes render on the page (small shapes ⇒ thinner
        # strokes), clamped to a narrow band so it only ever nudges the default weight down.
        lw_scale = float(np.clip((max_h / max(px_per_in, 1e-6)) / 0.6, 0.7, 1.0))
        for i, item in enumerate(placed):
            color = IBM_DARK[i % len(IBM_DARK)]
            lw = (1.0 if item["kind"] == "wire" else 1.8) * lw_scale
            xs_all, ys_all = [], []
            for poly in item["polys"]:
                xs = shapes_left + poly[:, 0]
                ys = baseline - poly[:, 1]                       # y grows up ⇒ smaller row
                ax.add_patch(patches.Polygon(np.column_stack((xs, ys)), closed=item["closed"],
                                             fill=False, edgecolor=color, lw=lw, zorder=6))
                xs_all.append(xs); ys_all.append(ys)
            if item.get("label"):
                xs_all, ys_all = np.concatenate(xs_all), np.concatenate(ys_all)
                ax.text((xs_all.min() + xs_all.max()) / 2.0, ys_all.min() - 0.006 * h_px, item["label"],
                        color=color, fontsize=label_fs, weight="bold", ha="center", va="bottom", zorder=6)
        ax.text((box_left + box_right) / 2.0, title_base, _FIT_TITLE, color="white", fontsize=title_fs,
                weight="bold", va="bottom", ha="center", zorder=6, linespacing=1.15)

    # ---- Lower-left: white scale bar + estimated magnification on a grey panel ----
    _add_scale_bar(ax, w_px, h_px, nmpp, unit, mag_kx=_estimate_mag_kx(w_px * nmpp))

def draw_fits_no_tem(ax, shape_type: str, geom: Dict[str, float], unit: str) -> None:
    """Fallback when no calibrated TEM image is available: the same fit drawables laid out
    in physical units on a blank axes (no scale bar to key against)."""
    placed, total_w, max_h = _layout_fits(_fit_drawables(shape_type, geom), 1.0)
    for i, item in enumerate(placed):
        color = IBM_DARK[i % len(IBM_DARK)]
        lw = 1.0 if item["kind"] == "wire" else 1.8
        xs_all, ys_all = [], []
        for poly in item["polys"]:
            ax.add_patch(patches.Polygon(poly, closed=item["closed"], fill=False, edgecolor=color, lw=lw))
            xs_all.append(poly[:, 0]); ys_all.append(poly[:, 1])
        if item.get("label"):
            xs_all, ys_all = np.concatenate(xs_all), np.concatenate(ys_all)
            ax.text((xs_all.min() + xs_all.max()) / 2.0, ys_all.max() + 0.02 * max_h, item["label"],
                    color=color, fontsize=10, weight="bold", ha="center", va="bottom")
    if total_w > 0:
        mx = 0.08 * max(total_w, max_h)
        ax.set_xlim(-mx, total_w + mx); ax.set_ylim(-mx, max_h + 0.14 * max_h)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{_FIT_TITLE} (no calibrated image, arbitrary scale, {unit})", fontsize=10)


# ═══════════════════════════════════════════════════════════════════════════
# Dimension guide: annotated example of what each measured dimension means
# ═══════════════════════════════════════════════════════════════════════════
def _dim_arrow(ax, p0, p1, label, lab_off=(0.0, 0.0), ha="center", va="center", color="#222222"):
    """Double-headed measurement arrow from ``p0`` to ``p1`` with a bold label offset from its
    midpoint by ``lab_off``."""
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.4, shrinkA=0, shrinkB=0))
    mx, my = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
    ax.text(mx + lab_off[0], my + lab_off[1], label, color=color, fontsize=9,
            ha=ha, va=va, weight="bold", linespacing=1.1)

def _dim_outline(ax, verts, color):
    ax.add_patch(patches.Polygon(verts, closed=True, fill=True, facecolor=color,
                                 edgecolor=color, alpha=0.18, lw=1.8))

def _dim_guide(ax, p0, p1, color="0.55"):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], ls=(0, (4, 3)), color=color, lw=0.9, zorder=1)

def _dim_finish(ax, x0, x1, y0, y1, title):
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=10, weight="bold")

def build_dimension_guide_figure(shape_type: str) -> plt.Figure:
    """A small annotated diagram of the projection(s) the tool measures, with every reported
    dimension called out, so 'width' vs 'body length' vs 'total length' etc. are unambiguous."""
    c0, c1 = IBM_DARK[0], IBM_DARK[1]
    if shape_type == "Sphere":
        fig, ax = plt.subplots(figsize=(3.2, 3.3), dpi=130, layout="constrained")
        _dim_outline(ax, _circle_poly(1.0), c0)
        _dim_arrow(ax, (-1, 0), (1, 0), "Diameter (D)", (0, 0.13), va="bottom")
        _dim_finish(ax, -1.5, 1.5, -1.4, 1.5, "Sphere → circle")

    elif shape_type == "Cube":
        fig, ax = plt.subplots(figsize=(3.2, 3.4), dpi=130, layout="constrained")
        s = 2.0
        _dim_outline(ax, np.array([[-s/2, -s/2], [s/2, -s/2], [s/2, s/2], [-s/2, s/2]]), c0)
        _dim_arrow(ax, (-s/2, -s/2 - 0.35), (s/2, -s/2 - 0.35), "Side length (s)", (0, -0.16), va="top")
        _dim_finish(ax, -1.7, 1.7, -2.0, 1.4, "Cube → square")

    elif shape_type == "Hexagonal Prism":
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.5), dpi=130, layout="constrained")
        ax = axes[0]
        _dim_outline(ax, _hexagon_vertices(0, 0, 1.0, 0.0), c0)
        _dim_guide(ax, (-1, 0), (-1, -1.35)); _dim_guide(ax, (1, 0), (1, -1.35))
        _dim_arrow(ax, (-1, -1.35), (1, -1.35), "Width (W)\nvertex-to-vertex", (0, -0.18), va="top")
        _dim_finish(ax, -1.7, 1.7, -2.2, 1.3, "Face-on (0001)")
        ax = axes[1]
        W, Hh = 2.0, 3.0
        _dim_outline(ax, np.array([[-W/2, -Hh/2], [W/2, -Hh/2], [W/2, Hh/2], [-W/2, Hh/2]]), c1)
        _dim_arrow(ax, (-W/2 - 0.4, -Hh/2), (-W/2 - 0.4, Hh/2), "Height (H)", (-0.15, 0), ha="right")
        _dim_arrow(ax, (-W/2, -Hh/2 - 0.4), (W/2, -Hh/2 - 0.4), "Width (W)", (0, -0.16), va="top")
        _dim_finish(ax, -2.5, 1.8, -2.5, 2.0, "Side-on (10-10)")
        fig.suptitle("Hexagonal prism dimensions", fontsize=11, weight="bold")

    elif shape_type == "Octahedron":
        fig, ax = plt.subplots(figsize=(3.6, 3.7), dpi=130, layout="constrained")
        maj, mn = 3.0, 2.0
        _dim_outline(ax, np.array([[0, maj/2], [mn/2, 0], [0, -maj/2], [-mn/2, 0]]), c0)
        _dim_arrow(ax, (mn/2 + 0.4, -maj/2), (mn/2 + 0.4, maj/2), "Major axis", (0.13, 0), ha="left")
        _dim_arrow(ax, (-mn/2, -maj/2 - 0.4), (mn/2, -maj/2 - 0.4), "Minor axis", (0, -0.16), va="top")
        _dim_finish(ax, -1.9, 2.6, -2.4, 2.1, "Octahedron → rhombus projection")

    elif shape_type == "Tic Tac":
        fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=130, layout="constrained")
        W, L_body, d = 1.6, 2.0, 0.7
        half, L_total = W / 2.0, L_body + 2.0 * d
        _dim_outline(ax, _stadium_vertices(0, 0, L_body, W, d, 0.0), c0)
        # Dashed guides mark where the straight body ends and the arc caps begin.
        _dim_guide(ax, (-L_body/2, -half), (-L_body/2, half + 0.95))
        _dim_guide(ax, (L_body/2, -half), (L_body/2, half + 0.95))
        _dim_arrow(ax, (L_total/2 + 0.4, -half), (L_total/2 + 0.4, half), "Width (W)", (0.13, 0), ha="left")
        _dim_arrow(ax, (L_body/2, half + 0.6), (L_total/2, half + 0.6), "Cap depth (d)", (0.0, 0.15), va="bottom")
        _dim_arrow(ax, (-L_total/2, -half - 0.55), (L_total/2, -half - 0.55), "Total length (L)", (0, -0.17), va="top")
        _dim_arrow(ax, (-L_body/2, -half - 1.05), (L_body/2, -half - 1.05), "Body length (L_body)", (0, -0.17), va="top")
        _dim_finish(ax, -L_total/2 - 1.3, L_total/2 + 1.7, -half - 1.8, half + 1.3,
                    "Tic Tac → stadium (straight body + arc caps)")

    else:
        fig, ax = plt.subplots(figsize=(3.0, 2.0), dpi=130, layout="constrained")
        ax.text(0.5, 0.5, "No dimension guide", ha="center", va="center"); ax.axis("off")
    return fig

def render_dimension_guide(shape_type: str) -> None:
    """Collapsible panel showing the annotated dimension diagram for the selected shape."""
    with st.expander(f"What each {shape_type} dimension means", expanded=False):
        fig = build_dimension_guide_figure(shape_type)
        st.pyplot(fig)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Annotation-driven auto-tuning
# ═══════════════════════════════════════════════════════════════════════════
# The user draws a small box, exhaustively annotates every particle inside it (one drawn line
# per particle per dimension), and the tuner searches the segmentation parameters for the
# setting whose measurements best reproduce those annotations.
#
# Why the box must be *exhaustively* annotated: the score penalises detected particles inside
# the box that no annotation matches. Without that penalty the search has a degenerate optimum:
# over-segment the image into many small blobs and something will overlap every annotation, so
# recall is perfect and nothing is ever charged for the spurious detections.
#
# Cost of one wrong particle (missed or spurious) relative to the mean relative size error. At
# 1.0, getting 1 particle in 10 wrong costs the same as a 10% size error on every particle.
GT_COUNT_PENALTY = 1.0
# A hand-drawn midpoint may land just off the particle; look this far (as a fraction of the
# annotated length) for a nearby watershed region before calling the annotation a miss.
GT_SNAP_FRAC = 0.25

def _dim_annotation_specs(shape_type: str) -> List[Dict[str, Any]]:
    """The dimensions that can be annotated for ``shape_type``, each with the diagram (outline +
    the single arrow being annotated) shown while drawing.

    Geometry deliberately mirrors ``build_dimension_guide_figure`` so the drawing prompt and the
    reference guide cannot drift apart.
    """
    if shape_type == "Sphere":
        return [dict(key="diameters", label="Diameter (D)", proj="Sphere → circle",
                     outline=_circle_poly(1.0), arrow=((-1, 0), (1, 0)), guides=(),
                     lim=(-1.5, 1.5, -1.4, 1.5), lab_off=(0, 0.13), va="bottom")]
    if shape_type == "Cube":
        s = 2.0
        sq = np.array([[-s/2, -s/2], [s/2, -s/2], [s/2, s/2], [-s/2, s/2]])
        return [dict(key="side_lengths", label="Side length (s)", proj="Cube → square",
                     outline=sq, arrow=((-s/2, -s/2 - 0.35), (s/2, -s/2 - 0.35)), guides=(),
                     lim=(-1.7, 1.7, -2.0, 1.4), lab_off=(0, -0.16), va="top")]
    if shape_type == "Hexagonal Prism":
        W, Hh = 2.0, 3.0
        rect = np.array([[-W/2, -Hh/2], [W/2, -Hh/2], [W/2, Hh/2], [-W/2, Hh/2]])
        rlim = (-2.5, 1.8, -2.5, 2.0)
        return [
            dict(key="hex_widths", label="Width (W)\nvertex-to-vertex", proj="Face-on (0001)",
                 outline=_hexagon_vertices(0, 0, 1.0, 0.0), arrow=((-1, -1.35), (1, -1.35)),
                 guides=(((-1, 0), (-1, -1.35)), ((1, 0), (1, -1.35))),
                 lim=(-1.7, 1.7, -2.2, 1.3), lab_off=(0, -0.18), va="top"),
            dict(key="hex_heights", label="Height (H)", proj="Side-on (10-10)", outline=rect,
                 arrow=((-W/2 - 0.4, -Hh/2), (-W/2 - 0.4, Hh/2)), guides=(),
                 lim=rlim, lab_off=(-0.15, 0), ha="right"),
            dict(key="hex_rect_widths", label="Width (W)", proj="Side-on (10-10)", outline=rect,
                 arrow=((-W/2, -Hh/2 - 0.4), (W/2, -Hh/2 - 0.4)), guides=(),
                 lim=rlim, lab_off=(0, -0.16), va="top"),
        ]
    if shape_type == "Octahedron":
        maj, mn = 3.0, 2.0
        rh = np.array([[0, maj/2], [mn/2, 0], [0, -maj/2], [-mn/2, 0]])
        lim = (-1.9, 2.6, -2.4, 2.1)
        return [
            dict(key="oct_major", label="Major axis", proj="Octahedron → rhombus", outline=rh,
                 arrow=((mn/2 + 0.4, -maj/2), (mn/2 + 0.4, maj/2)), guides=(),
                 lim=lim, lab_off=(0.13, 0), ha="left"),
            dict(key="oct_minor", label="Minor axis", proj="Octahedron → rhombus", outline=rh,
                 arrow=((-mn/2, -maj/2 - 0.4), (mn/2, -maj/2 - 0.4)), guides=(),
                 lim=lim, lab_off=(0, -0.16), va="top"),
        ]
    if shape_type == "Tic Tac":
        W, L_body, d = 1.6, 2.0, 0.7
        half, L_tot = W / 2.0, L_body + 2.0 * d
        st_out = _stadium_vertices(0, 0, L_body, W, d, 0.0)
        caps = (((-L_body/2, -half), (-L_body/2, half + 0.95)),
                ((L_body/2, -half), (L_body/2, half + 0.95)))
        lim = (-L_tot/2 - 1.3, L_tot/2 + 1.7, -half - 1.8, half + 1.3)
        return [
            dict(key="tic_tac_major", label="Total length (L)", proj="Tic Tac → stadium",
                 outline=st_out, arrow=((-L_tot/2, -half - 0.55), (L_tot/2, -half - 0.55)),
                 guides=(), lim=lim, lab_off=(0, -0.17), va="top"),
            dict(key="tic_tac_width", label="Width (W)", proj="Tic Tac → stadium", outline=st_out,
                 arrow=((L_tot/2 + 0.4, -half), (L_tot/2 + 0.4, half)), guides=(),
                 lim=lim, lab_off=(0.13, 0), ha="left"),
            dict(key="tic_tac_body_length", label="Body length (L_body)", proj="Tic Tac → stadium",
                 outline=st_out, arrow=((-L_body/2, -half - 1.05), (L_body/2, -half - 1.05)),
                 guides=caps, lim=lim, lab_off=(0, -0.17), va="top"),
            dict(key="tic_tac_cap_depth", label="Cap depth (d)", proj="Tic Tac → stadium",
                 outline=st_out, arrow=((L_body/2, half + 0.6), (L_tot/2, half + 0.6)),
                 guides=caps, lim=lim, lab_off=(0, 0.15), va="bottom"),
        ]
    return []

def build_dim_annotation_figure(shape_type: str, dim_key: str) -> Optional[plt.Figure]:
    """Diagram for a single annotation pass: the projection outline with only the dimension
    currently being drawn arrowed and highlighted."""
    spec = next((s for s in _dim_annotation_specs(shape_type) if s["key"] == dim_key), None)
    if spec is None:
        return None
    fig, ax = plt.subplots(figsize=(3.3, 3.1), dpi=130, layout="constrained")
    _dim_outline(ax, spec["outline"], IBM_DARK[0])
    for g0, g1 in spec["guides"]:
        _dim_guide(ax, g0, g1)
    _dim_arrow(ax, spec["arrow"][0], spec["arrow"][1], spec["label"], spec["lab_off"],
               ha=spec.get("ha", "center"), va=spec.get("va", "center"), color=IBM_DARK[1])
    x0, x1, y0, y1 = spec["lim"]
    _dim_finish(ax, x0, x1, y0, y1, f"Draw this distance\n{spec['proj']}")
    return fig

def _canvas_lines_to_truth(json_data: Optional[dict], scale: float, nm_per_px: float, unit: str,
                           origin: Tuple[float, float] = (0.0, 0.0)) -> List[Dict[str, Any]]:
    """Convert drawn canvas lines into ``{"mid": (x_px, y_px), "value": <length in unit>}``.

    Results are in full-ROI pixel coordinates: ``scale`` is ROI px per canvas px and ``origin``
    is the ROI pixel the canvas's top-left corner corresponds to (non-zero when the canvas shows
    a zoomed crop rather than the whole ROI).
    """
    sf = nm_per_px if unit == "nm" else 1.0
    out: List[Dict[str, Any]] = []
    for obj in (json_data or {}).get("objects", []):
        if obj.get("type") != "line":
            continue
        w = float(obj.get("width", 0.0)) * float(obj.get("scaleX", 1.0)) * scale
        h = float(obj.get("height", 0.0)) * float(obj.get("scaleY", 1.0)) * scale
        # streamlit-drawable-canvas builds line objects with originX/originY = "center", so
        # fabric reports `left`/`top` as the line's MIDPOINT, not its bounding-box corner. Its
        # endpoints are opposite corners of that box, so the length is the box diagonal,
        # both quantities are independent of which way the user dragged.
        mx = origin[0] + float(obj.get("left", 0.0)) * scale
        my = origin[1] + float(obj.get("top", 0.0)) * scale
        len_px = float(np.hypot(w, h))
        if len_px < 2.0:      # a stray click, not a measurement
            continue
        out.append({"mid": (mx, my), "value": len_px * sf})
    return out

def _canvas_rect_to_box(json_data: Optional[dict], scale: float) -> Optional[Tuple[float, float, float, float]]:
    """The last drawn rectangle as ``(x0, y0, x1, y1)`` in full-ROI pixel coordinates.

    Rectangles are built with originX/originY = "left"/"top", so ``left``/``top`` really is the
    corner; the result is still normalised so a box dragged in any direction comes back with
    x0 < x1 and y0 < y1.
    """
    rects = [o for o in (json_data or {}).get("objects", []) if o.get("type") == "rect"]
    if not rects:
        return None
    o = rects[-1]
    w = float(o.get("width", 0.0)) * float(o.get("scaleX", 1.0)) * scale
    h = float(o.get("height", 0.0)) * float(o.get("scaleY", 1.0)) * scale
    x, y = float(o.get("left", 0.0)) * scale, float(o.get("top", 0.0)) * scale
    x0, x1 = sorted((x, x + w))
    y0, y1 = sorted((y, y + h))
    if (x1 - x0) < 8 or (y1 - y0) < 8:
        return None
    return (x0, y0, x1, y1)

def _label_at(labels_ws: np.ndarray, x: float, y: float, search_px: int) -> int:
    """Watershed label under a hand-drawn point. If the point lands on background, the most
    common non-background label within ``search_px`` wins, so a slightly-off line still
    matches its particle."""
    h, w = labels_ws.shape
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return 0
    lab = int(labels_ws[yi, xi])
    if lab:
        return lab
    r = max(int(search_px), 1)
    win = labels_ws[max(0, yi - r):min(h, yi + r + 1), max(0, xi - r):min(w, xi + r + 1)]
    nz = win[win > 0]
    if nz.size == 0:
        return 0
    vals, counts = np.unique(nz, return_counts=True)
    return int(vals[int(np.argmax(counts))])

def score_against_annotations(
    labels_ws: np.ndarray, objects: List[Dict[str, Any]],
    truth: Dict[str, List[Dict[str, Any]]], box_px: Tuple[float, float, float, float],
    nm_per_px: float, unit: str,
) -> Dict[str, Any]:
    """Score a segmentation against exhaustively-annotated ground truth inside ``box_px``.

    Returns a scalar ``score`` (lower is better) plus a per-dimension breakdown and, for each
    annotation, whether it was matched; the caller uses those flags to spot annotations that no
    parameter setting can ever reach.
    """
    x0, y0, x1, y1 = box_px
    sf = nm_per_px if unit == "nm" else 1.0
    per_dim: Dict[str, Any] = {}
    total, n_dims = 0.0, 0
    for dim, lines in truth.items():
        if not lines:
            continue
        # Candidates: detections of this dimension whose centre lies in the annotated box.
        cands = {o["label"]: o for o in objects
                 if dim in o["dims"] and x0 <= o["cx"] <= x1 and y0 <= o["cy"] <= y1}
        errs: List[float] = []
        flags: List[bool] = []
        matched: set = set()
        n_missed = 0
        for ln in lines:
            snap = max(int(round(GT_SNAP_FRAC * ln["value"] / max(sf, 1e-9))), 1)
            lab = _label_at(labels_ws, ln["mid"][0], ln["mid"][1], snap)
            o = cands.get(lab)
            # `lab in matched` means two annotations fell in one region: the segmentation merged
            # two particles, so one of them counts as missed.
            if o is None or lab in matched:
                n_missed += 1
                flags.append(False)
                continue
            matched.add(lab)
            errs.append(abs(o["dims"][dim] - ln["value"]) / max(ln["value"], 1e-9))
            flags.append(True)
        n_extra = max(len(cands) - len(matched), 0)
        # No match at all scores as a 100% size error: bad, but bounded, so a setting that finds
        # nothing still ranks below one that finds particles at slightly wrong sizes.
        size_err = float(np.mean(errs)) if errs else 1.0
        count_err = (n_missed + n_extra) / len(lines)
        per_dim[dim] = {
            "n_gt": len(lines), "n_matched": len(matched), "n_missed": n_missed,
            "n_extra": n_extra, "size_err": size_err,
            "median_size_err": float(np.median(errs)) if errs else float("nan"),
            "flags": flags,
        }
        total += size_err + GT_COUNT_PENALTY * count_err
        n_dims += 1
    return {"score": total / n_dims if n_dims else float("inf"), "per_dim": per_dim}

# Search design, measured on real annotated data rather than assumed. The objective turns out to
# be jagged and multi-basin; thresh_offset can score well at 0.85 and at 1.00 while scoring
# poorly at 0.95 in between, and the two best settings found sat in unrelated regions of
# (contrast_clip, thresh_offset). Three consequences, all baked in below:
#   * gradient/simplex methods are useless (plateaus and cliffs), so this is a grid sweep;
#   * smoothing_sigma and thresh_offset need fine sampling and are refined *jointly*, since
#     descending them one at a time gets trapped between basins;
#   * refinement runs from several coarse starts, because good basins are isolated.
TUNE_SIGMA_GRID = (1.0, 1.25, 1.5, 1.75, 2.25)
TUNE_CLIP_GRID = (0.01, 0.04, 0.08)
TUNE_THRESH_GRID = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.15)
# Peak spacing matters only relative to particle size, so candidates are fractions of the
# largest annotated dimension rather than absolute values. Empirically only the small end of
# this range ever wins (larger spacings under-split clumps), hence the downward bias.
TUNE_PEAK_FRACS = (0.15, 0.22, 0.30, 0.40, 0.55)
# Its effect is close to separable from the other three (every top-scoring setting observed
# shared the same peak spacing), so the main sweep holds it fixed here and it gets its own
# pass afterwards. That turns a 4-D product into two much cheaper sweeps.
TUNE_PEAK_SEED_FRAC = 0.30
TUNE_N_PEAK_STARTS = 4      # coarse settings that get the dedicated peak-spacing sweep
TUNE_N_STARTS = 3           # coarse settings that get local refinement
# Parameter bounds, matching the manual sliders so a tuned value is always dialable by hand.
TUNE_BOUNDS = {"smoothing_sigma": (0.5, 5.0), "contrast_clip": (0.01, 0.10),
               "thresh_offset": (0.5, 1.5), "min_peak_distance": (0.5, 1e9)}
# Local refinement: offsets applied one parameter at a time, then a joint sigma × thresh grid.
TUNE_REFINE_STEPS = {"smoothing_sigma": (-0.25, -0.10, 0.10, 0.25),
                     "contrast_clip": (-0.02, -0.01, 0.01, 0.02),
                     "thresh_offset": (-0.05, -0.025, 0.025, 0.05)}
TUNE_REFINE_PEAK_FACTORS = (0.85, 0.925, 1.075, 1.15)
TUNE_JOINT_SIGMA_OFFSETS = (-0.25, 0.0, 0.25)
TUNE_JOINT_THRESH_OFFSETS = (-0.10, -0.05, 0.0, 0.05, 0.10)
# A tuned setting is only useful if the user can also dial it in by hand, so every value the
# search visits is snapped to the resolution of the corresponding manual widget. Snapping on the
# way *in* (rather than rounding the winner on the way out) keeps the reported score exactly
# equal to the score of the parameters that get applied.
TUNE_SNAP = {"smoothing_sigma": 0.05, "contrast_clip": 0.01,
             "thresh_offset": 0.025, "min_peak_distance": 0.1}

def _clamp(name: str, v: float) -> float:
    lo, hi = TUNE_BOUNDS[name]
    return float(min(max(v, lo), hi))

def _snap(name: str, v: float) -> float:
    step = TUNE_SNAP[name]
    return _clamp(name, round(round(float(v) / step) * step, 4))

def autotune_parameters(
    data: np.ndarray, nm_per_px: float, shape_type: str, min_size_value: float,
    measurement_unit: str, truth: Dict[str, List[Dict[str, Any]]],
    box_px: Tuple[float, float, float, float], progress=None,
) -> Dict[str, Any]:
    """Search segmentation parameters for the setting that best reproduces ``truth``.

    Runs on the whole ROI rather than a crop of the annotated box: Otsu's threshold depends on
    the histogram of whatever it is given, so scoring on a tight crop would tune
    ``thresh_offset`` against a histogram the full-image run never sees. Only detections inside
    ``box_px`` are scored.

    Stages are cached by the parameters they actually depend on: preprocessing by
    (sigma, clip), the binary mask additionally by thresh_offset, which is where the bulk of
    the speed-up comes from, since preprocessing alone is roughly half the cost of one
    segmentation.
    """
    sf = nm_per_px if measurement_unit == "nm" else 1.0
    img_h, img_w = data.shape
    pre_cache: Dict[tuple, np.ndarray] = {}
    mask_cache: Dict[tuple, np.ndarray] = {}
    memo: Dict[tuple, Dict[str, Any]] = {}
    # Annotations that no evaluated setting ever matched, usually rejected by
    # classify_projection's fixed thresholds rather than by any parameter under search.
    ever_matched: Dict[str, np.ndarray] = {d: np.zeros(len(v), dtype=bool) for d, v in truth.items() if v}

    def _eval(smoothing_sigma: float, contrast_clip: float, thresh_offset: float,
              min_peak_distance: float) -> Dict[str, Any]:
        smoothing_sigma = _snap("smoothing_sigma", smoothing_sigma)
        contrast_clip = _snap("contrast_clip", contrast_clip)
        thresh_offset = _snap("thresh_offset", thresh_offset)
        min_peak_distance = _snap("min_peak_distance", min_peak_distance)
        key = (smoothing_sigma, contrast_clip, thresh_offset, min_peak_distance)
        if key in memo:
            return memo[key]
        pk = key[:2]
        if pk not in pre_cache:
            pre_cache[pk] = preprocess_image(data, smoothing_sigma, contrast_clip)
        mk = key[:3]
        if mk not in mask_cache:
            mask_cache[mk] = _binary_mask_from(pre_cache[pk], thresh_offset)
        labels_ws = _watershed_labels(
            mask_cache[mk], _min_peak_px(min_peak_distance, measurement_unit, nm_per_px))
        _, _, objects = _measure_labels(labels_ws, shape_type, min_size_value, sf, img_h, img_w)
        sc = score_against_annotations(labels_ws, objects, truth, box_px, nm_per_px, measurement_unit)
        for dim, info in sc["per_dim"].items():
            ever_matched[dim] |= np.array(info["flags"], dtype=bool)
        out = {"params": {"smoothing_sigma": smoothing_sigma, "contrast_clip": contrast_clip,
                          "thresh_offset": thresh_offset, "min_peak_distance": min_peak_distance},
               "score": sc["score"], "per_dim": sc["per_dim"], "n_objects": len(objects)}
        memo[key] = out
        return out

    # Peak-spacing candidates scale with the largest annotated dimension, which is what sets how
    # far apart two particle centres actually are.
    medians = [float(np.median([ln["value"] for ln in v])) for v in truth.values() if v]
    ref = max(medians) if medians else 10.0
    peak_grid = sorted({round(_clamp("min_peak_distance", f * ref), 2) for f in TUNE_PEAK_FRACS})
    peak_seed = _clamp("min_peak_distance", TUNE_PEAK_SEED_FRAC * ref)

    n_total = (len(TUNE_SIGMA_GRID) * len(TUNE_CLIP_GRID) * len(TUNE_THRESH_GRID)
               + TUNE_N_PEAK_STARTS * len(peak_grid)
               + TUNE_N_STARTS * (sum(len(v) for v in TUNE_REFINE_STEPS.values())
                                  + len(TUNE_REFINE_PEAK_FACTORS)
                                  + len(TUNE_JOINT_SIGMA_OFFSETS) * len(TUNE_JOINT_THRESH_OFFSETS)))
    done = 0

    def _tick():
        nonlocal done
        done += 1
        if progress:
            progress(min(done / max(n_total, 1), 1.0))

    # Stage 1: sweep sigma × clip × thresh at a fixed, deliberately small peak spacing.
    coarse: List[Dict[str, Any]] = []
    for sigma in TUNE_SIGMA_GRID:
        for clip in TUNE_CLIP_GRID:
            for thresh in TUNE_THRESH_GRID:
                coarse.append(_eval(sigma, clip, thresh, peak_seed))
                _tick()
    coarse.sort(key=lambda r: r["score"])
    coarse_score = coarse[0]["score"]

    # Stage 2: give the most promising settings their own peak-spacing sweep.
    pool = list(coarse)
    for r0 in coarse[:TUNE_N_PEAK_STARTS]:
        for peak in peak_grid:
            cand = dict(r0["params"]); cand["min_peak_distance"] = peak
            pool.append(_eval(**cand))
            _tick()
    pool.sort(key=lambda r: r["score"])

    # Stage 3: refine from several starts, since the good basins are isolated. Each start gets
    # one-at-a-time steps plus a joint sigma × thresh grid (the two coupled, jagged parameters).
    starts, seen = [], set()
    for r in pool:
        sig = (round(r["params"]["smoothing_sigma"], 3), round(r["params"]["contrast_clip"], 4),
               round(r["params"]["thresh_offset"], 4))
        if sig in seen:
            continue
        seen.add(sig)
        starts.append(r)
        if len(starts) >= TUNE_N_STARTS:
            break

    best = pool[0]
    for start in starts:
        local = start
        for name, steps in TUNE_REFINE_STEPS.items():
            for step in steps:
                cand = dict(local["params"])
                cand[name] = _clamp(name, cand[name] + step)
                r = _eval(**cand)
                _tick()
                if r["score"] < local["score"]:
                    local = r
        for factor in TUNE_REFINE_PEAK_FACTORS:
            cand = dict(local["params"])
            cand["min_peak_distance"] = _clamp("min_peak_distance",
                                               cand["min_peak_distance"] * factor)
            r = _eval(**cand)
            _tick()
            if r["score"] < local["score"]:
                local = r
        base = dict(local["params"])
        for ds in TUNE_JOINT_SIGMA_OFFSETS:
            for dt in TUNE_JOINT_THRESH_OFFSETS:
                cand = dict(base)
                cand["smoothing_sigma"] = _clamp("smoothing_sigma", base["smoothing_sigma"] + ds)
                cand["thresh_offset"] = _clamp("thresh_offset", base["thresh_offset"] + dt)
                r = _eval(**cand)
                _tick()
                if r["score"] < local["score"]:
                    local = r
        if local["score"] < best["score"]:
            best = local

    if progress:
        progress(1.0)
    return {**best, "n_evals": len(memo), "coarse_score": coarse_score,
            "unreachable": {d: int((~m).sum()) for d, m in ever_matched.items()}}


# ═══════════════════════════════════════════════════════════════════════════
# Annotation-driven auto-tuning: UI
# ═══════════════════════════════════════════════════════════════════════════
ANNO_CANVAS_W = 560          # annotation canvases are drawn large: 1 canvas px ≈ 1.7 ROI px at
                             # the default 200 nm ROI, so hand precision, not display
                             # resolution, is what limits the annotations
TUNE_PARAM_KEYS = {"smoothing_sigma": "tem_sigma", "contrast_clip": "tem_clip",
                   "thresh_offset": "tem_thresh", "min_peak_distance": "tem_peak"}
# Staging slot for tuned parameters on their way into the manual widgets; see
# consume_pending_tuned_params.
TUNE_PENDING_KEY = "tem_pending_params"

def consume_pending_tuned_params() -> bool:
    """Move any parameters staged by the auto-tuner into the manual widgets' session-state
    entries, returning whether anything was applied.

    Must be called *before* those widgets are instantiated: Streamlit raises
    ``StreamlitAPIException`` if a widget's session-state entry is written after the widget
    exists, so the auto-tuner's Apply button cannot assign them directly; it stages them and
    reruns, and this consumes the staged values at the one point in the script where writing
    them is legal.
    """
    pending = st.session_state.get(TUNE_PENDING_KEY)
    if not pending:
        return False
    del st.session_state[TUNE_PENDING_KEY]
    for name, widget_key in TUNE_PARAM_KEYS.items():
        if name in pending:
            st.session_state[widget_key] = pending[name]
    return True

def _crop_bounds(shape: Tuple[int, ...],
                 box: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    """Integer, in-bounds pixel bounds for a drawn box.

    Single source of truth for both the zoomed background image and the mapping of canvas
    coordinates back to ROI pixels; if those two disagreed, every annotation would be offset.
    """
    h, w = shape[0], shape[1]
    x0 = int(max(0, np.floor(box[0])));  y0 = int(max(0, np.floor(box[1])))
    x1 = int(min(w, np.ceil(box[2])));   y1 = int(min(h, np.ceil(box[3])))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)

def _roi_to_pil(roi: np.ndarray, width: int,
                crop: Optional[Tuple[float, float, float, float]] = None,
                box: Optional[Tuple[float, float, float, float]] = None):
    """The ROI, or a ``crop`` of it, as an 8-bit PIL image scaled to ``width``.

    Contrast is stretched over whatever region is actually displayed, so a zoomed crop is as
    legible as possible rather than inheriting the full ROI's percentiles. ``box`` outlines a
    region given in the *displayed* image's pixel coordinates (used for the locator thumbnail).
    """
    if crop is not None:
        x0, y0, x1, y1 = _crop_bounds(roi.shape, crop)
        roi = roi[y0:y1, x0:x1]
    lo, hi = np.percentile(roi, [0.5, 99.5])
    img = np.clip((roi - lo) / max(hi - lo, 1e-12), 0, 1)
    pil = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
    h = max(int(round(width * roi.shape[0] / roi.shape[1])), 1)
    pil = pil.resize((width, h), resample=Image.Resampling.LANCZOS)
    if box is not None:
        s = width / roi.shape[1]
        ImageDraw.Draw(pil).rectangle(
            [box[0] * s, box[1] * s, box[2] * s, box[3] * s], outline=(220, 30, 120), width=3)
    return pil

def _plot_annotation_review(ax, roi, seg, truth, box, nm_per_px, unit) -> None:
    """The tuned segmentation over the ROI, with the annotation box and every annotated midpoint
    marked, so a mismatch is visible rather than just a number in a table."""
    plot_annotated_image(ax, roi, seg["draw_shapes"], nm_per_px, unit, mask_alpha=0.35)
    x0, y0, x1, y1 = box
    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=IBM_DARK[1], lw=2.0, zorder=6))
    for lines in truth.values():
        if lines:
            ax.scatter([l["mid"][0] for l in lines], [l["mid"][1] for l in lines],
                       s=26, marker="+", c=IBM_DARK[1], linewidths=1.4, zorder=7)

def _truth_fingerprint(truth: Dict[str, List[Dict[str, Any]]],
                       box: Tuple[float, float, float, float], min_feature: float) -> str:
    """Identifies the exact inputs a tuning run was based on, so a displayed result can be
    flagged as stale once the annotations, the box, or the size filter change."""
    parts = [f"{min_feature:.4f}", ",".join(f"{v:.1f}" for v in box)]
    for dim in sorted(truth):
        parts.append(dim + ":" + ";".join(
            f"{l['value']:.2f}@{l['mid'][0]:.0f},{l['mid'][1]:.0f}" for l in truth[dim]))
    return "|".join(parts)

def _annotation_token(file_name: str, roi_size: float, shape_type: str) -> str:
    """Annotations are pixel coordinates into one specific ROI, so they are scoped to the image,
    the ROI size that produced it, and the shape whose dimensions were drawn. Changing any of
    those invalidates them; the token is embedded in the canvas keys to force a clean slate."""
    return f"{file_name}|{roi_size:g}|{shape_type}"

def annotation_tuning_ui(
    roi_data: np.ndarray, nm_per_px: float, shape_type: str, unit: str,
    min_feature: float, token: str, live_seg: Optional[Dict[str, Any]] = None,
) -> None:
    """Draw-and-tune panel: annotate a small box exhaustively, then search the segmentation
    parameters for the setting that best reproduces the annotations."""
    if not _HAS_CANVAS:
        st.info("Install `streamlit-drawable-canvas` to enable annotation-driven tuning.")
        return
    specs = _dim_annotation_specs(shape_type)
    if not specs:
        st.info(f"No annotatable dimensions defined for {shape_type}.")
        return

    st.markdown(
        "Draw a box over a **small, representative** patch, then mark **every** particle inside "
        "it: one line per particle, for each dimension in turn. Exhaustive labelling is what "
        "lets the tuner count spurious detections; if particles inside the box are left "
        "unmarked, the search is rewarded for over-segmenting. Around **10 particles** is "
        "plenty."
    )

    scale = roi_data.shape[1] / ANNO_CANVAS_W
    st.markdown("**Step A. Box the patch you will annotate**")
    box_canvas = st_canvas(
        fill_color="rgba(220, 30, 120, 0.10)", stroke_width=2, stroke_color="#DC267F",
        background_image=_roi_to_pil(roi_data, ANNO_CANVAS_W), update_streamlit=True,
        height=int(round(ANNO_CANVAS_W * roi_data.shape[0] / roi_data.shape[1])),
        width=ANNO_CANVAS_W, drawing_mode="rect", key=f"anno_box_{token}",
    )
    drawn_box = _canvas_rect_to_box(box_canvas.json_data if box_canvas else None, scale)
    if drawn_box is None:
        st.info("Draw a box to continue.")
        return
    # Work in the integer crop bounds from here on, so the zoomed image, the coordinate mapping
    # and the scored region are all describing exactly the same pixels.
    cx0, cy0, cx1, cy1 = _crop_bounds(roi_data.shape, drawn_box)
    box = (float(cx0), float(cy0), float(cx1), float(cy1))
    sf_unit = nm_per_px if unit == "nm" else 1.0
    box_nm = ((cx1 - cx0) * sf_unit, (cy1 - cy0) * sf_unit)
    n_in_box = 0
    if live_seg is not None:
        n_in_box = sum(1 for o in live_seg["objects"]
                       if box[0] <= o["cx"] <= box[2] and box[1] <= o["cy"] <= box[3])
    st.caption(f"Box {box_nm[0]:.0f} × {box_nm[1]:.0f} {unit} · the current manual settings find "
               f"**{n_in_box}** particle(s) inside it.")

    st.markdown("**Step B. Annotate each dimension**")
    # Annotate on a zoom of the box, not the whole ROI: at the default 200 nm ROI this is
    # typically a 2–5× magnification over Step A, which is what makes hand-drawn lines precise
    # enough for the size errors to mean anything.
    bg = _roi_to_pil(roi_data, ANNO_CANVAS_W, crop=box)
    scale_b = (cx1 - cx0) / ANNO_CANVAS_W
    origin = (float(cx0), float(cy0))
    # Both views are ANNO_CANVAS_W wide, so the magnification gained is simply the ratio of the
    # ROI width to the box width.
    st.caption(f"Zoomed to the box: **{roi_data.shape[1] / (cx1 - cx0):.1f}×** the Step A view "
               f"(1 screen px ≈ {scale_b * sf_unit:.2f} {unit}).")
    truth: Dict[str, List[Dict[str, Any]]] = {}
    tabs = st.tabs([s["label"].replace("\n", " ") for s in specs])
    for tab, spec in zip(tabs, specs):
        with tab:
            c_fig, c_draw = st.columns([1, 2])
            with c_fig:
                fig = build_dim_annotation_figure(shape_type, spec["key"])
                if fig is not None:
                    st.pyplot(fig)
                    plt.close(fig)
                st.caption(f"Drag one line per particle, end to end, along **{spec['label']}**. "
                           "Leave a tab empty to skip that dimension.")
                st.image(_roi_to_pil(roi_data, 190, box=box), caption="Region being annotated")
            with c_draw:
                res = st_canvas(
                    fill_color="rgba(0,0,0,0)", stroke_width=2, stroke_color="#FFB000",
                    background_image=bg, update_streamlit=True, height=bg.size[1],
                    width=bg.size[0], drawing_mode="line",
                    key=f"anno_{spec['key']}_{token}",
                )
                lines = _canvas_lines_to_truth(res.json_data if res else None, scale_b,
                                               nm_per_px, unit, origin=origin)
                # The canvas covers exactly the box, but clamp anyway so a line nudged over the
                # edge can't be scored against a region the user never committed to labelling.
                lines = [l for l in lines
                         if box[0] <= l["mid"][0] <= box[2] and box[1] <= l["mid"][1] <= box[3]]
                if lines:
                    truth[spec["key"]] = lines
                    vals = np.array([l["value"] for l in lines])
                    st.caption(f"{len(lines)} line(s) · median {np.median(vals):.1f} {unit}, "
                               f"range {vals.min():.1f}–{vals.max():.1f} {unit}")

    n_anno = sum(len(v) for v in truth.values())
    if n_anno == 0:
        st.info("Annotate at least one dimension to enable tuning.")
        return
    st.caption(f"**{n_anno}** annotation(s) across {len(truth)} dimension(s).")

    if st.button("Run auto-tune", type="primary", key=f"anno_run_{token}"):
        bar = st.progress(0.0, text="Searching parameters...")
        try:
            res = autotune_parameters(
                data=roi_data, nm_per_px=nm_per_px, shape_type=shape_type,
                min_size_value=min_feature, measurement_unit=unit, truth=truth, box_px=box,
                progress=lambda f: bar.progress(f, text=f"Searching parameters... {f*100:.0f}%"),
            )
        finally:
            bar.empty()
        st.session_state[f"tuned_{token}"] = {
            "result": res, "truth": truth, "box": box,
            "fp": _truth_fingerprint(truth, box, min_feature)}

    stored = st.session_state.get(f"tuned_{token}")
    if not stored:
        return
    res, truth_used, box_used = stored["result"], stored["truth"], stored["box"]

    st.markdown("**Step C. Review and apply**")
    if stored["fp"] != _truth_fingerprint(truth, box, min_feature):
        st.warning("Annotations, box, or min feature size have changed since this result was "
                   "computed; re-run the auto-tune to match. The figures below still describe "
                   "the earlier annotation set.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Score (lower is better)", f"{res['score']:.3f}",
              delta=f"{res['score'] - res['coarse_score']:+.3f} vs coarse", delta_color="inverse")
    matched = sum(i["n_matched"] for i in res["per_dim"].values())
    m2.metric("Annotations matched", f"{matched}/{n_anno}")
    m3.metric("Evaluations", res["n_evals"])

    rows = []
    for dim, i in res["per_dim"].items():
        label = next((s["label"].replace("\n", " ") for s in specs if s["key"] == dim), dim)
        rows.append({"Dimension": label, "n": i["n_gt"], "Matched": i["n_matched"],
                     "Missed": i["n_missed"], "Spurious": i["n_extra"],
                     "Mean size error": f"{i['size_err']*100:.1f}%",
                     "Median size error": f"{i['median_size_err']*100:.1f}%"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    unreachable = {d: n for d, n in res.get("unreachable", {}).items() if n}
    if unreachable:
        st.warning(
            "Never matched at **any** setting tried: "
            + ", ".join(f"{n} × {d}" for d, n in unreachable.items())
            + ". These particles are being rejected by the fixed shape-classification "
              "thresholds in `classify_projection`, not by anything the search can change. "
              "so the parameters below are fitted to the remaining annotations only."
        )

    p = res["params"]
    st.code("\n".join([
        f"Image Smoothing (Sigma)   {p['smoothing_sigma']:.2f}",
        f"Contrast Enhancement      {p['contrast_clip']:.2f}",
        f"Threshold Sensitivity     {p['thresh_offset']:.3f}",
        f"Min peak distance         {p['min_peak_distance']:.1f} {unit}",
    ]), language=None)

    tuned_seg = segment_and_measure(
        data=roi_data, nm_per_px=nm_per_px, shape_type=shape_type,
        min_size_value=min_feature, measurement_unit=unit, **p)
    fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=150)
    _plot_annotation_review(ax, roi_data, tuned_seg, truth_used, box_used, nm_per_px, unit)
    fig.tight_layout(pad=0)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Tuned fit over the ROI. Pink box = annotated region, pink crosses = annotated "
               "particles.")

    if st.button("Apply these parameters to the sliders", key=f"anno_apply_{token}"):
        # Streamlit forbids writing a widget's session_state entry once that widget exists, and
        # this button is necessarily evaluated later in the script than the sliders. So stage the
        # values under a non-widget key; consume_pending_tuned_params() applies them on the next
        # run, before the widgets are created.
        st.session_state[TUNE_PENDING_KEY] = dict(p)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# Comprehensive summary figure + export
# ═══════════════════════════════════════════════════════════════════════════
# Locked summary-figure geometry: every exported summary has these exact page dimensions (one
# fixed aspect ratio) regardless of magnification, crop, shape, or histogram count, so a wall of
# them tiles cleanly. The TEM image is letterboxed (equal aspect) into its fixed panel; the
# histograms share the full right-column height (so a single histogram is as tall as the TEM
# panel, not a small strip). Export must NOT use bbox_inches="tight" or the crop would
# reintroduce variable sizes.
SUMMARY_FIG_W_IN = 12.0
SUMMARY_FIG_H_IN = 7.0
SUMMARY_TEM_FRAC = 0.55          # width fraction given to the TEM panel
SUMMARY_TITLE_IN = 1.0           # approx page band reserved for the (2-line) sup-title
# Fixed, legible font sizes (points), identical across every summary since the page size is
# locked, so text reads at the same physical size no matter the shape or histogram count.
SUMMARY_SUPTITLE_FS = 16
SUMMARY_NOTES_FS = 9.5

def build_summary_figure(
    shape_type: str, hist_specs: List[dict], geom: Dict[str, float],
    reff: float, reff_sd: float, unit: str, prefix: str, notes: str = "",
    tem: Optional[dict] = None, area: float = float("nan"), area_sd: float = float("nan"),
) -> plt.Figure:
    """One figure at a locked page size/aspect: the TEM image with the reported geometry (2-D
    projections + 3-D wireframe) overlaid to scale on the left, and every histogram + fit in
    fixed-height rows on the right. The file name is the main heading; r_eff ± SD sits bold
    beneath it. The fixed geometry (see the SUMMARY_* constants) makes many summaries tile
    cleanly no matter the magnification, shape, or histogram count."""
    n = max(len(hist_specs), 1)
    have_tem = tem is not None and tem.get("data") is not None

    # constrained_layout only redistributes space *within* the fixed figsize, so the output page
    # stays exactly SUMMARY_FIG_W_IN × SUMMARY_FIG_H_IN.
    fig = plt.figure(figsize=(SUMMARY_FIG_W_IN, SUMMARY_FIG_H_IN), dpi=150, layout="constrained")
    outer = fig.add_gridspec(1, 2, width_ratios=[SUMMARY_TEM_FRAC, 1.0 - SUMMARY_TEM_FRAC])

    ax_left = fig.add_subplot(outer[0, 0])
    if have_tem:
        img = np.asarray(tem["data"]); h_px, w_px = img.shape[:2]
        # The equal-aspect image is letterboxed inside the fixed panel; estimate its drawn width
        # so draw_tem_with_fits can size the heading box to its (point-based) text.
        tem_w_in = SUMMARY_FIG_W_IN * SUMMARY_TEM_FRAC
        tem_h_in = max(SUMMARY_FIG_H_IN - SUMMARY_TITLE_IN, 1.0)
        img_aspect = (w_px / h_px) if h_px else 1.4
        drawn_w_in = tem_w_in if img_aspect >= (tem_w_in / tem_h_in) else tem_h_in * img_aspect
        px_per_in = w_px / max(drawn_w_in, 1e-6)
        draw_tem_with_fits(ax_left, tem, shape_type, geom, unit, px_per_in=px_per_in)
    else:
        draw_fits_no_tem(ax_left, shape_type, geom, unit)

    # Histograms fill the full column height (so a lone histogram is as tall as the TEM panel
    # rather than a small strip); the locked page size holds regardless of the count.
    right = outer[0, 1].subgridspec(n, 1, hspace=0.45)
    for i, spec in enumerate(hist_specs):
        ax = fig.add_subplot(right[i])
        _plot_hist_on_ax(
            ax, spec["values"], spec["title"], unit,
            spec.get("n_components", 1), spec.get("fit_min"), spec.get("fit_max"), spec.get("mu_ranges"),
        )

    # File name is the heading; r_eff ± SD and surface area ± SD share the second line, same
    # bold size, one suptitle. Wrap a long prefix/notes so nothing clips at the fixed page edge
    # (no bbox expansion now).
    head = "\n".join(textwrap.wrap(prefix, width=72)) or prefix
    stat_parts = []
    if np.isfinite(reff):
        sd_txt = f" ± {reff_sd:.2f}" if np.isfinite(reff_sd) else ""
        stat_parts.append(f"$r_{{eff}}$ = {reff:.2f}{sd_txt} {unit}")
    if np.isfinite(area):
        asd_txt = f" ± {area_sd:.1f}" if np.isfinite(area_sd) else ""
        stat_parts.append(f"S.A. = {area:.1f}{asd_txt} {unit}$^2$")
    if stat_parts:
        head += "\n" + "     ".join(stat_parts)
    fig.suptitle(head, fontsize=SUMMARY_SUPTITLE_FS, weight="bold")
    if notes:
        # supxlabel is managed by constrained_layout, so it gets its own reserved band below the
        # axes x-labels instead of overlapping them.
        fig.supxlabel("\n".join(textwrap.wrap(notes, width=140)), fontsize=SUMMARY_NOTES_FS, style="italic")
    return fig

# Selectable summary-figure zoom levels; each maps (via the linear 30 kx reference) to a
# field-of-view crop size in nm.
ZOOM_OPTIONS_KX = (10.0, 30.0, 67.0, 100.0)

def _fov_for_kx(kx: float) -> Tuple[float, float]:
    """Field of view (width, height) in nm at magnification ``kx``, linear in 1/FOV against the
    known 30 kx reference (848.37 × 565.58 nm): a higher kx ⇒ a smaller FOV."""
    f = 30.0 / kx
    return MAG_30KX_FOV_NM[0] * f, MAG_30KX_FOV_NM[1] * f

def _tem_center_crop(tem: dict, crop_w: int, crop_h: int) -> dict:
    data = np.asarray(tem["data"]); h_px, w_px = data.shape[:2]
    c0, r0 = (w_px - crop_w) // 2, (h_px - crop_h) // 2
    return {**tem, "data": data[r0:r0 + crop_h, c0:c0 + crop_w]}

def _in_range_object_centroids(objects: list, ranges: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """(row, col) centroids of objects whose measured dimensions all fall inside the fit ranges.
    An object counts only if it has at least one dimension keyed in ``ranges`` and every such
    dimension is within its window. Returns an (N, 2) array (possibly empty)."""
    pts = []
    for ob in objects:
        matched = [(k, v) for k, v in ob.get("dims", {}).items() if k in ranges]
        if matched and all(ranges[k][0] <= v <= ranges[k][1] for k, v in matched):
            pts.append((ob["cy"], ob["cx"]))
    return np.array(pts, dtype=float) if pts else np.zeros((0, 2))

def _crop_tem_to_fov(tem: Optional[dict], crop_w_nm: float, crop_h_nm: float,
                     hist_specs: List[dict]) -> Optional[dict]:
    """Crop ``tem`` to a ``crop_w_nm × crop_h_nm`` window placed over the region richest in
    watershed objects whose dimensions lie within the histogram (Gaussian-fit) ranges.
    Failsafes: returns the image unchanged when uncalibrated or already at/below the requested
    FOV; falls back to a centered crop when no in-range object positions are available."""
    if tem is None or tem.get("data") is None:
        return tem
    data = np.asarray(tem["data"])
    nmpp = float(tem.get("nm_per_px", float("nan")))
    if not (np.isfinite(nmpp) and nmpp > 0):
        return tem
    h_px, w_px = data.shape[:2]
    crop_w = int(round(crop_w_nm / nmpp))
    crop_h = int(round(crop_h_nm / nmpp))
    if crop_w < 1 or crop_h < 1 or crop_w >= w_px or crop_h >= h_px:
        return tem   # failsafe: the whole image is already at/below the requested FOV

    ranges = {k: (float(s["fit_min"]), float(s["fit_max"]))
              for s in hist_specs for k in s.get("keys", [])
              if s.get("fit_min") is not None and s.get("fit_max") is not None
              and s["fit_max"] > s["fit_min"]}
    cents = _in_range_object_centroids(tem.get("objects", []), ranges)
    if cents.shape[0] == 0:
        return _tem_center_crop(tem, crop_w, crop_h)   # failsafe: nothing to localise on

    # Densest window: box-count the in-range centroids (a uniform filter is a sliding window
    # sum up to a constant), then centre the crop on the max, clamped to stay inside the image
    # mirroring extract_representative_roi's approach.
    pmap = np.zeros((h_px, w_px))
    rr = np.clip(cents[:, 0].astype(int), 0, h_px - 1)
    cc = np.clip(cents[:, 1].astype(int), 0, w_px - 1)
    np.add.at(pmap, (rr, cc), 1.0)
    counts = uniform_filter(pmap, size=(crop_h, crop_w), mode="constant")
    # Many windows can tie at the max object count (a plateau); among them pick the centre
    # closest to the in-range centroid cloud, so the particles sit centred in the crop rather
    # than at a corner of that plateau.
    ys, xs = np.where(counts >= counts.max() - 1e-9)
    tgt = cents.mean(axis=0)
    j = int(np.argmin((ys - tgt[0]) ** 2 + (xs - tgt[1]) ** 2))
    y, x = int(ys[j]), int(xs[j])
    r0 = max(0, min(y - crop_h // 2, h_px - crop_h))
    c0 = max(0, min(x - crop_w // 2, w_px - crop_w))
    return {**tem, "data": data[r0:r0 + crop_h, c0:c0 + crop_w]}

def _select_tem_for_summary(results: Optional[list], key: str) -> Optional[dict]:
    """Dropdown to pick the TEM image shown in the summary; defaults to the image whose
    field of view is closest to 30 kx (848.37 × 565.58 nm)."""
    if not results:
        return None
    names = [r["name"] for r in results]
    fovs = []
    for r in results:
        nmpp = float(r.get("nm_per_px", float("nan")))
        w = int(r["data"].shape[1])
        fovs.append(w * nmpp if np.isfinite(nmpp) else float("nan"))
    finite = [(i, f) for i, f in enumerate(fovs) if np.isfinite(f)]
    default_idx = min(finite, key=lambda t: abs(t[1] - MAG_30KX_FOV_NM[0]))[0] if finite else 0
    sel = st.selectbox(
        "TEM image for summary figure", names, index=default_idx, key=f"temsel_{key}",
        help=f"Defaults to the image nearest 30 kx ({MAG_30KX_FOV_NM[0]:.0f} × {MAG_30KX_FOV_NM[1]:.0f} nm FOV).",
    )
    r = next(rr for rr in results if rr["name"] == sel)
    return {"data": r["data"], "nm_per_px": float(r.get("nm_per_px", float("nan"))),
            "name": sel, "objects": r.get("objects", [])}

def summary_export_ui(
    shape_type: str, hist_specs: List[dict], geom: Dict[str, float],
    reff: float, reff_sd: float, unit: str, prefix: str, notes: str = "",
    results: Optional[list] = None, area: float = float("nan"), area_sd: float = float("nan"),
) -> None:
    """A checkbox that builds the comprehensive summary figure on demand, previews it,
    and offers a high-resolution PNG download."""
    if not st.checkbox("Build comprehensive summary figure", value=True, key=f"summary_{shape_type}"):
        st.caption("Tick to assemble one figure with all histograms, r_eff ± SD, and a TEM image "
                   "with the reported geometry (2-D projections + 3-D wireframe) overlaid to scale.")
        return
    tem = _select_tem_for_summary(results, shape_type)
    # Read the zoom choice before building the figure; the dropdown itself is rendered below
    # the figure (Streamlit persists it by key, so changing it reruns and regenerates the crop).
    zoom_key = f"summary_zoom_{shape_type}"
    zoom_labels = ["Full image"] + [f"{int(k)} kx" for k in ZOOM_OPTIONS_KX]
    zoom_default = "30 kx"
    zoom_choice = st.session_state.get(zoom_key, zoom_default)
    if zoom_choice != "Full image" and unit == "nm":
        kx = float(zoom_choice.split()[0])
        crop_w_nm, crop_h_nm = _fov_for_kx(kx)
        tem = _crop_tem_to_fov(tem, crop_w_nm, crop_h_nm, hist_specs)
    fig = build_summary_figure(shape_type, hist_specs, geom, reff, reff_sd, unit, prefix, notes,
                               tem, area, area_sd)
    st.pyplot(fig, use_container_width=True)
    st.selectbox(
        "Crop field of view", zoom_labels, index=zoom_labels.index(zoom_default), key=zoom_key,
        help="Zoom the summary TEM image to the field of view for the chosen magnification, placed "
             "over the region densest in particles whose sizes fall within the histogram fit "
             "range(s). Falls back to the full image when it is already at/below that FOV.",
    )
    buf = io.BytesIO()
    # No bbox_inches="tight": keep the exact locked page size so exported summaries tile cleanly.
    fig.savefig(buf, format="png", dpi=300)
    st.download_button(
        "Export summary figure (PNG)", buf.getvalue(),
        f"{prefix}_{shape_type.replace(' ', '_').lower()}_summary.png", "image/png",
    )
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Main Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════
def run() -> None:
    # Note: st.set_page_config and the page title are owned by app.py.
    if "full_run_complete" not in st.session_state:
        st.session_state.full_run_complete = False

    with st.sidebar:
        st.header("Inputs")
        accepted_types = ["dm3"] if _HAS_DM3 else []
        if _HAS_H5PY: accepted_types.append("emd")

        if _is_streamlit_cloud():
            st.warning(
                "**Running on Streamlit Cloud.** Memory here is limited. "
                "Upload only **1–2 TEM images** at a time or the app may crash. "
                "For larger batches, run the app locally."
            )

        files = file_uploader_with_clear("Upload TEM image(s)", key="tem_uploads", accept_multiple_files=True, type=accepted_types, on_change=_clear_cache, on_clear=_clear_cache)
        shape_type = st.selectbox("Particle shape", list(SHAPE_CHOICES), index=0)
        
        if not files:
            st.warning("Upload files to begin.")
            tune_file = None
        else:
            tune_file_name = st.selectbox("Select image for tuning:", [f.name for f in files])
            tune_file = next(f for f in files if f.name == tune_file_name)

    # Placed outside the sidebar block so the hint lands in the main panel.
    if tune_file is None:
        st.info("Upload one or more TEM images (.dm3 or .emd) in the sidebar to begin.")
        return

    tab_tune, tab_results = st.tabs(["Rapid Tuning (ROI)", "Full Batch Results"])
    
    tem_tune = read_tem_file(tune_file.getvalue(), tune_file.name)
    calibrated = not np.isnan(tem_tune.nm_per_px)
    unit = "nm" if calibrated else "px"

    with tab_tune:
        st.subheader("Tune watershed parameters")
        render_dimension_guide(shape_type)

        c_roi, c_size, c_thresh, c_peak = st.columns(4)
        with c_roi:
            roi_size = st.number_input(f"ROI Size ({unit})", min_value=50, max_value=2000, value=200, step=50)
        with c_size:
            min_feature = st.number_input(f"Min feature size ({unit})", min_value=0.0, value=5.0, step=0.5)
        # These four widgets are keyed and seeded from session state rather than given a
        # `value=`, so the auto-tuner can write a tuned setting straight into them without
        # Streamlit complaining that a default and a session-state value were both supplied.
        # Their steps match TUNE_SNAP, so every tuned value lands exactly on a widget stop.
        for _k, _v in (("tem_thresh", 1.0), ("tem_sigma", 1.5), ("tem_clip", 0.01),
                       ("tem_peak", 10.0 if calibrated else 5.0)):
            if _k not in st.session_state:
                st.session_state[_k] = _v
        # Anything the auto-tuner staged has to land here, before the widgets below exist.
        if consume_pending_tuned_params():
            st.success("Auto-tuned parameters applied to the controls below.")

        with c_thresh:
            thresh_offset = st.slider(
                "Threshold Sensitivity", min_value=0.5, max_value=1.5, step=0.025,
                key="tem_thresh",
                help="Low = Picks up faint particles. High = Stricter boundaries.")
        with c_peak:
            min_peak_distance = st.number_input(
                f"Min peak distance ({unit})", min_value=0.0, step=1.0, key="tem_peak",
                help="Minimum spacing between particle centers. Increase if particles are being "
                     "overly fragmented; decrease if touching particles aren't splitting.")

        roi_data = extract_representative_roi(tem_tune.data, tem_tune.nm_per_px, roi_size)

        with st.expander("Advanced Settings", expanded=False):
            adv_c1, adv_c2 = st.columns(2)
            with adv_c1:
                smoothing_sigma = st.slider(
                    "Image Smoothing (Sigma)", min_value=0.5, max_value=5.0, step=0.05,
                    key="tem_sigma", help="Blurs image to reduce background noise.")
            with adv_c2:
                contrast_clip = st.slider(
                    "Contrast Enhancement", min_value=0.01, max_value=0.10, step=0.01,
                    key="tem_clip",
                    help="Enhances contrast to separate particles from the background.")

        seg_roi = segment_and_measure(
            data=roi_data, nm_per_px=tem_tune.nm_per_px, shape_type=shape_type,
            min_size_value=min_feature, measurement_unit=unit,
            smoothing_sigma=smoothing_sigma, contrast_clip=contrast_clip,
            thresh_offset=thresh_offset, min_peak_distance=min_peak_distance
        )

        plot_c1, plot_c2 = st.columns(2)
        with plot_c1:
            st.caption("Watershed Mask (ROI)")
            st.image(seg_roi["rgb_ws"], use_container_width=True)
        with plot_c2:
            st.caption("Fitted Shapes (ROI)")
            plot_slot = st.empty()
            roi_mask_alpha = st.slider("Mask Opacity", 0.0, 1.0, 0.4, 0.05,
                                       key="roi_mask_alpha",
                                       help="Adjust the transparency of the fitted shape overlays to see how well they fit.")
            fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
            plot_annotated_image(ax, seg_roi["data"], seg_roi["draw_shapes"],
                                 seg_roi["nm_per_px"], seg_roi["unit"], mask_alpha=roi_mask_alpha)
            fig.tight_layout(pad=0)
            plot_slot.pyplot(fig)
            plt.close(fig)

        with st.expander("Auto-tune from annotations", expanded=False):
            annotation_tuning_ui(
                roi_data=roi_data, nm_per_px=tem_tune.nm_per_px, shape_type=shape_type,
                unit=unit, min_feature=min_feature,
                token=_annotation_token(tune_file_name, roi_size, shape_type),
                live_seg=seg_roi,
            )

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
                st.caption("Watershed Mask Full Image")
                st.image(sel_res["rgb_ws"], use_container_width=True)
            with img_c2:
                st.caption(f"Annotated Full Image ({sel_res['unit']})")
                full_plot_slot = st.empty()
                full_mask_alpha = st.slider("Mask Opacity", 0.0, 1.0, 0.4, 0.05,
                                            key="full_mask_alpha",
                                            help="Adjust the transparency of the fitted shape overlays to see how well they fit.")
                fig_full, ax_full = plt.subplots(figsize=(8, 8), dpi=150)
                plot_annotated_image(ax_full, sel_res["data"], sel_res["draw_shapes"],
                                     sel_res["nm_per_px"], sel_res["unit"], show_mag=True, mask_alpha=full_mask_alpha)
                fig_full.tight_layout(pad=0)
                full_plot_slot.pyplot(fig_full)
                plt.close(fig_full)

            st.markdown("---")
            st.markdown("### Histograms & Effective Radius ($r_{eff}$)")
            render_dimension_guide(shape_type)

            prefix = _common_prefix([r["name"] for r in results])
            unit_full = results[0]["unit"] if results else "nm"

            if shape_type == "Sphere":
                all_d = np.concatenate([r["diameters"] for r in results])
                if all_d.size > 0:
                    fmin, fmax = get_min_max_ui("sph_d", f"Diameter ({unit_full})", all_d)
                    d_crop = all_d[(all_d >= fmin) & (all_d <= fmax)]
                    fig, mu, std, n = histogram_with_fit(all_d, "Diameter", unit_full, fit_min=fmin, fit_max=fmax)
                    st.pyplot(fig, use_container_width=True)
                    _download_row(fig, pd.DataFrame({"diameter": d_crop}), f"{prefix}_diameter")

                    # Calculate r_eff from the cropped window
                    if d_crop.size > 0:
                        if d_crop.size < MIN_SHAPE_COUNT:
                            st.warning(
                                f"Only {d_crop.size} sphere(s) in the selected range (< {MIN_SHAPE_COUNT}); "
                                "r_eff is reported directly as the mean radius but may be unreliable."
                            )
                        mean_d, sd_d = _mean_sd(d_crop)
                        r_eff, r_eff_sd = reff_with_sd(lambda x: (np.pi / 6.0) * x[0] ** 3, [(mean_d, sd_d)])
                        # Sphere surface area S = π·D².
                        area, area_sd = measure_with_sd(lambda x: np.pi * x[0] ** 2, [(mean_d, sd_d)])
                        st.metric(
                            label="Effective Radius (r_eff)",
                            value=f"{r_eff:.2f} ± {r_eff_sd:.2f} {unit_full}",
                            help=f"r_eff = mean diameter / 2, n = {d_crop.size} in selected range. ± is 1 SD.",
                        )
                        summary_export_ui(
                            shape_type,
                            [{"values": all_d, "title": "Diameter", "fit_min": fmin, "fit_max": fmax,
                              "keys": ["diameters"]}],
                            {"R": mean_d / 2.0}, r_eff, r_eff_sd, unit_full, prefix, results=results,
                            area=area, area_sd=area_sd,
                        )
                    else:
                        st.warning("No particles within the selected range.")
                else:
                    st.info("No spherical particles detected.")

            elif shape_type == "Hexagonal Prism":
                all_w = np.concatenate([r["hex_widths"] for r in results])        # face-on hexagons
                all_h = np.concatenate([r["hex_heights"] for r in results])       # side-on rectangle major (height)
                all_rw = np.concatenate([r.get("hex_rect_widths", np.array([])) for r in results])  # side-on rectangle minor (≈ width)

                n_hex, n_rect = all_w.size, all_h.size
                st.caption(f"Detected {n_hex} face-on hexagon(s) and {n_rect} side-on rectangle(s).")

                # Auto-enable the rectangle-GMM width estimate when face-on hexagons
                # are scarce relative to rectangles; let the user override either way.
                auto_gmm = n_rect > 0 and n_hex < HEX_GMM_AUTO_RATIO * n_rect
                use_rect_gmm = st.checkbox(
                    "Estimate hexagonal width from rectangle GMM",
                    value=auto_gmm,
                    help=(
                        "When few particles are face-on (hexagons ≪ rectangles), the "
                        "face-on width sample is unreliable. Instead, pool each "
                        "rectangle's two axes and fit a 2-component GMM: the smaller "
                        "peak is taken as the hexagonal width, the larger as the prism "
                        "height. Auto-enabled when hexagons < "
                        f"{HEX_GMM_AUTO_RATIO:.0%} of rectangles."
                    ),
                )

                # W is the vertex-to-vertex hexagon width (as the notes below state and as the
                # side-on rectangle short axis reports), so the cross-section area is
                # (3√3/8)·W², the same vertex-to-vertex convention used by the Tic Tac model.
                hex_vol = lambda x: (3.0 * np.sqrt(3.0) / 8.0) * (x[0] ** 2) * x[1]   # V = (3√3/8)·W²·H
                # Surface area: two hexagonal faces + six rectangular sides. For a regular hexagon
                # of vertex-to-vertex width W the face area is (3√3/8)·W² and the perimeter is 3W,
                # so S = 2·(3√3/8)·W² + 3·W·H = (3√3/4)·W² + 3·W·H.
                hex_area = lambda x: (3.0 * np.sqrt(3.0) / 4.0) * (x[0] ** 2) + 3.0 * x[0] * x[1]

                if use_rect_gmm:
                    # Pool both rectangle axes (height = major, width = minor) so the
                    # GMM resolves the two characteristic dimensions globally.
                    all_axes = np.concatenate([all_h, all_rw])
                    if all_axes.size > 0:
                        amin, amax = get_min_max_ui("hex_gmm", f"Rectangle axis ({unit_full})", all_axes)
                        axes_crop = all_axes[(all_axes >= amin) & (all_axes <= amax)]
                        mu_ranges = get_mu_ranges_ui("hex_gmm", "rectangle-axis", axes_crop, unit_full, n_components=2)
                        fig_g, mus, stds, n = histogram_with_fit(
                            all_axes, "Rectangle axes", unit_full,
                            n_components=2, fit_min=amin, fit_max=amax, mu_ranges=mu_ranges,
                        )
                        st.pyplot(fig_g, use_container_width=True)
                        _download_row(fig_g, pd.DataFrame({"rectangle_axes": axes_crop}), f"{prefix}_hex_rect_gmm")

                        notes = []
                        if n_hex < MIN_SHAPE_COUNT:
                            notes.append(
                                f"Only {n_hex} face-on hexagon(s) found (< {MIN_SHAPE_COUNT}). The hexagonal "
                                "width is estimated from the side-on rectangle short axis (GMM smaller peak), "
                                "assuming it equals the hexagon vertex-to-vertex width."
                            )
                        # Smaller peak = hexagonal width, larger peak = prism height.
                        if len(mus) >= 2:
                            mean_w, mean_h = mus[0], mus[1]
                            r_eff, r_eff_sd = reff_with_sd(hex_vol, [(mean_w, stds[0]), (mean_h, stds[1])])
                            area, area_sd = measure_with_sd(hex_area, [(mean_w, stds[0]), (mean_h, stds[1])])
                            for msg in notes:
                                st.warning(msg)
                            st.metric(
                                label="Effective Radius (r_eff)",
                                value=f"{r_eff:.2f} ± {r_eff_sd:.2f} {unit_full}",
                                help=(
                                    f"From rectangle GMM: width (smaller μ) = {mean_w:.2f}, height (larger μ) "
                                    f"= {mean_h:.2f} {unit_full}. V = (3√3/8)·W²·H (vertex-to-vertex W); ± is 1 SD propagated from the GMM peak widths."
                                ),
                            )
                            summary_export_ui(
                                shape_type,
                                [{"values": all_axes, "title": "Rectangle axes",
                                  "n_components": 2, "fit_min": amin, "fit_max": amax, "mu_ranges": mu_ranges,
                                  "keys": ["hex_heights", "hex_rect_widths"]}],
                                {"W": mean_w, "H": mean_h}, r_eff, r_eff_sd, unit_full, prefix, " ".join(notes),
                                results=results, area=area, area_sd=area_sd,
                            )
                        else:
                            st.warning("Not enough rectangle data to fit two distinct peaks for r_eff.")
                    else:
                        st.info("No side-on rectangular projections detected.")
                else:
                    hist_specs, w_crop, h_crop = [], np.array([]), np.array([])
                    if all_w.size > 0:
                        wmin, wmax = get_min_max_ui("hex_w", f"Width ({unit_full})", all_w)
                        w_crop = all_w[(all_w >= wmin) & (all_w <= wmax)]
                        fig_w, *_ = histogram_with_fit(all_w, "Hex width (face-on)", unit_full, fit_min=wmin, fit_max=wmax)
                        st.pyplot(fig_w, use_container_width=True)
                        _download_row(fig_w, pd.DataFrame({"hex_width": w_crop}), f"{prefix}_hex_width")
                        hist_specs.append({"values": all_w, "title": "Hex width (face-on)", "fit_min": wmin, "fit_max": wmax,
                                           "keys": ["hex_widths"]})
                    if all_h.size > 0:
                        hmin, hmax = get_min_max_ui("hex_h", f"Height ({unit_full})", all_h)
                        h_crop = all_h[(all_h >= hmin) & (all_h <= hmax)]
                        fig_h, *_ = histogram_with_fit(all_h, "Height (side-on)", unit_full, fit_min=hmin, fit_max=hmax)
                        st.pyplot(fig_h, use_container_width=True)
                        _download_row(fig_h, pd.DataFrame({"hex_height": h_crop}), f"{prefix}_hex_height")
                        hist_specs.append({"values": all_h, "title": "Height (side-on)", "fit_min": hmin, "fit_max": hmax,
                                           "keys": ["hex_heights"]})

                    # Resolve width & height, stating the assumption whenever a projection is scarce.
                    notes = []
                    if w_crop.size > 0:
                        mean_w, sd_w = _mean_sd(w_crop)
                        if w_crop.size < MIN_SHAPE_COUNT:
                            notes.append(f"Only {w_crop.size} face-on hexagon(s) (< {MIN_SHAPE_COUNT}); the width sample is small and r_eff may be unreliable.")
                    elif all_rw.size > 0:
                        mean_w, sd_w = _mean_sd(all_rw)
                        notes.append(f"No face-on hexagons found; width taken from the {all_rw.size} side-on rectangle short axis/axes, assuming it equals the hexagon vertex-to-vertex width.")
                    else:
                        mean_w, sd_w = float("nan"), float("nan")

                    if h_crop.size > 0:
                        mean_h, sd_h = _mean_sd(h_crop)
                        if h_crop.size < MIN_SHAPE_COUNT:
                            notes.append(f"Only {h_crop.size} side-on rectangle(s) (< {MIN_SHAPE_COUNT}); the height sample is small and r_eff may be unreliable.")
                    elif np.isfinite(mean_w):
                        mean_h, sd_h = mean_w, sd_w
                        notes.append("No side-on rectangles found; prism height is assumed equal to the width (aspect ratio 1).")
                    else:
                        mean_h, sd_h = float("nan"), float("nan")

                    for msg in notes:
                        st.warning(msg)

                    if np.isfinite(mean_w) and np.isfinite(mean_h):
                        r_eff, r_eff_sd = reff_with_sd(hex_vol, [(mean_w, sd_w), (mean_h, sd_h)])
                        area, area_sd = measure_with_sd(hex_area, [(mean_w, sd_w), (mean_h, sd_h)])
                        st.metric(
                            label="Effective Radius (r_eff)",
                            value=f"{r_eff:.2f} ± {r_eff_sd:.2f} {unit_full}",
                            help=f"Mean width = {mean_w:.2f}, mean height = {mean_h:.2f} {unit_full}. V = (3√3/8)·W²·H (vertex-to-vertex W); ± is 1 SD.",
                        )
                        summary_export_ui(
                            shape_type, hist_specs, {"W": mean_w, "H": mean_h},
                            r_eff, r_eff_sd, unit_full, prefix, " ".join(notes), results=results,
                            area=area, area_sd=area_sd,
                        )
                    else:
                        st.warning("Both face-on (width) and side-on (height) measurements are needed to calculate r_eff for hexagonal prisms.")

            elif shape_type == "Cube":
                all_s = np.concatenate([r["side_lengths"] for r in results])
                if all_s.size > 0:
                    smin, smax = get_min_max_ui("cube_s", f"Side length ({unit_full})", all_s)
                    s_crop = all_s[(all_s >= smin) & (all_s <= smax)]
                    fig, mu, std, n = histogram_with_fit(all_s, "Cube side length", unit_full, fit_min=smin, fit_max=smax)
                    st.pyplot(fig, use_container_width=True)
                    _download_row(fig, pd.DataFrame({"side_length": s_crop}), f"{prefix}_side_length")

                    # Calculate r_eff from the cropped window
                    if s_crop.size > 0:
                        if s_crop.size < MIN_SHAPE_COUNT:
                            st.warning(
                                f"Only {s_crop.size} cube(s) in the selected range (< {MIN_SHAPE_COUNT}); "
                                "r_eff assumes a regular cube (V = s³) but may be unreliable."
                            )
                        mean_s, sd_s = _mean_sd(s_crop)
                        r_eff, r_eff_sd = reff_with_sd(lambda x: x[0] ** 3, [(mean_s, sd_s)])
                        # Cube surface area S = 6·s².
                        area, area_sd = measure_with_sd(lambda x: 6.0 * x[0] ** 2, [(mean_s, sd_s)])
                        st.metric(
                            label="Effective Radius (r_eff)",
                            value=f"{r_eff:.2f} ± {r_eff_sd:.2f} {unit_full}",
                            help=f"V = s³ with mean side = {mean_s:.2f} {unit_full}, n = {s_crop.size}. ± is 1 SD.",
                        )
                        summary_export_ui(
                            shape_type,
                            [{"values": all_s, "title": "Cube side length", "fit_min": smin, "fit_max": smax,
                              "keys": ["side_lengths"]}],
                            {"s": mean_s}, r_eff, r_eff_sd, unit_full, prefix, results=results,
                            area=area, area_sd=area_sd,
                        )
                    else:
                        st.warning("No particles within the selected range.")
                else:
                    st.info("No cubic particles detected.")

            elif shape_type == "Octahedron":
                all_maj, all_min = np.concatenate([r["oct_major"] for r in results]), np.concatenate([r["oct_minor"] for r in results])
                if all_maj.size > 0 or all_min.size > 0:
                    all_axes = np.concatenate([all_maj, all_min])
                    omin, omax = get_min_max_ui("oct", f"Axis length ({unit_full})", all_axes)
                    axes_crop = all_axes[(all_axes >= omin) & (all_axes <= omax)]

                    n_comp = 2 if axes_crop.size >= 6 else 1
                    mu_ranges = get_mu_ranges_ui("oct", "axis", axes_crop, unit_full, n_components=n_comp)
                    fig, mus, stds, n = histogram_with_fit(all_axes, "Octahedron combined axes", unit_full,
                                                           n_components=n_comp, fit_min=omin, fit_max=omax, mu_ranges=mu_ranges)
                    st.pyplot(fig, use_container_width=True)
                    _download_row(fig, pd.DataFrame({"combined_axes": axes_crop}), f"{prefix}_octahedron")

                    # Axes are vertex-to-vertex (tip-to-tip) projected distances, not edge
                    # lengths, so a regular octahedron of diameter D has V = D³/6. The general
                    # (elongated) form with distinct major/minor axes is V = (1/6)·major·minor².
                    oct_vol = lambda x: (1.0 / 6.0) * x[1] * (x[0] ** 2)   # V = (1/6)·major·minor²
                    # Surface area of the square bipyramid (8 congruent triangles): with an
                    # equatorial square of diagonal = minor (x[0]) and tip-to-tip major = x[1],
                    # each face has base minor/√2 and slant height √(major²/4 + minor²/8), giving
                    # S = 2√2·minor·√(major²/4 + minor²/8) (→ √3·D² for a regular octahedron).
                    oct_area = lambda x: 2.0 * np.sqrt(2.0) * x[0] * np.sqrt((x[1] ** 2) / 4.0 + (x[0] ** 2) / 8.0)
                    notes = []
                    # Calculate r_eff using the two dominant GMM peaks; fall back to a
                    # regular octahedron (major = minor) when the sample is too small to
                    # resolve two peaks.
                    if len(mus) >= 2:
                        mu_minor, mu_major = mus[0], mus[1]  # Pre-sorted smallest to largest
                        dims = [(mu_minor, stds[0]), (mu_major, stds[1])]
                        help_txt = f"GMM peaks: minor = {mu_minor:.2f}, major = {mu_major:.2f} {unit_full}."
                    else:
                        mu_minor = mu_major = float(mus[0]) if mus else float(np.mean(axes_crop))
                        sd_axes = float(stds[0]) if stds else float(np.std(axes_crop))
                        dims = [(mu_minor, sd_axes), (mu_major, sd_axes)]
                        notes.append(
                            f"Only {axes_crop.size} axis measurement(s) (< {MIN_SHAPE_COUNT}); could not resolve two "
                            "distinct peaks, so a regular octahedron (major = minor) is assumed."
                        )
                        help_txt = f"Regular-octahedron assumption: major = minor = {mu_minor:.2f} {unit_full}."
                    r_eff, r_eff_sd = reff_with_sd(oct_vol, dims)
                    area, area_sd = measure_with_sd(oct_area, dims)
                    for msg in notes:
                        st.warning(msg)
                    st.metric(
                        label="Effective Radius (r_eff)",
                        value=f"{r_eff:.2f} ± {r_eff_sd:.2f} {unit_full}",
                        help=f"{help_txt} V = (1/6)·major·minor² (vertex-to-vertex axes); ± is 1 SD.",
                    )
                    summary_export_ui(
                        shape_type,
                        [{"values": all_axes, "title": "Octahedron combined axes", "n_components": n_comp,
                          "fit_min": omin, "fit_max": omax, "mu_ranges": mu_ranges,
                          "keys": ["oct_major", "oct_minor"]}],
                        {"major": mu_major, "minor": mu_minor}, r_eff, r_eff_sd, unit_full, prefix, " ".join(notes),
                        results=results, area=area, area_sd=area_sd,
                    )
                else:
                    st.info("No octahedral particles detected.")
                
            elif shape_type == "Tic Tac":
                # Stadium model: rectangle body + two circular-arc end caps.
                # Volume = hexagonal-prism body + two half-ellipsoid caps:
                #   V = (3√3/8)·W·T·L_body + (π/3)·W·T·d
                # where W,T are the cross-section dimensions (from the width
                # distribution), L_body the straight body length, d the cap depth.
                all_maj  = np.concatenate([r["tic_tac_major"] for r in results])
                all_w    = np.concatenate([r.get("tic_tac_width", np.array([])) for r in results])
                all_body = np.concatenate([r.get("tic_tac_body_length", np.array([])) for r in results])
                all_cap  = np.concatenate([r.get("tic_tac_cap_depth", np.array([])) for r in results])

                if all_w.size == 0:
                    st.info("No Tic Tac particles detected.")
                else:
                    st.caption(
                        f"Detected {all_w.size} tic-tac particle(s). Volume model: hexagonal-prism "
                        "body with two half-ellipsoid arc caps."
                    )

                    # Width distribution → one or two cross-section dimensions (W, T).
                    # Default the bimodal toggle from 2-component GMM peak separation.
                    bimodal_default = False
                    if all_w.size >= 6:
                        try:
                            gmm_w = GaussianMixture(n_components=2, random_state=42).fit(all_w.reshape(-1, 1))
                            m = np.sort(gmm_w.means_.flatten())
                            s = float(np.mean(np.sqrt(gmm_w.covariances_.flatten())))
                            bimodal_default = (m[1] - m[0]) > 1.5 * s
                        except Exception:
                            pass
                    bimodal = st.checkbox(
                        "Width has two distinct cross-section dimensions (W ≠ T)",
                        value=bimodal_default,
                        help=(
                            "When the width distribution is bimodal, a 2-component GMM is fit: the "
                            "smaller peak is taken as the hexagonal width W, the larger as the "
                            "perpendicular dimension T. When unchecked, the cross-section is assumed "
                            "square (T = W). Auto-checked when the two peaks are well separated."
                        ),
                    )

                    # Width histogram (2-component GMM when bimodal).
                    wmin, wmax = get_min_max_ui("tic_w", f"Width ({unit_full})", all_w)
                    w_crop = all_w[(all_w >= wmin) & (all_w <= wmax)]
                    w_ncomp = 2 if bimodal else 1
                    mu_ranges = get_mu_ranges_ui("tic_w", "width", w_crop, unit_full, n_components=w_ncomp)
                    fig_w, mus_w, stds_w, n_w = histogram_with_fit(
                        all_w, "Tic Tac width", unit_full,
                        n_components=w_ncomp, fit_min=wmin, fit_max=wmax, mu_ranges=mu_ranges,
                    )
                    st.pyplot(fig_w, use_container_width=True)
                    _download_row(fig_w, pd.DataFrame({"tic_tac_width": w_crop}), f"{prefix}_tictac_width")

                    notes = []
                    mean_w_crop, sd_w_crop = _mean_sd(w_crop)
                    if bimodal and len(mus_w) >= 2:
                        W_dim, sd_W = float(mus_w[0]), float(stds_w[0])   # smaller mode
                        T_dim, sd_T = float(mus_w[1]), float(stds_w[1])   # larger mode
                    else:
                        W_dim = T_dim = mean_w_crop
                        sd_W = sd_T = sd_w_crop
                        notes.append("Width distribution treated as unimodal; the cross-section is assumed square (T = W).")
                    if w_crop.size < MIN_SHAPE_COUNT:
                        notes.append(f"Only {w_crop.size} tic-tac(s) in the selected width range (< {MIN_SHAPE_COUNT}); r_eff may be unreliable.")

                    hist_specs = [{"values": all_w, "title": "Tic Tac width", "n_components": w_ncomp,
                                   "fit_min": wmin, "fit_max": wmax, "mu_ranges": mu_ranges,
                                   "keys": ["tic_tac_width"]}]

                    # Body length (straight plateau between the caps).
                    mean_body, sd_body = float("nan"), float("nan")
                    if all_body.size > 0:
                        blmin, blmax = get_min_max_ui("tic_body", f"Body length ({unit_full})", all_body)
                        body_crop = all_body[(all_body >= blmin) & (all_body <= blmax)]
                        fig_b, _, _, _ = histogram_with_fit(
                            all_body, "Tic Tac body length", unit_full, fit_min=blmin, fit_max=blmax,
                        )
                        st.pyplot(fig_b, use_container_width=True)
                        _download_row(fig_b, pd.DataFrame({"tic_tac_body_length": body_crop}), f"{prefix}_tictac_body")
                        if body_crop.size > 0:
                            mean_body, sd_body = _mean_sd(body_crop)
                            hist_specs.append({"values": all_body, "title": "Tic Tac body length",
                                               "fit_min": blmin, "fit_max": blmax,
                                               "keys": ["tic_tac_body_length"]})

                    # Total length (body + both caps) for reference.
                    if all_maj.size > 0:
                        mjmin, mjmax = get_min_max_ui("tic_maj", f"Total length ({unit_full})", all_maj)
                        maj_crop = all_maj[(all_maj >= mjmin) & (all_maj <= mjmax)]
                        fig_maj, _, _, _ = histogram_with_fit(
                            all_maj, "Tic Tac total length", unit_full, fit_min=mjmin, fit_max=mjmax,
                        )
                        st.pyplot(fig_maj, use_container_width=True)
                        _download_row(fig_maj, pd.DataFrame({"tic_tac_total_length": maj_crop}), f"{prefix}_tictac_total")
                        hist_specs.append({"values": all_maj, "title": "Tic Tac total length",
                                           "fit_min": mjmin, "fit_max": mjmax,
                                           "keys": ["tic_tac_major"]})

                    mean_cap, sd_cap = _mean_sd(all_cap) if all_cap.size > 0 else (0.0, 0.0)

                    # r_eff from the stadium volume: (3√3/8)·W·T·L_body + (π/3)·W·T·d.
                    def tictac_vol(x):
                        return (3.0 * np.sqrt(3.0) / 8.0) * x[0] * x[1] * x[2] + (np.pi / 3.0) * x[0] * x[1] * x[3]

                    # Surface area = the (possibly anisotropic) hexagonal body lateral surface,
                    # perimeter(W,T)·L_body, plus the two half-ellipsoid caps, whose outer surface
                    # together equals one full ellipsoid of semi-axes (W/2, T/2, d). No flat end
                    # faces (the caps replace them).
                    def tictac_area(x):
                        return (_hexagon_perimeter(x[0] / 2.0, x[1] / 2.0) * x[2]
                                + _ellipsoid_surface_area(x[0] / 2.0, x[1] / 2.0, x[3]))

                    for msg in notes:
                        st.warning(msg)

                    if np.isfinite(W_dim) and np.isfinite(T_dim) and np.isfinite(mean_body):
                        dims_tt = [(W_dim, sd_W), (T_dim, sd_T), (mean_body, sd_body), (mean_cap, sd_cap)]
                        r_eff, r_eff_sd = reff_with_sd(tictac_vol, dims_tt)
                        area, area_sd = measure_with_sd(tictac_area, dims_tt)
                        v_body = (3.0 * np.sqrt(3.0) / 8.0) * W_dim * T_dim * mean_body
                        v_caps = (np.pi / 3.0) * W_dim * T_dim * mean_cap
                        st.metric(
                            label="Effective Radius (r_eff)",
                            value=f"{r_eff:.2f} ± {r_eff_sd:.2f} {unit_full}",
                            help=(
                                f"Hex-prism body (W={W_dim:.2f}, T={T_dim:.2f}, L_body={mean_body:.2f}) "
                                f"+ half-ellipsoid caps (depth d={mean_cap:.2f}). "
                                f"V_body={v_body:.1f}, V_caps={v_caps:.1f} {unit_full}³. ± is 1 SD."
                            ),
                        )
                        summary_export_ui(
                            shape_type, hist_specs,
                            {"W": W_dim, "T": T_dim, "L_body": mean_body, "d": mean_cap},
                            r_eff, r_eff_sd, unit_full, prefix, " ".join(notes), results=results,
                            area=area, area_sd=area_sd,
                        )
                    else:
                        st.warning("Both width and body-length measurements are needed to compute r_eff for Tic Tacs.")
if __name__ == "__main__":
    run()