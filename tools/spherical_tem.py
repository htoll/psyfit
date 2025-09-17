"""Interactive TEM particle analysis tool.

This Streamlit app reads `.dm3` files and measures particle sizes.  It now
supports spherical, hexagonal and cubic nanoparticles and provides quick
visual checks of the segmentation quality.

Main features added compared to the previous version:

* Overlay of detected shapes on the original grayscale image.
* A second tab displays the raw watershed labels used for segmentation.
* A dropdown lets the user choose which uploaded image to view.
* Shape type dropdown (spherical, hexagonal or cubic/rectangular).
* Size distribution histograms with Gaussian fits and reported volumes.

The goal of the implementation is not a perfect particle classifier but to
translate the MATLAB prototype provided by the user into a working Python
Streamlit workflow.
"""

from __future__ import annotations

import io
import math
import os
import tempfile
import datetime as dt
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import ndimage as ndi

from skimage import morphology as morph
from skimage import segmentation as seg
from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture
from scipy import ndimage as ndi
from skimage import morphology as morph
from skimage import segmentation as seg
from skimage.measure import label


SESSION_CACHE_KEY = "_tem_size_cache"
BASE_CACHE_KEY = "_tem_gmm_base_cache"

PLOTLY_SHAPE_CONFIG = {"displaylogo": False}

# Optional import of ncempy (for reading dm3 files)
try:  # pragma: no cover - simply a convenience check
    from ncempy.io import dm as ncem_dm
except Exception:  # pragma: no cover
    ncem_dm = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DM3Image:
    data: np.ndarray
    nm_per_px: float


@dataclass
class PreprocessedImage:
    name: str
    data: np.ndarray
    threshold: float
    nm_per_px: float
    measurement_unit: str


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


UNIT_TO_NM: Dict[str, float] = {
    "m": 1e9,
    "meter": 1e9,
    "metre": 1e9,
    "mm": 1e6,
    "millimeter": 1e6,
    "millimetre": 1e6,
    "um": 1e3,
    "micrometer": 1e3,
    "micrometre": 1e3,
    "micron": 1e3,
    "µm": 1e3,
    "pm": 1e-3,
    "picometer": 1e-3,
    "picometre": 1e-3,
    "nm": 1.0,
    "nanometer": 1.0,
    "nanometre": 1.0,
    "angstrom": 0.1,
    "angstroem": 0.1,
    "ang": 0.1,
    "a": 0.1,
}


def _normalize_unit(unit: Any) -> Optional[str]:
    """Normalize unit strings for lookup in :data:`UNIT_TO_NM`."""

    if unit is None:
        return None
    if isinstance(unit, bytes):
        try:
            unit = unit.decode("utf-8")
        except Exception:  # pragma: no cover - best effort decoding
            unit = unit.decode("latin1", errors="ignore")
    unit_str = str(unit).strip().lower()
    if not unit_str:
        return None
    replacements = (
        ("µ", "u"),
        ("μ", "u"),
        ("ångström", "angstrom"),
        ("angström", "angstrom"),
        ("meters", "meter"),
        ("metres", "metre"),
        ("nanometers", "nanometer"),
        ("nanometres", "nanometre"),
        ("micrometers", "micrometer"),
        ("micrometres", "micrometre"),
        ("millimeters", "millimeter"),
        ("millimetres", "millimetre"),
        ("microns", "micron"),
    )
    for old, new in replacements:
        unit_str = unit_str.replace(old, new)
    for suffix in ("/pixel", " per pixel", "per pixel", "/px", " perpix", "perpix"):
        if suffix in unit_str:
            unit_str = unit_str.replace(suffix, "")
    unit_str = unit_str.replace("pixel", "")
    unit_str = unit_str.replace(" per", "")
    unit_str = unit_str.strip()
    if unit_str.endswith("s") and unit_str not in {"ms"}:
        unit_str = unit_str[:-1]
    return unit_str or None


def _factor_from_unit(unit: Any) -> Optional[float]:
    cleaned = _normalize_unit(unit)
    if not cleaned or "/" in cleaned:
        return None
    return UNIT_TO_NM.get(cleaned)


def _convert_scale_value(value: Any, unit: Any = None, default_factor: Optional[float] = None) -> Optional[float]:
    """Convert calibration values to nm/pixel if possible."""

    if isinstance(value, dict):
        unit_candidates = [
            value.get("unit"),
            value.get("units"),
            value.get("Unit"),
            value.get("Units"),
        ]
        unit_pref = next((u for u in unit_candidates if u is not None), unit)
        value_candidates = [
            value.get("value"),
            value.get("Value"),
            value.get("scale"),
            value.get("Scale"),
        ]
        for candidate in value_candidates:
            nm = _convert_scale_value(candidate, unit=unit_pref, default_factor=default_factor)
            if nm is not None:
                return nm
        for sub_value in value.values():
            nm = _convert_scale_value(sub_value, unit=unit, default_factor=default_factor)
            if nm is not None:
                return nm
        return None

    if hasattr(value, "value"):
        attr_unit = getattr(value, "unit", None) or getattr(value, "units", None)
        nm = _convert_scale_value(value.value, unit=attr_unit or unit, default_factor=default_factor)
        if nm is not None:
            return nm

    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.array(value, dtype=object).ravel()
        for item in arr:
            nm = _convert_scale_value(item, unit=unit, default_factor=default_factor)
            if nm is not None:
                return nm
        return None

    try:
        val_float = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(val_float) or val_float <= 0:
        return None

    factor = _factor_from_unit(unit)
    if factor is None:
        factor = default_factor
    if factor is None:
        return None
    return val_float * factor


def _flatten_to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out: List[Any] = []
        for item in value:
            out.extend(_flatten_to_list(item))
        return out
    return [value]


def _extract_from_pixel_info(pixel_size: Any, pixel_unit: Any) -> Optional[float]:
    sizes = _flatten_to_list(pixel_size)
    if not sizes:
        return None
    units = _flatten_to_list(pixel_unit)
    for idx, size in enumerate(sizes):
        unit = units[idx] if idx < len(units) else (units[-1] if units else None)
        nm = _convert_scale_value(size, unit=unit)
        if nm is not None:
            return nm
    return None


def _get_nested(metadata: Dict[str, Any], path: str) -> Any:
    current: Any = metadata
    for part in path.split("."):
        if isinstance(current, dict):
            if part in current:
                current = current[part]
                continue
            if part.isdigit():
                idx = int(part)
                if idx in current:
                    current = current[idx]
                    continue
            return None
        elif isinstance(current, (list, tuple)):
            if not part.isdigit():
                return None
            idx = int(part)
            if idx >= len(current):
                return None
            current = current[idx]
        else:
            return None
    return current


def _search_scale_in_metadata(metadata: Any) -> Optional[float]:
    tokens = ("pixel", "scale", "calibration", "dimension", "step", "size")

    if isinstance(metadata, dict):
        for key, value in metadata.items():
            key_lower = str(key).lower()
            default_factor = 1e9 if any(tok in key_lower for tok in ("scale", "calibration", "dimension")) else None
            if any(tok in key_lower for tok in tokens):
                nm = _convert_scale_value(value, default_factor=default_factor)
                if nm is not None:
                    return nm
            nm = _search_scale_in_metadata(value)
            if nm is not None:
                return nm
    elif isinstance(metadata, (list, tuple)):
        for item in metadata:
            nm = _search_scale_in_metadata(item)
            if nm is not None:
                return nm
    return None


# ---------------------------------------------------------------------------
# File handling utilities
# ---------------------------------------------------------------------------


def try_read_dm3(file_bytes: bytes) -> DM3Image:
    """Read a dm3 file into a :class:`DM3Image` instance."""

    if ncem_dm is None:  # pragma: no cover - runtime check in UI
        raise RuntimeError("ncempy is not installed. Please install it (pip install ncempy).")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dm3")
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()

        with ncem_dm.fileDM(tmp.name, verbose=False) as rdr:
            im = rdr.getDataset(0)
            data = np.array(im["data"], dtype=np.float32)

            nm_per_px = float("nan")
            md = rdr.allTags

            nm_from_dataset = _extract_from_pixel_info(im.get("pixelSize"), im.get("pixelUnit"))
            if nm_from_dataset is not None:
                nm_per_px = nm_from_dataset

            if not np.isfinite(nm_per_px):
                candidates = [
                    "ImageList.1.ImageData.Calibrations.Dimension.0.Scale",
                    "ImageList.1.ImageData.Calibrations.Dimension.1.Scale",
                    "ImageList.0.ImageData.Calibrations.Dimension.0.Scale",
                    "ImageList.0.ImageData.Calibrations.Dimension.1.Scale",
                    "ImageList.2.ImageData.Calibrations.Dimension.0.Scale",
                    "ImageList.2.ImageData.Calibrations.Dimension.1.Scale",
                    "pixelSize.x",
                    "pixelSize",
                    "xscale",
                ]
                for key in candidates:
                    raw_val = _get_nested(md, key)
                    if raw_val is None:
                        continue
                    nm_candidate = _convert_scale_value(raw_val, default_factor=1e9)
                    if nm_candidate is not None:
                        nm_per_px = nm_candidate
                        break

            if not np.isfinite(nm_per_px):
                nm_meta = _search_scale_in_metadata(md)
                if nm_meta is not None:
                    nm_per_px = nm_meta

        return DM3Image(data=data, nm_per_px=nm_per_px)
    finally:  # pragma: no cover - best effort cleanup
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------


def robust_percentile_cut(data: np.ndarray, p: float = 99.5) -> np.ndarray:
    flat = data.reshape(-1)
    cutoff = np.percentile(flat, p)
    return flat[flat <= cutoff]


def histogram_for_intensity(data: np.ndarray, nbins: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    vals = robust_percentile_cut(data, 99.5)
    if nbins is None:
        nbins = max(10, int(round(math.sqrt(len(vals)) / 2)))
    counts, edges = np.histogram(vals, bins=nbins)
    centers = edges[:-1] + np.diff(edges) / 2
    return centers, counts


def _gmm_intersection_point(
    means: np.ndarray, covariances: np.ndarray, weights: np.ndarray
) -> Optional[float]:
    """Return the intensity where two 1-D Gaussians intersect."""

    if means.size != 2 or weights.size != 2:
        return None

    order = np.argsort(means)
    mu1, mu2 = float(means[order[0]]), float(means[order[1]])
    w1, w2 = float(weights[order[0]]), float(weights[order[1]])

    covs = np.asarray(covariances)

    def _variance(idx: int) -> float:
        cov = covs[order[idx]]
        arr = np.asarray(cov, dtype=float)
        if arr.size == 1:
            return float(arr.reshape(()))
        return float(arr.flat[0])

    var1 = max(_variance(0), 1e-12)
    var2 = max(_variance(1), 1e-12)
    std1 = math.sqrt(var1)
    std2 = math.sqrt(var2)

    log_ratio = math.log((w2 / std2) / (w1 / std1))
    a = (1.0 / (2.0 * var2)) - (1.0 / (2.0 * var1))
    b = (mu1 / var1) - (mu2 / var2)
    c = (mu2 ** 2) / (2.0 * var2) - (mu1 ** 2) / (2.0 * var1) - log_ratio

    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return float((mu1 + mu2) / 2.0)
        root = -c / b
        return float(root)

    disc = b ** 2 - 4 * a * c
    if disc < 0:
        disc = 0.0
    sqrt_disc = math.sqrt(disc)
    roots = [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]

    lower, upper = sorted((mu1, mu2))
    candidates = [r for r in roots if lower - 5 * std1 <= r <= upper + 5 * std2]
    between = [r for r in candidates if lower <= r <= upper]
    if between:
        return float(between[0])
    if candidates:
        return float(min(candidates, key=lambda r: abs(r - (mu1 + mu2) / 2.0)))
    return float((mu1 + mu2) / 2.0)


def gmm_threshold(data: np.ndarray, nbins: Optional[int] = None, sample: int = 200_000) -> float:
    flat = data.reshape(-1)
    if len(flat) > sample:
        idx = np.random.choice(len(flat), sample, replace=False)
        flat = flat[idx]
    gm = GaussianMixture(n_components=2, random_state=42)
    gm.fit(flat.reshape(-1, 1))
    threshold = _gmm_intersection_point(
        gm.means_.flatten(), gm.covariances_, gm.weights_.flatten()
    )

    if threshold is not None and np.isfinite(threshold):
        return float(threshold)

    mu = np.sort(gm.means_.flatten())
    left_mu, right_mu = mu[0], mu[1]
    centers, counts = histogram_for_intensity(flat, nbins)
    in_range = (centers >= left_mu) & (centers <= right_mu)
    if not np.any(in_range):
        return float((left_mu + right_mu) / 2)
    sub_centers = centers[in_range]
    sub_counts = counts[in_range]
    return float(sub_centers[np.argmin(sub_counts)])


# ---------------------------------------------------------------------------
# Segmentation and measurement
# ---------------------------------------------------------------------------


def _approximate_region_vertices(region: Any) -> Tuple[np.ndarray, int]:
    """Return approximated polygon vertices for a region."""

    image = np.asarray(getattr(region, "image", None))
    if image is None or image.size == 0 or not np.any(image):
        return np.empty((0, 2), dtype=np.float32), 0

    try:
        contours = find_contours(image.astype(float), 0.5)
    except Exception:
        return np.empty((0, 2), dtype=np.float32), 0

    if not contours:
        return np.empty((0, 2), dtype=np.float32), 0

    contour = max(contours, key=len)
    diag = math.hypot(image.shape[0], image.shape[1])
    tolerance = max(0.5, 0.02 * diag)
    approx = approximate_polygon(contour, tolerance)
    if approx.shape[0] <= 2:
        return np.empty((0, 2), dtype=np.float32), 0

    if np.allclose(approx[0], approx[-1]):
        approx = approx[:-1]

    if approx.size == 0:
        return np.empty((0, 2), dtype=np.float32), 0

    approx = approx.astype(np.float32)
    minr, minc, _, _ = region.bbox
    approx[:, 0] += float(minr)
    approx[:, 1] += float(minc)
    vertices = np.column_stack((approx[:, 1], approx[:, 0])).astype(np.float32)
    return vertices, vertices.shape[0]


def _draw_polygon(ax: Any, vertices: np.ndarray, bbox: Tuple[float, float, float, float], *, color: str) -> None:
    if vertices.shape[0] >= 3:
        ax.add_patch(
            plt.Polygon(vertices, closed=True, fill=False, edgecolor=color, linewidth=1)
        )
    else:
        minc, minr, maxc, maxr = bbox
        ax.add_patch(
            plt.Rectangle(
                (minc, minr),
                maxc - minc,
                maxr - minr,
                fill=False,
                edgecolor=color,
                linewidth=1,
            )
        )


def _draw_debug_circle(ax: Any, cx: float, cy: float, radius: float, color: str) -> None:
    if radius <= 0:
        radius = 1.0
    ax.add_patch(plt.Circle((cx, cy), radius, fill=False, color=color, linewidth=1))


def _is_hexagon(vertex_count: int, aspect: float, solidity: float, circularity: float, extent: float) -> bool:
    if vertex_count >= 5 and vertex_count <= 8:
        if (
            solidity >= 0.85
            and circularity >= 0.7
            and 0.6 <= aspect <= 1.7
            and 0.55 <= extent <= 0.95
        ):
            return True
    if (
        solidity >= 0.9
        and circularity >= 0.82
        and 0.7 <= aspect <= 1.5
        and extent <= 0.95
    ):
        return True
    return False


def _is_rectangle(
    vertex_count: int,
    aspect: float,
    solidity: float,
    circularity: float,
    extent: float,
    *,
    allow_square: bool,
) -> bool:
    if vertex_count in {4, 5} and solidity >= 0.75 and extent >= 0.6:
        if allow_square:
            return aspect >= 0.9
        return aspect >= 1.2
    if (
        aspect >= (1.15 if allow_square else 1.3)
        and solidity >= 0.75
        and circularity <= 0.85
        and extent >= 0.55
    ):
        return True
    return False


def segment_and_measure_shapes(
    data: np.ndarray,
    threshold: float,
    nm_per_px: float,
    shape_type: str,
    min_size_value: float,
    measurement_unit: str,
    min_area_px: int = 5,
) -> Dict[str, np.ndarray]:
    """Segment particles and measure their dimensions.

    Parameters
    ----------
    data: ndarray
        The image data.
    threshold: float
        Threshold value for creating a binary mask.
    nm_per_px: float
        Pixel to nanometre scaling.
    shape_type: str
        'Sphere', 'Hexagon', or 'Cube'.
    min_size_value: float
        Minimum accepted feature size expressed in ``measurement_unit``.
    measurement_unit: str
        Unit label for size measurements (``"nm"`` or ``"px"``).
    min_area_px: int
        Minimum area (in pixels) for removing small objects.

    Returns
    -------
    Dict with measurement arrays and PNG overlays.
    """

    # Binary image and watershed segmentation
    im_bi = data < threshold
    im_bi = morph.binary_closing(im_bi, morph.disk(3))
    im_bi = morph.remove_small_objects(im_bi, min_size=max(min_area_px, 32))

    if np.any(im_bi):
        distance_map = ndi.distance_transform_edt(im_bi)
        distance_smooth = ndi.gaussian_filter(distance_map, sigma=1.0)
        h_val = 2
        markers_bin = morph.h_maxima(distance_smooth, h=h_val)
        markers = label(markers_bin)
        if markers.max() == 0:
            markers = label(im_bi)

        labels_ws = seg.watershed(-distance_smooth, markers=markers, mask=im_bi)
        im_bi_split = im_bi.copy()
        im_bi_split[labels_ws == 0] = False
        im_bi_filtered = morph.remove_small_objects(
            im_bi_split,
            min_size=max(min_area_px, 5000),
        )
        im_bi_filtered = morph.remove_small_holes(
            im_bi_filtered,
            area_threshold=max(4 * min_area_px, 256),
        )
        labels_ws = label(im_bi_filtered)
        im_bi = im_bi_filtered
    smoothed = gaussian(data.astype(np.float32), sigma=0.8, preserve_range=True)
    # im_bi = smoothed < threshold
    # im_bi = binary_closing(im_bi, disk(2))
    # im_bi = binary_opening(im_bi, disk(1))
    # hole_area = max(int(min_area_px * 4), 16)
    # im_bi = remove_small_holes(im_bi, area_threshold=hole_area)
    # im_bi = im_bi.astype(bool)

    im_bi = (data < threshold)
    im_bi = morph.binary_closing(im_bi, morph.disk(3))
    im_bi = morph.remove_small_objects(im_bi, min_size=max(min_area_px, 32))




    

    if np.any(im_bi):
        dist = distance_transform_edt(im_bi)
        smoothed_dist = gaussian(dist, sigma=1.0, preserve_range=True)
        h_val = 2  # try 1–3; increase if over-seg, decrease if under-seg
        markers_bin = morph.h_maxima(smoothed_dist, h=h_val)
        markers = label(markers_bin)
        labels_ws = seg.watershed(-smoothed_dist, markers=markers, mask=im_bi)
        im_bi_split = im_bi.copy()
        im_bi_split[labels_ws == 0] = False
        im_bi_filtered = morph.remove_small_objects(im_bi_split, min_size=max(min_area_px, 5000))
        im_bi_filtered = morph.remove_small_holes(im_bi_filtered, area_threshold=max(4*min_area_px, 256))
        labels_ws = label(im_bi_filtered)


        
        component_labels = label(im_bi)
        local_max = peak_local_max(
            smoothed_dist,
            footprint=np.ones((3, 3), dtype=bool),
            labels=im_bi,
            threshold_abs=0.0,
            exclude_border=False,
        )
        if local_max.size == 0:
            markers = component_labels
        else:
            marker_mask = np.zeros_like(im_bi, dtype=bool)
            marker_mask[tuple(local_max.T)] = True
            markers = label(marker_mask)

            component_count = int(component_labels.max())
            if component_count > 0:
                has_marker = np.zeros(component_count + 1, dtype=bool)
                if np.any(marker_mask):
                    has_marker[component_labels[marker_mask]] = True
                missing_components = np.where(~has_marker)[0]
                missing_components = missing_components[missing_components > 0]
                if missing_components.size:
                    current_max = int(markers.max())
                    for comp_id in missing_components:
                        comp_mask = component_labels == comp_id
                        if not np.any(comp_mask):
                            continue
                        rows, cols = np.where(comp_mask)
                        if rows.size == 0:
                            continue
                        comp_dist = dist[comp_mask]
                        if comp_dist.size == 0:
                            continue
                        max_idx = int(np.argmax(comp_dist))
                        current_max += 1
                        markers[rows[max_idx], cols[max_idx]] = current_max

        refined_mask = labels_ws > 0
        refined_mask = remove_small_objects(refined_mask, min_size=max(min_area_px, 3))
        refined_mask = remove_small_holes(refined_mask, area_threshold=hole_area)
        labels_ws = label(refined_mask)
        im_bi = refined_mask
    else:
        labels_ws = np.zeros_like(data, dtype=np.int32)

    # Prepare measurement containers
    diameters_nm: List[float] = []
    hex_axes_nm: List[float] = []
    lengths_nm: List[float] = []
    widths_nm: List[float] = []

    # Determine scaling and exclusion zone based on available calibration
    if measurement_unit == "nm" and np.isfinite(nm_per_px) and nm_per_px > 0:
        scale_factor = nm_per_px
        exclusion_zone_px = 2 / nm_per_px
    else:
        scale_factor = 1.0
        exclusion_zone_px = 0.0
    img_h, img_w = data.shape

    aspect_ratio = img_w / img_h if img_h else 1.0
    base_height = 4.0
    fig_width = float(np.clip(base_height * aspect_ratio, 3.0, 7.5))

    fig, ax = plt.subplots(figsize=(fig_width, base_height), dpi=200)
    ax.imshow(data, cmap="gray", interpolation="nearest")
    ax.axis("off")

    for p in regionprops(labels_ws):
        minr, minc, maxr, maxc = p.bbox
        if (
            minr <= exclusion_zone_px
            or minc <= exclusion_zone_px
            or maxr >= img_h - exclusion_zone_px
            or maxc >= img_w - exclusion_zone_px
        ):
            continue

        maj = getattr(p, "major_axis_length", 0.0) or 0.0
        minr_axis = getattr(p, "minor_axis_length", 0.0) or 0.0
        area = float(p.area)
        perim = float(getattr(p, "perimeter", 0.0)) or 0.0
        circ = 4 * np.pi * area / (perim ** 2) if perim > 0 else 0.0
        aspect = maj / minr_axis if minr_axis > 0 else 0.0
        extent = area / ((maxr - minr) * (maxc - minc)) if (maxr - minr) * (maxc - minc) > 0 else 0.0
        solidity = getattr(p, "solidity", 0.0)
        cy, cx = p.centroid  # note: regionprops returns (row, col)

        if shape_type == "Sphere":
            diam_px = (maj + minr_axis) / 2
            d_val = diam_px * scale_factor
            if d_val < min_size_value:
                continue
            diameters_nm.append(d_val)
            _draw_debug_circle(ax, cx, cy, diam_px / 2, "r")
            continue

        vertices, vertex_count = _approximate_region_vertices(p)
        length_val = maj * scale_factor
        width_val = minr_axis * scale_factor
        bbox_tuple = (float(minc), float(minr), float(maxc), float(maxr))
        circularity = circ

        if shape_type == "Hexagon":
            diag_px = math.sqrt((8.0 * area) / (3.0 * math.sqrt(3.0))) if area > 0 else 0.0
            diag_val = diag_px * scale_factor
            if (
                diag_val < min_size_value
                or length_val < min_size_value
                or width_val < min_size_value
            ):
                continue

            if _is_hexagon(vertex_count, aspect, solidity, circularity, extent):
                hex_axes_nm.append(diag_val)
                lengths_nm.append(length_val)
                widths_nm.append(width_val)
                _draw_polygon(ax, vertices, bbox_tuple, color="y")
            else:
                if _is_rectangle(
                    vertex_count,
                    aspect,
                    solidity,
                    circularity,
                    extent,
                    allow_square=False,
                ):
                    _draw_polygon(ax, vertices, bbox_tuple, color="r")
                else:
                    _draw_debug_circle(ax, cx, cy, max(diag_px, maj, minr_axis) / 2, "b")
            continue

        # Cube / rectangular case
        if length_val < min_size_value or width_val < min_size_value:
            continue

        if _is_rectangle(
            vertex_count,
            aspect,
            solidity,
            circularity,
            extent,
            allow_square=True,
        ):
            lengths_nm.append(length_val)
            widths_nm.append(width_val)
            _draw_polygon(ax, vertices, bbox_tuple, color="r")
        else:
            _draw_debug_circle(ax, cx, cy, max(maj, minr_axis) / 2, "b")

    if shape_type == "Hexagon":
        legend_elements = [
            Line2D([0], [0], color="yellow", lw=2, label="Hexagon"),
            Line2D([0], [0], color="red", lw=2, label="Rectangle"),
            Line2D([0], [0], color="blue", lw=2, label="Other"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower left",
            frameon=True,
            facecolor="white",
            framealpha=0.85,
            fontsize=8,
        )

    buf_ann = io.BytesIO()
    fig.savefig(buf_ann, format="png", bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)

    hist_vals = robust_percentile_cut(data, 99.5)
    counts, edges = np.histogram(hist_vals, bins=128)
    centers = edges[:-1] + np.diff(edges) / 2 if len(edges) > 1 else np.array([])
    bar_width = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0

    fig_hist, ax_hist = plt.subplots(figsize=(fig_width, base_height), dpi=200)
    if centers.size:
        ax_hist.bar(
            centers,
            counts,
            width=bar_width,
            align="center",
            color="#888888",
            edgecolor="#444444",
        )
    ax_hist.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"GMM threshold = {threshold:.3f}",
    )
    ax_hist.set_xlabel("Intensity (a.u.)")
    ax_hist.set_ylabel("Pixel count")
    ax_hist.set_title("Intensity histogram")
    ax_hist.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.85)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    buf_hist = io.BytesIO()
    fig_hist.tight_layout()
    fig_hist.savefig(buf_hist, format="png", bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close(fig_hist)

    fig_ws, ax_ws = plt.subplots(figsize=(fig_width, base_height), dpi=200)
    ax_ws.imshow(labels_ws, cmap="gray", interpolation="nearest")
    ax_ws.axis("off")
    buf_ws = io.BytesIO()
    fig_ws.savefig(buf_ws, format="png", bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig_ws)

    out: Dict[str, np.ndarray] = {
        "diameters_nm": np.array(diameters_nm, dtype=np.float32),
        "hex_axes_nm": np.array(hex_axes_nm, dtype=np.float32),
        "lengths_nm": np.array(lengths_nm, dtype=np.float32),
        "widths_nm": np.array(widths_nm, dtype=np.float32),
        "annotated_png": buf_ann.getvalue(),
        "watershed_png": buf_ws.getvalue(),
        "intensity_hist_png": buf_hist.getvalue(),
        "gmm_threshold": float(threshold),
        "image_shape": (int(img_h), int(img_w)),
    }

    # For cubic particles we treat height equal to width (2‑D approximation)
    if shape_type == "Cube":
        out["heights_nm"] = np.array(widths_nm, dtype=np.float32)

    out["unit"] = measurement_unit
    out["nm_per_px"] = float(nm_per_px) if measurement_unit == "nm" else float("nan")

    return out


# ---------------------------------------------------------------------------
# Histogram utilities
# ---------------------------------------------------------------------------


def _gaussian_curve(x: np.ndarray, mu: float, std: float) -> np.ndarray:
    coeff = 1.0 / (std * math.sqrt(2 * math.pi))
    return coeff * np.exp(-0.5 * ((x - mu) / std) ** 2)


def histogram_with_fit(
    values: np.ndarray,
    nbins: int,
    xrange: Tuple[float, float],
    title: str,
    unit_label: str,
) -> Tuple[go.Figure, float, float, int]:
    """Return a histogram figure with a Gaussian fit and dispersion metrics."""

    vals = values[(values >= xrange[0]) & (values <= xrange[1])]
    n = int(len(vals))
    counts, edges = np.histogram(vals, bins=nbins, range=xrange)
    centers = edges[:-1] + np.diff(edges) / 2
    fig = go.Figure(data=[go.Bar(x=centers, y=counts, name=title)])

    if n > 0:
        mu = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else float(np.std(vals, ddof=0))
    else:
        mu = float("nan")
        std = float("nan")

    bin_width = float(edges[1] - edges[0]) if len(edges) > 1 else float(
        (xrange[1] - xrange[0]) / max(nbins, 1)
    )

    if n > 1 and std > 0:
        xline = np.linspace(xrange[0], xrange[1], 300)
        yline = _gaussian_curve(xline, mu, std) * n * bin_width
        fig.add_trace(go.Scatter(x=xline, y=yline, mode="lines", line=dict(color="red")))

    fig.update_layout(title=title, xaxis_title=f"Size ({unit_label})", yaxis_title="Count")
    fig.update_xaxes(range=[float(xrange[0]), float(xrange[1])])
    return fig, float(mu), float(std), n


def _uncertainty_value(std: float, n: int, use_standard_error: bool) -> Optional[float]:
    if not np.isfinite(std):
        return None
    if use_standard_error:
        if n <= 0:
            return None
        if n <= 1:
            return 0.0
        return std / math.sqrt(n)
    return std


def summarise_stats(
    mu: float,
    std: float,
    n: int,
    use_standard_error: bool,
    *,
    extra_parts: Optional[Iterable[str]] = None,
    include_sample_size: bool = True,
) -> Tuple[str, Optional[str], Optional[float]]:
    """Return a formatted summary string and dispersion details."""

    if n <= 0 or not np.isfinite(mu):
        return "", None, None

    parts: List[str] = [f"μ={mu:.2f}"]
    label = "SE" if use_standard_error else "σ"
    unc_value = _uncertainty_value(std, n, use_standard_error)

    if unc_value is not None and np.isfinite(unc_value):
        parts.append(f"{label}={unc_value:.2f}")
    if extra_parts:
        parts.extend(list(extra_parts))
    if include_sample_size:
        parts.append(f"n={n}")

    summary = ", ".join(parts)
    if unc_value is None or not np.isfinite(unc_value):
        label_out: Optional[str] = None
    else:
        label_out = label
    return summary, label_out, unc_value


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def nice_scale_bar_length(size: float) -> Optional[float]:
    """Return a rounded scale-bar length (multiples of 5)."""

    if not np.isfinite(size) or size <= 0:
        return None

    base_values = []
    for exponent in range(0, 6):
        factor = 10 ** exponent
        for base in (5.0, 10.0, 15.0, 20.0, 25.0, 50.0, 75.0):
            base_values.append(base * factor)
    base_values = sorted(set(base_values))

    target = size / 1.2
    candidate = max((val for val in base_values if val <= target), default=None)
    if candidate is None:
        candidate = min((val for val in base_values if val >= target), default=5.0)
    return float(candidate)


def _finalize_shape_figure(
    fig: go.Figure,
    x_extent: float,
    y_extent: float,
    z_extent: float,
    scale_length: Optional[float],
    unit_label: str,
    title: str,
) -> go.Figure:
    x_extent = max(x_extent, 1e-3)
    y_extent = max(y_extent, 1e-3)
    z_extent = max(z_extent, 1e-3)

    if scale_length is not None and scale_length > 0:
        x_extent = max(x_extent, scale_length * 0.6)

    fig.update_layout(
        showlegend=False,
        title=title,
        margin=dict(l=0, r=0, t=40, b=70),
        paper_bgcolor="white",
        scene=dict(
            xaxis=dict(
                range=[-x_extent, x_extent],
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            yaxis=dict(
                range=[-y_extent, y_extent],
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            zaxis=dict(
                range=[-z_extent, z_extent],
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            aspectmode="data",
            bgcolor="white",
            dragmode="orbit",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.25),
                projection=dict(type="orthographic"),
            ),
        ),
    )

    if scale_length is not None and scale_length > 0:
        _add_scale_bar(fig, scale_length, unit_label)

    return fig


def _add_scale_bar(
    fig: go.Figure,
    scale_length: float,
    unit_label: str,
) -> None:
    fig.add_shape(
        type="line",
        x0=0.65,
        x1=0.9,
        y0=0.08,
        y1=0.08,
        xref="paper",
        yref="paper",
        line=dict(color="black", width=4),
        layer="above",
    )
    fig.add_annotation(
        x=0.775,
        y=0.02,
        xref="paper",
        yref="paper",
        text=f"{scale_length:g} {unit_label}",
        showarrow=False,
        font=dict(color="black", size=12),
        align="center",
        bgcolor="rgba(255, 255, 255, 0.85)",
        bordercolor="rgba(0, 0, 0, 0.3)",
        borderwidth=1,
        borderpad=4,
    )


def create_shape_figure(
    shape_type: str,
    unit_label: str,
    *,
    diameter: Optional[float] = None,
    diagonal: Optional[float] = None,
    length: Optional[float] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
) -> Optional[go.Figure]:
    """Create a minimalist 3‑D preview of the fitted shape."""

    if shape_type == "Sphere":
        if diameter is None or not np.isfinite(diameter) or diameter <= 0:
            return None
        radius = diameter / 2.0
        traces: List[go.Scatter3d] = []
        theta = np.linspace(0, 2 * np.pi, 60)
        phi_vals = np.linspace(0, np.pi, 5)[1:-1]
        for phi in phi_vals:
            x = radius * np.cos(theta) * np.sin(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = np.full_like(theta, radius * np.cos(phi))
            traces.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(color="black", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        phi = np.linspace(0, np.pi, 50)
        theta_vals = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        for theta_val in theta_vals:
            x = radius * np.cos(theta_val) * np.sin(phi)
            y = radius * np.sin(theta_val) * np.sin(phi)
            z = radius * np.cos(phi)
            traces.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(color="black", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig = go.Figure(data=traces)
        scale_length = nice_scale_bar_length(diameter)
        return _finalize_shape_figure(fig, radius * 1.2, radius * 1.2, radius * 1.2, scale_length, unit_label, "Sphere preview")

    if shape_type == "Hexagon":
        if (
            diagonal is None
            or not np.isfinite(diagonal)
            or diagonal <= 0
            or length is None
            or width is None
            or not np.isfinite(length)
            or not np.isfinite(width)
        ):
            return None

        radius = diagonal / 2.0
        width_dim = width if abs(width - diagonal) < abs(length - diagonal) else length
        height_dim = length if width_dim == width else width
        if not np.isfinite(height_dim) or height_dim <= 0:
            return None

        theta = np.linspace(0, 2 * np.pi, 7)
        vx = radius * np.cos(theta)
        vy = radius * np.sin(theta)
        hz = height_dim / 2.0

        traces = [
            go.Scatter3d(
                x=vx,
                y=vy,
                z=np.full_like(vx, hz),
                mode="lines",
                line=dict(color="black", width=2),
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter3d(
                x=vx,
                y=vy,
                z=np.full_like(vx, -hz),
                mode="lines",
                line=dict(color="black", width=2),
                hoverinfo="skip",
                showlegend=False,
            ),
        ]
        for i in range(6):
            traces.append(
                go.Scatter3d(
                    x=[vx[i], vx[i]],
                    y=[vy[i], vy[i]],
                    z=[-hz, hz],
                    mode="lines",
                    line=dict(color="black", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        fig = go.Figure(data=traces)
        scale_length = nice_scale_bar_length(max(diagonal, height_dim))
        return _finalize_shape_figure(
            fig,
            radius * 1.2,
            radius * 1.2,
            max(height_dim / 2.0, 1e-3) * 1.2,
            scale_length,
            unit_label,
            "Hexagonal prism preview",
        )

    if shape_type == "Cube":
        if (
            length is None
            or width is None
            or height is None
            or not np.isfinite(length)
            or not np.isfinite(width)
            or not np.isfinite(height)
            or length <= 0
            or width <= 0
            or height <= 0
        ):
            return None

        hx = length / 2.0
        hy = width / 2.0
        hz = height / 2.0
        vertices = np.array(
            [
                [-hx, -hy, -hz],
                [hx, -hy, -hz],
                [hx, hy, -hz],
                [-hx, hy, -hz],
                [-hx, -hy, hz],
                [hx, -hy, hz],
                [hx, hy, hz],
                [-hx, hy, hz],
            ]
        )
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        traces = []
        for start, end in edges:
            traces.append(
                go.Scatter3d(
                    x=vertices[[start, end], 0],
                    y=vertices[[start, end], 1],
                    z=vertices[[start, end], 2],
                    mode="lines",
                    line=dict(color="black", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        fig = go.Figure(data=traces)
        scale_length = nice_scale_bar_length(max(length, width, height))
        return _finalize_shape_figure(
            fig,
            max(hx, 1e-3) * 1.2,
            max(hy, 1e-3) * 1.2,
            max(hz, 1e-3) * 1.2,
            scale_length,
            unit_label,
            "Rectangular prism preview",
        )

    return None


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def preprocess_dm3_payloads(
    file_payloads: Tuple[Tuple[str, bytes], ...],
) -> Tuple[List[PreprocessedImage], List[str], List[Tuple[str, float]]]:
    """Return raw data, measurement metadata and GMM thresholds per file."""

    records: List[PreprocessedImage] = []
    missing_scale_files: List[str] = []
    thresholds: List[Tuple[str, float]] = []

    for file_name, file_bytes in file_payloads:
        dm3 = try_read_dm3(file_bytes)
        data = np.asarray(dm3.data)
        nm_per_px = float(dm3.nm_per_px)

        if np.isfinite(nm_per_px) and nm_per_px > 0:
            measurement_unit = "nm"
        else:
            measurement_unit = "px"
            missing_scale_files.append(file_name)
            nm_per_px = float("nan")

        chosen_threshold = gmm_threshold(data)
        thresholds.append((file_name, chosen_threshold))
        records.append(
            PreprocessedImage(
                name=file_name,
                data=data,
                threshold=chosen_threshold,
                nm_per_px=nm_per_px,
                measurement_unit=measurement_unit,
            )
        )

    return records, missing_scale_files, thresholds


def process_dm3_files(
    records: List[PreprocessedImage],
    shape_type: str,
    min_shape_size_input: float,
) -> List[Dict[str, np.ndarray]]:
    """Segment cached payloads using stored GMM thresholds."""

    results: List[Dict[str, np.ndarray]] = []
    min_shape_size_value = float(min_shape_size_input)

    for record in records:
        seg = segment_and_measure_shapes(
            data=record.data,
            threshold=record.threshold,
            nm_per_px=record.nm_per_px,
            shape_type=shape_type,
            min_size_value=min_shape_size_value,
            measurement_unit=record.measurement_unit,
            min_area_px=5,
        )
        seg["name"] = record.name
        results.append(seg)

    return results


def build_file_payload_key(file_payloads: Tuple[Tuple[str, bytes], ...]) -> str:
    """Return a hash that uniquely represents uploaded files."""

    hasher = hashlib.sha256()
    for file_name, file_bytes in file_payloads:
        hasher.update(file_name.encode("utf-8"))
        hasher.update(len(file_bytes).to_bytes(8, "little"))
        hasher.update(file_bytes)
    return hasher.hexdigest()


def build_processing_key(
    file_key: str,
    shape_type: str,
    min_shape_size_input: float,
) -> str:
    """Return a stable hash describing the current processing inputs."""

    hasher = hashlib.sha256()
    hasher.update(file_key.encode("utf-8"))
    hasher.update(shape_type.encode("utf-8"))
    hasher.update(np.float64(min_shape_size_input).tobytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def run() -> None:  # pragma: no cover - Streamlit entry point
    st.title("TEM Particle Characterization (.dm3)")

    if ncem_dm is None:
        st.error("`ncempy` is not installed. Please run: `pip install ncempy`")
        return

    st.caption(
        "Upload one or more `.dm3` images, choose the particle shape, "
        "then view size distributions."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        files = st.file_uploader("Upload .dm3 file(s)", accept_multiple_files=True, type=["dm3"])

    with col_right:
        shape_type = st.selectbox("Shape type", ["Sphere", "Hexagon", "Cube"], index=0)
        min_shape_size_input = st.number_input(
            "Minimum shape size (nm, or pixels if calibration missing)",
            min_value=0.0,
            max_value=10_000.0,
            value=4.0,
            step=0.5,
        )
        report_standard_error = st.checkbox(
            "Show standard error instead of standard deviation",
            value=False,
            help="Tick to report the standard error (σ/√n) rather than the standard deviation (σ).",
        )
        st.caption(
            "Summaries below include "
            f"{'standard error' if report_standard_error else 'standard deviation'} values."
        )

    results: List[Dict[str, np.ndarray]] = []

    missing_scale_files: List[str] = []

    threshold_messages: List[Tuple[str, float]] = []

    if files:
        file_payloads: Tuple[Tuple[str, bytes], ...] = tuple((f.name, f.getvalue()) for f in files)
        file_key = build_file_payload_key(file_payloads)

        base_entry = st.session_state.get(BASE_CACHE_KEY)
        if base_entry and base_entry.get("key") == file_key:
            base_cache = base_entry
        else:
            with st.spinner("Estimating thresholds …"):
                records, missing_scale_files, threshold_messages = preprocess_dm3_payloads(file_payloads)
            base_cache = {
                "key": file_key,
                "records": records,
                "missing_scale_files": missing_scale_files,
                "threshold_messages": threshold_messages,
            }
            st.session_state[BASE_CACHE_KEY] = base_cache
            st.session_state.pop(SESSION_CACHE_KEY, None)

        records: List[PreprocessedImage] = base_cache["records"]
        min_shape_size_value = float(min_shape_size_input)


        processing_key = build_processing_key(
            file_key=file_key,
            shape_type=shape_type,
            min_shape_size_input=min_shape_size_value,
        )
        cache_entry = st.session_state.get(SESSION_CACHE_KEY)

        if cache_entry and cache_entry.get("key") == processing_key:
            results = cache_entry["results"]
        else:
            with st.spinner("Segmenting shapes …"):
                results = process_dm3_files(
                    records=records,
                    shape_type=shape_type,
                    min_shape_size_input=min_shape_size_value,
                )
            st.session_state[SESSION_CACHE_KEY] = {
                "key": processing_key,
                "results": results,
            }

        for file_name, threshold_value in threshold_messages:
            st.info(f"**{file_name}**: GMM threshold = **{threshold_value:.4f}**")
    else:
        st.session_state.pop(SESSION_CACHE_KEY, None)


    # Dropdown to select image for display
    if results:
        names = [r["name"] for r in results]
        sel_name = st.selectbox("Select image to display", names)
        sel = next(r for r in results if r["name"] == sel_name)

        shape = sel.get("image_shape") if isinstance(sel, dict) else None
        if shape and len(shape) == 2:
            try:
                width_px = int(shape[1])
            except Exception:
                width_px = 0
        else:
            width_px = 0
        display_width = max(320, min(900, width_px // 2)) if width_px > 0 else 600

        tab1, tab2 = st.tabs(["Annotated", "Watershed"])
        with tab1:
            st.image(
                sel["annotated_png"],
                caption=f"Annotated segmentation preview ({sel['unit']})",
                width=display_width,
                use_container_width=False,
            )
        with tab2:
            hist_png = sel.get("intensity_hist_png")
            ws_png = sel.get("watershed_png")
            threshold_value = sel.get("gmm_threshold")
            col_hist, col_ws = st.columns(2)
            with col_hist:
                if hist_png:
                    if threshold_value is not None and np.isfinite(threshold_value):
                        caption = f"Intensity histogram (threshold={threshold_value:.4f})"
                    else:
                        caption = "Intensity histogram"
                    st.image(hist_png, caption=caption, use_container_width=True)
                else:
                    st.caption("Intensity histogram unavailable.")
            with col_ws:
                if ws_png:
                    st.image(ws_png, caption="Watershed labels", use_container_width=True)
                else:
                    st.caption("Watershed labels unavailable.")

        if missing_scale_files:
            missing_list = "\n".join(f"• {name}" for name in sorted(set(missing_scale_files)))
            st.warning(
                "Pixel size metadata was not found for the following file(s). "
                "All measurements (including the minimum shape size filter) are reported in pixels:\n"
                f"{missing_list}"
            )

        # ------------------------------------------------------------------
        # Combined histograms
        # ------------------------------------------------------------------
        st.markdown("---")

        units_present = {r.get("unit", "nm") for r in results}
        if len(units_present) > 1:
            st.error("Cannot combine histograms because uploaded images use mixed measurement units.")
            return

        unit_label = units_present.pop() if units_present else "nm"

        if shape_type == "Sphere":
            all_d = np.concatenate([r["diameters_nm"] for r in results]) if results else np.array([])
            if all_d.size:
                range_slider = st.slider(
                    f"Diameter range ({unit_label})",
                    float(all_d.min()),
                    float(all_d.max()),
                    (float(all_d.min()), float(all_d.max())),
                )
                nbins = st.slider("Histogram bins", 10, 150, 50, step=5)
                fig, mu, std, n = histogram_with_fit(
                    all_d,
                    nbins,
                    range_slider,
                    f"Diameter ({unit_label})",
                    unit_label,
                )
                volume_parts: List[str] = []
                if unit_label == "nm" and np.isfinite(mu):
                    volume = (4 / 3) * np.pi * (mu / 2) ** 3
                    volume_parts.append(f"Vol={volume:.2f} nm³")
                summary, _, _ = summarise_stats(
                    mu,
                    std,
                    n,
                    report_standard_error,
                    extra_parts=volume_parts,
                )
                if summary:
                    fig.update_layout(title=f"Diameter ({unit_label}): {summary}")
                else:
                    fig.update_layout(title=f"Diameter ({unit_label})")
                shape_fig = create_shape_figure(
                    "Sphere",
                    unit_label,
                    diameter=mu if np.isfinite(mu) else None,
                )
                col_hist, col_shape = st.columns([3, 2])
                with col_hist:
                    st.plotly_chart(fig, use_container_width=True)
                    if summary:
                        st.caption(f"Stats: {summary}")
                with col_shape:
                    if shape_fig is not None:
                        st.plotly_chart(
                            shape_fig,
                            use_container_width=True,
                            config=PLOTLY_SHAPE_CONFIG,
                        )
                    else:
                        st.caption("Shape preview available when μ is defined.")
            else:
                st.info("No particles detected.")

        elif shape_type == "Hexagon":
            all_hex = np.concatenate([r["hex_axes_nm"] for r in results]) if results else np.array([])
            all_len = np.concatenate([r["lengths_nm"] for r in results]) if results else np.array([])
            all_wid = np.concatenate([r["widths_nm"] for r in results]) if results else np.array([])
            if all_hex.size and all_len.size and all_wid.size:
                nbins = st.slider("Histogram bins", 10, 150, 50, step=5)

                diag_fig_col, diag_ctrl_col, shape_preview_col = st.columns([4, 1, 2])
                with diag_ctrl_col:
                    range_hex = st.slider(
                        f"Hexagon diagonal range ({unit_label})",
                        float(all_hex.min()),
                        float(all_hex.max()),
                        (float(all_hex.min()), float(all_hex.max())),
                    )

                len_fig_col, len_ctrl_col = st.columns([4, 1])
                with len_ctrl_col:
                    range_len = st.slider(
                        f"Length range ({unit_label})",
                        float(all_len.min()),
                        float(all_len.max()),
                        (float(all_len.min()), float(all_len.max())),
                    )

                wid_fig_col, wid_ctrl_col = st.columns([4, 1])
                with wid_ctrl_col:
                    range_wid = st.slider(
                        f"Width range ({unit_label})",
                        float(all_wid.min()),
                        float(all_wid.max()),
                        (float(all_wid.min()), float(all_wid.max())),
                    )

                fig_hex, mu_hex, std_hex, n_hex = histogram_with_fit(
                    all_hex,
                    nbins,
                    range_hex,
                    f"Hexagon diagonal ({unit_label})",
                    unit_label,
                )
                fig_len, mu_len, std_len, n_len = histogram_with_fit(
                    all_len,
                    nbins,
                    range_len,
                    f"Length ({unit_label})",
                    unit_label,
                )
                fig_wid, mu_wid, std_wid, n_wid = histogram_with_fit(
                    all_wid,
                    nbins,
                    range_wid,
                    f"Width ({unit_label})",
                    unit_label,
                )

                volume_parts: List[str] = []
                if (
                    unit_label == "nm"
                    and np.isfinite(mu_hex)
                    and np.isfinite(mu_len)
                    and np.isfinite(mu_wid)
                ):
                    width = mu_wid if abs(mu_wid - mu_hex) < abs(mu_len - mu_hex) else mu_len
                    height = mu_len if width == mu_wid else mu_wid
                    area_hex = (3 * np.sqrt(3) / 8) * mu_hex ** 2
                    volume = area_hex * height
                    volume_parts.append(f"Vol={volume:.2f} nm³")
                n_min = int(min(n_hex, n_len, n_wid)) if min(n_hex, n_len, n_wid) > 0 else 0

                summary_hex, _, _ = summarise_stats(
                    mu_hex,
                    std_hex,
                    n_hex,
                    report_standard_error,
                    extra_parts=volume_parts,
                )
                summary_len, _, _ = summarise_stats(
                    mu_len,
                    std_len,
                    n_len,
                    report_standard_error,
                    extra_parts=volume_parts,
                )
                summary_wid, _, _ = summarise_stats(
                    mu_wid,
                    std_wid,
                    n_wid,
                    report_standard_error,
                    extra_parts=volume_parts,
                )

                fig_hex.update_layout(
                    title=(
                        f"Hexagon diagonal ({unit_label}): {summary_hex}"
                        if summary_hex
                        else f"Hexagon diagonal ({unit_label})"
                    )
                )
                fig_len.update_layout(
                    title=(
                        f"Length ({unit_label}): {summary_len}"
                        if summary_len
                        else f"Length ({unit_label})"
                    )
                )
                fig_wid.update_layout(
                    title=(
                        f"Width ({unit_label}): {summary_wid}"
                        if summary_wid
                        else f"Width ({unit_label})"
                    )
                )

                with diag_fig_col:
                    st.plotly_chart(fig_hex, use_container_width=True)
                    if summary_hex:
                        st.caption(f"Stats: {summary_hex}")

                with len_fig_col:
                    st.plotly_chart(fig_len, use_container_width=True)
                    if summary_len:
                        st.caption(f"Stats: {summary_len}")

                with wid_fig_col:
                    st.plotly_chart(fig_wid, use_container_width=True)
                    if summary_wid:
                        st.caption(f"Stats: {summary_wid}")

                shape_fig = create_shape_figure(
                    "Hexagon",
                    unit_label,
                    diagonal=mu_hex if np.isfinite(mu_hex) else None,
                    length=mu_len if np.isfinite(mu_len) else None,
                    width=mu_wid if np.isfinite(mu_wid) else None,
                )

                with shape_preview_col:
                    if shape_fig is not None:
                        st.plotly_chart(
                            shape_fig,
                            use_container_width=True,
                            config=PLOTLY_SHAPE_CONFIG,
                        )
                        if volume_parts and n_min:
                            st.caption(
                                f"Volume estimate uses μ values above (overlap n≥{n_min})."
                            )
                    else:
                        st.caption("Shape preview available when fits are valid.")
            else:
                st.info("Insufficient classified particles for histograms.")

        else:  # Cube / rectangular prism
            all_len = np.concatenate([r["lengths_nm"] for r in results]) if results else np.array([])
            all_wid = np.concatenate([r["widths_nm"] for r in results]) if results else np.array([])
            all_hgt = np.concatenate([r.get("heights_nm", []) for r in results]) if results else np.array([])
            if all_len.size and all_wid.size and all_hgt.size:
                range_len = st.slider(
                    f"Length range ({unit_label})",
                    float(all_len.min()),
                    float(all_len.max()),
                    (float(all_len.min()), float(all_len.max())),
                )
                range_wid = st.slider(
                    f"Width range ({unit_label})",
                    float(all_wid.min()),
                    float(all_wid.max()),
                    (float(all_wid.min()), float(all_wid.max())),
                )
                range_hgt = st.slider(
                    f"Height range ({unit_label})",
                    float(all_hgt.min()),
                    float(all_hgt.max()),
                    (float(all_hgt.min()), float(all_hgt.max())),
                )
                nbins = st.slider("Histogram bins", 10, 150, 50, step=5)

                fig_len, mu_len, std_len, n_len = histogram_with_fit(
                    all_len,
                    nbins,
                    range_len,
                    f"Length ({unit_label})",
                    unit_label,
                )
                fig_wid, mu_wid, std_wid, n_wid = histogram_with_fit(
                    all_wid,
                    nbins,
                    range_wid,
                    f"Width ({unit_label})",
                    unit_label,
                )
                fig_hgt, mu_hgt, std_hgt, n_hgt = histogram_with_fit(
                    all_hgt,
                    nbins,
                    range_hgt,
                    f"Height ({unit_label})",
                    unit_label,
                )

                volume_parts: List[str] = []
                if unit_label == "nm" and np.isfinite(mu_len) and np.isfinite(mu_wid) and np.isfinite(mu_hgt):
                    volume = mu_len * mu_wid * mu_hgt
                    volume_parts.append(f"Vol={volume:.2f} nm³")
                n_min = int(min(n_len, n_wid, n_hgt)) if min(n_len, n_wid, n_hgt) > 0 else 0

                summary_len, _, _ = summarise_stats(
                    mu_len,
                    std_len,
                    n_len,
                    report_standard_error,
                    extra_parts=volume_parts,
                )
                summary_wid, _, _ = summarise_stats(
                    mu_wid,
                    std_wid,
                    n_wid,
                    report_standard_error,
                    extra_parts=volume_parts,
                )
                summary_hgt, _, _ = summarise_stats(
                    mu_hgt,
                    std_hgt,
                    n_hgt,
                    report_standard_error,
                    extra_parts=volume_parts,
                )

                fig_len.update_layout(
                    title=(
                        f"Length ({unit_label}): {summary_len}"
                        if summary_len
                        else f"Length ({unit_label})"
                    )
                )
                fig_wid.update_layout(
                    title=(
                        f"Width ({unit_label}): {summary_wid}"
                        if summary_wid
                        else f"Width ({unit_label})"
                    )
                )
                fig_hgt.update_layout(
                    title=(
                        f"Height ({unit_label}): {summary_hgt}"
                        if summary_hgt
                        else f"Height ({unit_label})"
                    )
                )

                shape_fig = create_shape_figure(
                    "Cube",
                    unit_label,
                    length=mu_len if np.isfinite(mu_len) else None,
                    width=mu_wid if np.isfinite(mu_wid) else None,
                    height=mu_hgt if np.isfinite(mu_hgt) else None,
                )
                col_len, col_shape = st.columns([3, 2])
                with col_len:
                    st.plotly_chart(fig_len, use_container_width=True)
                    if summary_len:
                        st.caption(f"Stats: {summary_len}")
                with col_shape:
                    if shape_fig is not None:

                        if volume_parts and n_min:
                            st.caption(
                                f"Volume estimate uses μ values above (overlap n≥{n_min})."
                            )
                    else:
                        st.caption("Shape preview available when fits are valid.")

                st.plotly_chart(fig_wid, use_container_width=True)
                if summary_wid:
                    st.caption(f"Stats: {summary_wid}")
                st.plotly_chart(fig_hgt, use_container_width=True)
                if summary_hgt:
                    st.caption(f"Stats: {summary_hgt}")
            else:
                st.info("Insufficient classified particles for histograms.")
    else:
        st.info("Upload one or more `.dm3` files to begin.")


# When run as `python tools/spherical_tem.py`, start the Streamlit app.
if __name__ == "__main__":  # pragma: no cover - manual execution convenience
    run()

