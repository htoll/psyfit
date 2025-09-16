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
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import distance_transform_edt
from scipy.stats import norm
from skimage.measure import regionprops, label
from skimage.morphology import (
    remove_small_objects,
    binary_closing,
    disk, 
    h_maxima)
from skimage.segmentation import watershed
from streamlit_plotly_events import plotly_events
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

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


def kmeans_threshold(data: np.ndarray, sample: int = 200_000) -> float:
    flat = data.reshape(-1)
    if len(flat) > sample:
        idx = np.random.choice(len(flat), sample, replace=False)
        flat = flat[idx]
    km = KMeans(n_clusters=2, n_init="auto", random_state=42)
    labels = km.fit_predict(flat.reshape(-1, 1))
    c1_max = flat[labels == 0].max()
    c2_max = flat[labels == 1].max()
    return float(min(c1_max, c2_max))


def gmm_threshold(data: np.ndarray, nbins: Optional[int] = None, sample: int = 200_000) -> float:
    flat = data.reshape(-1)
    if len(flat) > sample:
        idx = np.random.choice(len(flat), sample, replace=False)
        flat = flat[idx]
    gm = GaussianMixture(n_components=2, random_state=42)
    gm.fit(flat.reshape(-1, 1))
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
    im_bi = binary_closing(im_bi, disk(3))

    dist = distance_transform_edt(im_bi)
    # Use h-maxima transform to mimic MATLAB's imextendedmin
    hmax = h_maxima(dist, 2)
    markers = label(hmax)
    labels_ws = watershed(-dist, markers=markers, mask=im_bi)
    im_bi[labels_ws == 0] = 0
    im_bi = remove_small_objects(im_bi, min_size=min_area_px)
    labels_ws = label(im_bi)

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
            circ_patch = plt.Circle((cx, cy), diam_px / 2, fill=False, color="r", linewidth=1)
            ax.add_patch(circ_patch)
        else:  # Hexagon or Cube (rectangular)
            if 1.2 < aspect < 1.8 and solidity > 0.8:
                # Rectangle
                length_val = maj * scale_factor
                width_val = minr_axis * scale_factor
                if length_val < min_size_value or width_val < min_size_value:
                    continue
                lengths_nm.append(length_val)
                widths_nm.append(width_val)
                rect = plt.Rectangle(
                    (minc, minr),
                    maxc - minc,
                    maxr - minr,
                    fill=False,
                    color="r",
                    linewidth=1,
                )
                ax.add_patch(rect)
            elif solidity > 0.85 and extent > 0.6:
                # Potential hexagon
                d = (maj + minr_axis) / 2
                d_val = d * scale_factor
                if d_val < min_size_value:
                    continue
                hex_axes_nm.append(d_val)
                radius = d / 2
                theta = np.linspace(0, 2 * np.pi, 7)
                x = cx + radius * np.cos(theta)
                y = cy + radius * np.sin(theta)
                ax.plot(x, y, "y-", linewidth=1)
            else:
                # Mark other shapes lightly for visual debugging
                diam_px = (maj + minr_axis) / 2
                d_val = diam_px * scale_factor
                if d_val >= min_size_value:
                    circ_patch = plt.Circle((cx, cy), diam_px / 2, fill=False, color="b", linewidth=1)
                    ax.add_patch(circ_patch)

    buf_ann = io.BytesIO()
    fig.savefig(buf_ann, format="png", bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)

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


def histogram_with_fit(
    values: np.ndarray,
    nbins: int,
    xrange: Tuple[float, float],
    title: str,
    unit_label: str,
) -> Tuple[go.Figure, float, int]:
    """Return a histogram figure with a Gaussian fit.

    Parameters
    ----------
    values: ndarray
        Data values expressed in ``unit_label`` units.
    nbins: int
        Number of histogram bins.
    xrange: tuple
        (min, max) range to include in the fit.
    title: str
        Base title for the figure.
    """

    vals = values[(values >= xrange[0]) & (values <= xrange[1])]
    n = len(vals)
    counts, edges = np.histogram(vals, bins=nbins)
    centers = edges[:-1] + np.diff(edges) / 2
    fig = go.Figure(data=[go.Bar(x=centers, y=counts, name=title)])

    if n >= 3:
        mu, std = norm.fit(vals)
        xline = np.linspace(xrange[0], xrange[1], 300)
        yline = norm.pdf(xline, mu, std) * n * (centers[1] - centers[0])
        fig.add_trace(go.Scatter(x=xline, y=yline, mode="lines", line=dict(color="red")))
    else:
        mu = float("nan")

    fig.update_layout(title=title, xaxis_title=f"Size ({unit_label})", yaxis_title="Count")
    return fig, float(mu), n


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def run() -> None:  # pragma: no cover - Streamlit entry point
    st.title("TEM Particle Characterization (.dm3)")

    if ncem_dm is None:
        st.error("`ncempy` is not installed. Please run: `pip install ncempy`")
        return

    st.caption(
        "Upload one or more `.dm3` images, choose a thresholding method and the particle shape, "
        "then view size distributions."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        files = st.file_uploader("Upload .dm3 file(s)", accept_multiple_files=True, type=["dm3"])

    with col_right:
        method = st.selectbox(
            "Threshold method",
            ["GMM", "K-means", "Manual"],
            index=0,
        )
        shape_type = st.selectbox("Shape type", ["Sphere", "Hexagon", "Cube"], index=0)
        nbins_int = st.slider("Intensity histogram bins", 20, 200, 60, step=5)
        min_shape_size_input = st.number_input(
            "Minimum shape size (nm, or pixels if calibration missing)",
            min_value=0.0,
            max_value=10_000.0,
            value=4.0,
            step=0.5,
        )

    if "manual_threshold" not in st.session_state:
        st.session_state.manual_threshold = None

    results: List[Dict[str, np.ndarray]] = []

    missing_scale_files: List[str] = []

    if files:
        with st.spinner("Processing …"):
            for i, f in enumerate(files, start=1):
                dm3 = try_read_dm3(f.read())
                data = dm3.data
                nm_per_px = dm3.nm_per_px

                if np.isfinite(nm_per_px) and nm_per_px > 0:
                    measurement_unit = "nm"
                    min_size_value = float(min_shape_size_input)
                else:
                    measurement_unit = "px"
                    min_size_value = float(min_shape_size_input)
                    missing_scale_files.append(f.name)
                    nm_per_px = float("nan")

                # Threshold selection
                if method == "Manual":
                    fig_h = go.Figure()
                    centers, counts = histogram_for_intensity(data, nbins_int)
                    fig_h.add_trace(go.Bar(x=centers, y=counts))
                    fig_h.update_layout(
                        title="Intensity histogram (click to set threshold)",
                        xaxis_title="Intensity",
                        yaxis_title="Frequency",
                    )
                    clicked = plotly_events(fig_h, click_event=True, hover_event=False, select_event=False, key=f"click_{i}")
                    if clicked:
                        st.session_state.manual_threshold = float(clicked[-1]["x"])
                    st.session_state.manual_threshold = st.number_input(
                        "Manual threshold (intensity)",
                        value=float(st.session_state.manual_threshold)
                        if st.session_state.manual_threshold is not None
                        else float(np.median(data)),
                        format="%.6f",
                        key=f"manual_thr_{i}",
                    )
                    chosen_threshold = float(st.session_state.manual_threshold)
                elif method == "K-means":
                    chosen_threshold = kmeans_threshold(data)
                    st.info(f"K-means threshold = **{chosen_threshold:.4f}**")
                else:  # GMM
                    chosen_threshold = gmm_threshold(data, nbins_int)
                    st.info(f"GMM threshold = **{chosen_threshold:.4f}**")

                seg = segment_and_measure_shapes(
                    data=data,
                    threshold=chosen_threshold,
                    nm_per_px=nm_per_px,
                    shape_type=shape_type,
                    min_size_value=min_size_value,
                    measurement_unit=measurement_unit,
                    min_area_px=5,
                )
                seg["name"] = f.name
                results.append(seg)

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
                st.image(
                    sel["watershed_png"],
                    caption="Watershed labels",
                    width=display_width,
                    use_container_width=False,
                )

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
                fig, mu, n = histogram_with_fit(all_d, nbins, range_slider, f"Diameter ({unit_label})", unit_label)
                if unit_label == "nm" and np.isfinite(mu):
                    volume = (4 / 3) * np.pi * (mu / 2) ** 3
                    title_suffix = f"μ={mu:.2f}, Vol={volume:.2f} nm³, n={n}"
                else:
                    title_suffix = f"μ={mu:.2f}, n={n}"
                fig.update_layout(title=f"Diameter ({unit_label}): {title_suffix}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No particles detected.")

        elif shape_type == "Hexagon":
            all_hex = np.concatenate([r["hex_axes_nm"] for r in results]) if results else np.array([])
            all_len = np.concatenate([r["lengths_nm"] for r in results]) if results else np.array([])
            all_wid = np.concatenate([r["widths_nm"] for r in results]) if results else np.array([])
            if all_hex.size and all_len.size and all_wid.size:
                range_hex = st.slider(
                    f"Hexagon diagonal range ({unit_label})",
                    float(all_hex.min()),
                    float(all_hex.max()),
                    (float(all_hex.min()), float(all_hex.max())),
                )
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
                nbins = st.slider("Histogram bins", 10, 150, 50, step=5)

                fig_hex, mu_hex, n_hex = histogram_with_fit(all_hex, nbins, range_hex, f"Hexagon diagonal ({unit_label})", unit_label)
                fig_len, mu_len, n_len = histogram_with_fit(all_len, nbins, range_len, f"Length ({unit_label})", unit_label)
                fig_wid, mu_wid, n_wid = histogram_with_fit(all_wid, nbins, range_wid, f"Width ({unit_label})", unit_label)

                # Volume of hexagonal prism
                if unit_label == "nm" and np.isfinite(mu_hex) and np.isfinite(mu_len) and np.isfinite(mu_wid):
                    width = mu_wid if abs(mu_wid - mu_hex) < abs(mu_len - mu_hex) else mu_len
                    height = mu_len if width == mu_wid else mu_wid
                    area_hex = (3 * np.sqrt(3) / 8) * mu_hex ** 2
                    volume = area_hex * height
                    volume_text = f", Vol={volume:.2f} nm³"
                else:
                    volume_text = ""
                n_min = int(min(n_hex, n_len, n_wid))

                fig_hex.update_layout(title=f"Hexagon diagonal ({unit_label}): μ={mu_hex:.2f}{volume_text}, n≥{n_min}")
                fig_len.update_layout(title=f"Length ({unit_label}): μ={mu_len:.2f}{volume_text}, n≥{n_min}")
                fig_wid.update_layout(title=f"Width ({unit_label}): μ={mu_wid:.2f}{volume_text}, n≥{n_min}")

                st.plotly_chart(fig_hex, use_container_width=True)
                st.plotly_chart(fig_len, use_container_width=True)
                st.plotly_chart(fig_wid, use_container_width=True)
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

                fig_len, mu_len, n_len = histogram_with_fit(all_len, nbins, range_len, f"Length ({unit_label})", unit_label)
                fig_wid, mu_wid, n_wid = histogram_with_fit(all_wid, nbins, range_wid, f"Width ({unit_label})", unit_label)
                fig_hgt, mu_hgt, n_hgt = histogram_with_fit(all_hgt, nbins, range_hgt, f"Height ({unit_label})", unit_label)

                if unit_label == "nm" and np.isfinite(mu_len) and np.isfinite(mu_wid) and np.isfinite(mu_hgt):
                    volume = mu_len * mu_wid * mu_hgt
                    volume_text = f", Vol={volume:.2f} nm³"
                else:
                    volume_text = ""
                n_min = int(min(n_len, n_wid, n_hgt))

                fig_len.update_layout(title=f"Length ({unit_label}): μ={mu_len:.2f}{volume_text}, n≥{n_min}")
                fig_wid.update_layout(title=f"Width ({unit_label}): μ={mu_wid:.2f}{volume_text}, n≥{n_min}")
                fig_hgt.update_layout(title=f"Height ({unit_label}): μ={mu_hgt:.2f}{volume_text}, n≥{n_min}")

                st.plotly_chart(fig_len, use_container_width=True)
                st.plotly_chart(fig_wid, use_container_width=True)
                st.plotly_chart(fig_hgt, use_container_width=True)
            else:
                st.info("Insufficient classified particles for histograms.")
    else:
        st.info("Upload one or more `.dm3` files to begin.")


# When run as `python tools/spherical_tem.py`, start the Streamlit app.
if __name__ == "__main__":  # pragma: no cover - manual execution convenience
    run()

