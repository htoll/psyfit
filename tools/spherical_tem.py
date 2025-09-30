"""Interactive TEM particle analysis tool.

This Streamlit app reads `.dm3` and `.tif/.tiff` files and measures particle sizes.  It now
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
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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
from sklearn.mixture import GaussianMixture

# Optional import of ncempy (for reading dm3 files)
try:  # pragma: no cover - simply a convenience check
    from ncempy.io import dm as ncem_dm
except Exception:  # pragma: no cover
    ncem_dm = None

# Optional import for TIFF reading
try:  # pragma: no cover - runtime dependency check
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DM3Image:
    data: np.ndarray
    nm_per_px: float


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

            nm_per_px = np.nan
            md = rdr.allTags
            candidates = [
                ("ImageList.1.ImageData.Calibrations.Dimension.0.Scale", 1e9),
                ("ImageList.1.ImageData.Calibrations.Dimension.1.Scale", 1e9),
                ("pixelSize.x", 1e9),
                ("pixelSize", 1e9),
                ("xscale", 1e9),
                ("ImageList.2.ImageData.Calibrations.Dimension.0.Scale", 1e9),
                ("ImageList.2.ImageData.Calibrations.Dimension.1.Scale", 1e9),
            ]
            for key, factor in candidates:
                try:
                    val = md
                    for k in key.split("."):
                        val = val[k]
                    if isinstance(val, (int, float)):
                        nm_per_px = float(val) * factor
                        break
                except Exception:
                    continue

        return DM3Image(data=data, nm_per_px=nm_per_px)
    finally:  # pragma: no cover - best effort cleanup
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def try_read_tiff(file_bytes: bytes) -> DM3Image:
    """Read a TIFF file into a :class:`DM3Image` instance."""

    if tifffile is None:  # pragma: no cover - runtime check in UI
        raise RuntimeError("tifffile is not installed. Please install it (pip install tifffile).")

    with tifffile.TiffFile(io.BytesIO(file_bytes)) as tif:
        arr = tif.asarray()
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim == 3:
            if arr.shape[-1] in (3, 4):
                arr = arr[..., :3].mean(axis=-1)
            else:
                arr = arr[0]
        data = np.array(arr, dtype=np.float32)

        nm_per_px = float("nan")
        try:
            page0 = tif.pages[0]
            res_tag = page0.tags.get("XResolution")
            unit_tag = page0.tags.get("ResolutionUnit")
            if res_tag is not None and unit_tag is not None:
                num, den = res_tag.value
                if den:
                    px_per_unit = num / den
                    unit_val = unit_tag.value
                    if unit_val == 2:  # inch
                        nm_per_unit = 25_400_000.0
                    elif unit_val == 3:  # centimeter
                        nm_per_unit = 10_000_000.0
                    else:
                        nm_per_unit = float("nan")
                    if np.isfinite(px_per_unit) and np.isfinite(nm_per_unit):
                        nm_per_px = nm_per_unit / px_per_unit
        except Exception:
            nm_per_px = float("nan")

    return DM3Image(data=data, nm_per_px=nm_per_px)


def read_tem_image(file_bytes: bytes, filename: str) -> DM3Image:
    """Dispatch reading logic based on file extension."""

    ext = os.path.splitext(filename)[1].lower()
    if ext == ".dm3":
        return try_read_dm3(file_bytes)
    if ext in {".tif", ".tiff"}:
        return try_read_tiff(file_bytes)
    raise RuntimeError(f"Unsupported file type: {ext or 'unknown'}")


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

    fig, ax = plt.subplots()
    ax.imshow(data, cmap="gray")
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
    fig.savefig(buf_ann, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    fig_ws, ax_ws = plt.subplots()
    ax_ws.imshow(labels_ws, cmap="gray")
    ax_ws.axis("off")
    buf_ws = io.BytesIO()
    fig_ws.savefig(buf_ws, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig_ws)

    out: Dict[str, np.ndarray] = {
        "diameters_nm": np.array(diameters_nm, dtype=np.float32),
        "hex_axes_nm": np.array(hex_axes_nm, dtype=np.float32),
        "lengths_nm": np.array(lengths_nm, dtype=np.float32),
        "widths_nm": np.array(widths_nm, dtype=np.float32),
        "annotated_png": buf_ann.getvalue(),
        "watershed_png": buf_ws.getvalue(),
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
# Cached processing helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def process_tem_file(
    file_bytes: bytes,
    filename: str,
    nbins_int: int,
    shape_type: str,
    min_shape_size_input: float,
) -> Dict[str, np.ndarray]:
    """Process an uploaded TEM image and return segmentation results."""

    image = read_tem_image(file_bytes, filename)
    data = image.data
    nm_per_px = image.nm_per_px

    if np.isfinite(nm_per_px) and nm_per_px > 0:
        measurement_unit = "nm"
        min_size_value = float(min_shape_size_input)
    else:
        measurement_unit = "px"
        min_size_value = float(min_shape_size_input)
        nm_per_px = float("nan")

    chosen_threshold = gmm_threshold(data, nbins_int)

    seg = segment_and_measure_shapes(
        data=data,
        threshold=chosen_threshold,
        nm_per_px=nm_per_px,
        shape_type=shape_type,
        min_size_value=min_size_value,
        measurement_unit=measurement_unit,
        min_area_px=5,
    )
    seg["name"] = filename
    seg["threshold"] = chosen_threshold
    seg["missing_scale"] = measurement_unit != "nm"
    return seg


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def run() -> None:  # pragma: no cover - Streamlit entry point
    st.title("TEM Particle Characterization (.dm3/.tif)")

    st.caption(
        "Upload one or more `.dm3` or `.tif/.tiff` images, choose the particle shape, "
        "then view size distributions. Thresholding uses a Gaussian mixture model."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        files = st.file_uploader(
            "Upload TEM file(s)",
            accept_multiple_files=True,
            type=["dm3", "tif", "tiff"],
        )

    with col_right:
        shape_type = st.selectbox("Shape type", ["Sphere", "Hexagon", "Cube"], index=0)
        nbins_int = st.slider("Intensity histogram bins", 20, 200, 60, step=5)
        min_shape_size_input = st.number_input(
            "Minimum shape size (nm, or pixels if calibration missing)",
            min_value=0.0,
            max_value=10_000.0,
            value=4.0,
            step=0.5,
        )

    results: List[Dict[str, np.ndarray]] = []
    missing_scale_files: List[str] = []
    processing_errors: List[str] = []

    if files:
        with st.spinner("Processing …"):
            for f in files:
                file_bytes = f.getvalue()
                try:
                    seg = process_tem_file(
                        file_bytes=file_bytes,
                        filename=f.name,
                        nbins_int=int(nbins_int),
                        shape_type=shape_type,
                        min_shape_size_input=float(min_shape_size_input),
                    )
                except RuntimeError as exc:
                    processing_errors.append(f"{f.name}: {exc}")
                    continue

                results.append(seg)
                if seg.get("missing_scale", False):
                    missing_scale_files.append(seg.get("name", f.name))

        if processing_errors:
            for err in processing_errors:
                st.error(err)

        if results:
            for seg in results:
                st.info(f"{seg['name']}: GMM threshold = **{seg['threshold']:.4f}**")

            names = [r["name"] for r in results]
            sel_name = st.selectbox("Select image to display", names)
            sel = next(r for r in results if r["name"] == sel_name)

            tab1, tab2 = st.tabs(["Annotated", "Watershed"])
            with tab1:
                st.image(sel["annotated_png"], caption=f"Annotated segmentation preview ({sel['unit']})", use_column_width=True)
            with tab2:
                st.image(sel["watershed_png"], caption="Watershed labels", use_column_width=True)
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
        elif processing_errors:
            st.info("No files were processed successfully. Please review the errors above.")
    else:
        st.info("Upload one or more `.dm3` or `.tif/.tiff` files to begin.")


# When run as `python tools/spherical_tem.py`, start the Streamlit app.
if __name__ == "__main__":  # pragma: no cover - manual execution convenience
    run()

