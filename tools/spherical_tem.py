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
    remove_small_holes,  
    binary_closing,
    disk, 
    h_maxima)
from skimage.segmentation import watershed
from streamlit_plotly_events import plotly_events
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
import zipfile
from sklearn.mixture import GaussianMixture
import pandas as pd

import dm3_lib as pyDM3reader
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


def _find_dimension_tags(tags: dict) -> list:
    """Recursively find all 'Dimension' tag groups in the metadata."""
    found = []
    if isinstance(tags, dict):
        for key, value in tags.items():
            if key == "Dimension" and isinstance(value, dict):
                # A Dimension group contains axis info keyed by '0', '1', etc.
                # We add the whole group to our list of candidates.
                found.append(value)
            elif isinstance(value, dict):
                found.extend(_find_dimension_tags(value))
            elif isinstance(value, list):
                for item in value:
                    found.extend(_find_dimension_tags(item))
    return found


def try_read_dm3(file_bytes: bytes) -> DM3Image:
    """Read a dm3 file into a :class:`DM3Image` instance."""

    if pyDM3reader is None:  # pragma: no cover - runtime check in UI
        raise RuntimeError("dm3_lib is not installed. Please install it (pip install dm3_lib).")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dm3")
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()

        # pyDM3reader (dm3_lib) reads from a file path.
        dm3_file = pyDM3reader.DM3(tmp.name)
        data = np.array(dm3_file.imagedata, dtype=np.float32)
        
        # Initialize defaults
        pixel_size = 0
        pixel_unit = ""

        # 1. Try standard attribute provided by the library
        if hasattr(dm3_file, "pxsize"):
            pixel_size, pixel_unit = dm3_file.pxsize

        # 2. If missing, try to dig into the tags using the existing helper function
        if pixel_size == 0 and hasattr(dm3_file, "tags"):
            # Find all "Dimension" groups in the metadata
            dim_groups = _find_dimension_tags(dm3_file.tags)
            
            for group in dim_groups:
                # 'group' corresponds to the "Dimension" tag. 
                # In dm3_lib, arrays often appear as dicts with keys '0', '1', etc.
                # We check the first dimension (usually X-axis, index '0')
                if '0' in group and isinstance(group['0'], dict):
                    dim_data = group['0']
                    scale = dim_data.get('Scale', 0)
                    units = dim_data.get('Units', '')
                    if scale > 0:
                        pixel_size = scale
                        pixel_unit = units
                        break # Stop if we found a valid scale

        # 3. Fallback: Check for specific "Pixel Size (um)" tag found in your file
        # This manually searches the "ImageTags" dictionary if it exists.
        if pixel_size == 0 and hasattr(dm3_file, "tags"):
             root_tags = dm3_file.tags
             # Navigate roughly: Root -> ImageTags -> Pixel Size (um)
             # Note: Structure depends on exact file version, we search somewhat safely
             if 'ImageTags' in root_tags:
                 img_tags = root_tags['ImageTags']
                 if 'Pixel Size (um)' in img_tags:
                     pixel_size = img_tags['Pixel Size (um)']
                     pixel_unit = 'µm' # Explicitly set microns based on tag name

        # 4. Convert to nanometers for the application
        nm_per_px = float("nan")
        if pixel_size > 0:
            # Clean up unit string (remove null bytes or weird formatting)
            pixel_unit = str(pixel_unit).strip().replace('\x00', '')
            
            if pixel_unit == "m":
                nm_per_px = pixel_size * 1e9
            elif pixel_unit == "nm":
                nm_per_px = pixel_size
            elif pixel_unit in ["µm", "um", "micron", "microns"]:
                nm_per_px = pixel_size * 1e3
            else:
                # Assume nm if unit is weird but size is small, or just pass raw
                # But for safety, let's assume raw if unknown unit, or maybe user has nm
                nm_per_px = pixel_size # Default fallback

        return DM3Image(data=data, nm_per_px=nm_per_px)
    except Exception as e:
        st.warning(f"Failed to read DM3 file with dm3_lib: {e}")
        return DM3Image(data=np.array([[]]), nm_per_px=float("nan"))
    finally:
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
    
    if flat.size < 2:
        return float(np.median(data)) if data.size > 0 else 0.0

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
) -> Dict[str, any]:
    """Segment particles and measure their dimensions."""

    # Binary image and watershed segmentation
    im_bi = data < threshold
    # Robustness: Fill internal holes common in dark TEM particles
    im_bi = remove_small_holes(im_bi, area_threshold=min_area_px)
    im_bi = binary_closing(im_bi, disk(3))

    dist = distance_transform_edt(im_bi)
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

    # Scaling and exclusion
    if measurement_unit == "nm" and np.isfinite(nm_per_px) and nm_per_px > 0:
        scale_factor = nm_per_px
        exclusion_zone_px = 2 / nm_per_px
    else:
        scale_factor = 1.0
        exclusion_zone_px = 0.0
    img_h, img_w = data.shape

    # Visualization: Contrast stretching for better visibility
    fig = go.Figure()
    fig.add_trace(go.Image(z=data))#, colormap="gray"))

    # Store shapes to be added to the figure
    fig_shapes = []
    fig_annotations = []

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
        cy, cx = p.centroid

        if shape_type == "Sphere":
            diam_px = (maj + minr_axis) / 2
            d_val = diam_px * scale_factor
            if d_val < min_size_value:
                continue
            diameters_nm.append(d_val)
            fig_shapes.append(
                dict(type="circle", x0=cx-diam_px/2, y0=cy-diam_px/2, x1=cx+diam_px/2, y1=cy+diam_px/2, 
                     line=dict(color="rgba(255, 0, 0, 0.5)", width=2))
            )
        else:
            # Heuristic classification for Hexagons vs. Cubes/Rectangles
            # This part can be refined based on more geometric properties
            is_hex = solidity > 0.85 and extent > 0.6
            is_rect = 1.2 < aspect < 1.8 and solidity > 0.8

            if 1.2 < aspect < 1.8 and solidity > 0.8:
                length_val = maj * scale_factor
                width_val = minr_axis * scale_factor
                if length_val < min_size_value or width_val < min_size_value:
                    continue
                lengths_nm.append(length_val)
                widths_nm.append(width_val)
                fig_shapes.append(
                    dict(type="rect", x0=minc, y0=minr, x1=maxc, y1=maxr,
                         line=dict(color="rgba(255, 0, 0, 0.5)", width=2))
                )
            elif solidity > 0.85 and extent > 0.6:
                d = (maj + minr_axis) / 2
                d_val = d * scale_factor
                if d_val < min_size_value:
                    continue
                hex_axes_nm.append(d_val)
                radius = d / 2
                # Create a path for a hexagon
                path = f"M {cx+radius} {cy} " + " ".join([f"L {cx+radius*np.cos(t)} {cy+radius*np.sin(t)}" for t in np.linspace(np.pi/3, 2*np.pi-np.pi/3, 5)]) + " Z"
                fig_shapes.append(
                    dict(type="path", path=path, line=dict(color="rgba(255, 255, 0, 0.5)", width=2))
                )
            else:
                diam_px = (maj + minr_axis) / 2
                d_val = diam_px * scale_factor
                if d_val >= min_size_value:
                    fig_shapes.append(
                        dict(type="circle", x0=cx-diam_px/2, y0=cy-diam_px/2, x1=cx+diam_px/2, y1=cy+diam_px/2,
                             line=dict(color="rgba(0, 0, 255, 0.5)", width=2))
                    )

    fig.update_layout(
        shapes=fig_shapes,
        margin=dict(l=0, r=0, b=0, t=0),
    )
    fig.update_xaxes(visible=False, range=[0, img_w])
    fig.update_yaxes(visible=False, range=[img_h, 0]) # Invert y-axis for image coordinates

    # Create a white-background watershed image
    # Generate a colormap with a white background
    rand_cmap = np.random.rand(labels_ws.max() + 1, 3)
    rand_cmap[0] = [1, 1, 1]  # Set background (label 0) to white
    cmap_ws = ListedColormap(rand_cmap)
    fig_ws = go.Figure(data=go.Heatmap(
    z=labels_ws,
    colorscale='Rainbow',  # 'Rainbow' is a good high-contrast built-in choice
    zmin=0,
    zmax=labels_ws.max(),
    showscale=False
    ))
    # Fix orientation to match standard image coordinates (0,0 at top-left)
    fig_ws.update_yaxes(autorange='reversed')
    fig_ws.update_layout(width=img_w, height=img_h, margin=dict(l=0, r=0, b=0, t=0))
    fig_ws.update_xaxes(visible=False)
    fig_ws.update_yaxes(visible=False, autorange="reversed")
    out: Dict[str, any] = {
        "diameters_nm": np.array(diameters_nm, dtype=np.float32),
        "hex_axes_nm": np.array(hex_axes_nm, dtype=np.float32),
        "lengths_nm": np.array(lengths_nm, dtype=np.float32),
        "widths_nm": np.array(widths_nm, dtype=np.float32),
        "annotated_fig": fig,    # Return figure object
        "watershed_fig": fig_ws, # Return figure object
    }

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
        bin_width = centers[1] - centers[0] if len(centers) > 1 else 1
        yline = norm.pdf(xline, mu, std) * n * bin_width
        fig.add_trace(go.Scatter(x=xline, y=yline, mode="lines", line=dict(color="red"), name=f"μ={mu:.2f}, σ={std:.2f}"))
    else:
        mu = float("nan")

    fig.update_layout(title=title, xaxis_title=f"Size ({unit_label})", yaxis_title="Count")
    return fig, float(mu), n


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
method = "GMM"


def get_common_prefix(names: List[str]) -> str:
    """Find the common prefix among a list of strings."""
    if not names:
        return "analysis"
    shortest = min(names, key=len)
    for i, char in enumerate(shortest):
        if any(name[i] != char for name in names):
            return shortest[:i].rstrip(" _-") or "analysis"
    return shortest.rstrip(" _-") or "analysis"


@st.cache_data
def analyze_image(file_bytes: bytes, method: str, shape_type: str, min_shape_size_input: float, nbins_int: int, manual_threshold: float | None) -> dict:
    """Cached function to perform the full analysis of a single image."""
    dm3 = try_read_dm3(file_bytes)
    data = dm3.data
    nm_per_px = dm3.nm_per_px

    if data.size == 0:
        return {"error": "Image data is empty."}

    if np.isfinite(nm_per_px) and nm_per_px > 0:
        measurement_unit = "nm"
    else:
        measurement_unit = "px"
        nm_per_px = float("nan")

    # Threshold selection
    if method == "Manual":
        if manual_threshold is not None:
            chosen_threshold = manual_threshold
        else:
            chosen_threshold = float(np.median(data))
    elif method == "K-means":
        chosen_threshold = kmeans_threshold(data)
    else:  # GMM
        chosen_threshold = gmm_threshold(data, nbins_int)

    seg = segment_and_measure_shapes(
        data=data,
        threshold=chosen_threshold,
        nm_per_px=nm_per_px,
        shape_type=shape_type,
        min_size_value=float(min_shape_size_input),
        measurement_unit=measurement_unit,
        min_area_px=5,
    )
    
    # Add chosen threshold to results for display
    seg["chosen_threshold"] = chosen_threshold

    return seg


def run() -> None:  # pragma: no cover - Streamlit entry point
    st.title("TEM Particle Characterization (.dm3)")

    if pyDM3reader is None:
        st.error("`dm3_lib` is not installed. Please run: `pip install dm3_lib`")
        return

    st.caption(
        "Upload one or more `.dm3` images, choose a thresholding method and the particle shape, "
        "then view size distributions."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        files = st.file_uploader("Upload .dm3 file(s)", accept_multiple_files=True, type=["dm3"])

    with col_right:
        # method = st.selectbox(
        #     "Threshold method",
        #     ["GMM", "K-means", "Manual"],
        #     index=0,
        
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
        with st.status("Processing images...", expanded=True) as status:
            for i, f in enumerate(files, start=1):
                status.update(label=f"Processing image {i}/{len(files)}: {f.name}")
                
                # Use the cached analysis function
                seg = analyze_image(f.read(), method, shape_type, min_shape_size_input, nbins_int, st.session_state.manual_threshold)
                
                if seg.get("unit", "px") == "px":
                    missing_scale_files.append(f.name)

                seg["name"] = f.name
                results.append(seg)
            status.update(label="Processing complete!", state="complete", expanded=False)

        # Dropdown to select image for display
        if results:
            names = [r["name"] for r in results]
            sel_name = st.selectbox("Select image to display", names)
            sel = next(r for r in results if r["name"] == sel_name)
            
            st.info(f"Computed threshold = **{sel.get('chosen_threshold', 0):.4f}**")
            tab1, tab2 = st.tabs(["Annotated", "Watershed"])
            with tab1:
                st.caption(f"Annotated segmentation preview ({sel['unit']})")
                st.plotly_chart(sel["annotated_fig"], use_container_width=True)
            with tab2:
                st.caption("Watershed labels (randomly colored)")
                st.plotly_chart(sel["watershed_fig"], use_container_width=True)
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
                    sample_name = get_common_prefix([r["name"] for r in results])
                    volume = (4 / 3) * np.pi * (mu / 2) ** 3
                    title_text = f"{sample_name}<br><sup>Vol={volume:.2f} nm³, n={n}</sup>"
                else:
                    title_text = f"Diameter ({unit_label}): μ={mu:.2f}, n={n}"
                fig.update_layout(title_text=title_text)
                st.plotly_chart(fig, use_container_width=True)

                # Download buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button("Download PNG", fig.to_image(format="png"), f"{sample_name}_diameter.png", "image/png")
                with col2:
                    st.download_button("Download CSV", pd.DataFrame({'diameter': all_d}).to_csv(index=False), f"{sample_name}_diameter.csv", "text/csv")
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
