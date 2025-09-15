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

    # Edge exclusion zone (2 nm)
    exclusion_zone_px = 2 / nm_per_px if nm_per_px > 0 else 0
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
            d_nm = diam_px * nm_per_px
            if d_nm < 2:
                continue
            diameters_nm.append(d_nm)
            circ_patch = plt.Circle((cx, cy), diam_px / 2, fill=False, color="r", linewidth=1)
            ax.add_patch(circ_patch)
        else:  # Hexagon or Cube (rectangular)
            if 1.2 < aspect < 1.8 and solidity > 0.8:
                # Rectangle
                lengths_nm.append(maj * nm_per_px)
                widths_nm.append(minr_axis * nm_per_px)
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
                hex_axes_nm.append(d * nm_per_px)
                radius = d / 2
                theta = np.linspace(0, 2 * np.pi, 7)
                x = cx + radius * np.cos(theta)
                y = cy + radius * np.sin(theta)
                ax.plot(x, y, "y-", linewidth=1)
            else:
                # Mark other shapes lightly for visual debugging
                diam_px = (maj + minr_axis) / 2
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

    return out


# ---------------------------------------------------------------------------
# Histogram utilities
# ---------------------------------------------------------------------------


def histogram_with_fit(
    values: np.ndarray,
    nbins: int,
    xrange: Tuple[float, float],
    title: str,
) -> Tuple[go.Figure, float, int]:
    """Return a histogram figure with a Gaussian fit.

    Parameters
    ----------
    values: ndarray
        Data values in nanometres.
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

    fig.update_layout(title=title, xaxis_title="Size (nm)", yaxis_title="Count")
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
        default_nm_per_px = st.number_input(
            "Fallback pixel size (nm per px)",
            min_value=0.0001,
            max_value=1000.0,
            value=1.0,
            step=0.1,
        )

    if "manual_threshold" not in st.session_state:
        st.session_state.manual_threshold = None

    results: List[Dict[str, np.ndarray]] = []

    if files:
        with st.spinner("Processing …"):
            for i, f in enumerate(files, start=1):
                dm3 = try_read_dm3(f.read())
                data = dm3.data
                nm_per_px = dm3.nm_per_px if np.isfinite(dm3.nm_per_px) else default_nm_per_px

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
                    min_area_px=5,
                )
                seg["name"] = f.name
                results.append(seg)

        # Dropdown to select image for display
        names = [r["name"] for r in results]
        sel_name = st.selectbox("Select image to display", names)
        sel = next(r for r in results if r["name"] == sel_name)

        tab1, tab2 = st.tabs(["Annotated", "Watershed"])


        # ------------------------------------------------------------------
        # Combined histograms
        # ------------------------------------------------------------------
        st.markdown("---")
        if shape_type == "Sphere":
            all_d = np.concatenate([r["diameters_nm"] for r in results]) if results else np.array([])
            if all_d.size:
                range_slider = st.slider(
                    "Diameter range (nm)",
                    float(all_d.min()),
                    float(all_d.max()),
                    (float(all_d.min()), float(all_d.max())),
                )
                nbins = st.slider("Histogram bins", 10, 150, 50, step=5)
                fig, mu, n = histogram_with_fit(all_d, nbins, range_slider, "Diameter (nm)")
                volume = (4 / 3) * np.pi * (mu / 2) ** 3 if np.isfinite(mu) else float("nan")
                fig.update_layout(title=f"Diameter (nm): μ={mu:.2f}, Vol={volume:.2f} nm³, n={n}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No particles detected.")

        elif shape_type == "Hexagon":
            all_hex = np.concatenate([r["hex_axes_nm"] for r in results]) if results else np.array([])
            all_len = np.concatenate([r["lengths_nm"] for r in results]) if results else np.array([])
            all_wid = np.concatenate([r["widths_nm"] for r in results]) if results else np.array([])
            if all_hex.size and all_len.size and all_wid.size:
                range_hex = st.slider(
                    "Hexagon diagonal range (nm)",
                    float(all_hex.min()),
                    float(all_hex.max()),
                    (float(all_hex.min()), float(all_hex.max())),
                )
                range_len = st.slider(
                    "Length range (nm)",
                    float(all_len.min()),
                    float(all_len.max()),
                    (float(all_len.min()), float(all_len.max())),
                )
                range_wid = st.slider(
                    "Width range (nm)",
                    float(all_wid.min()),
                    float(all_wid.max()),
                    (float(all_wid.min()), float(all_wid.max())),
                )
                nbins = st.slider("Histogram bins", 10, 150, 50, step=5)

                fig_hex, mu_hex, n_hex = histogram_with_fit(all_hex, nbins, range_hex, "Hexagon diagonal (nm)")
                fig_len, mu_len, n_len = histogram_with_fit(all_len, nbins, range_len, "Length (nm)")
                fig_wid, mu_wid, n_wid = histogram_with_fit(all_wid, nbins, range_wid, "Width (nm)")

                # Volume of hexagonal prism
                width = mu_wid if abs(mu_wid - mu_hex) < abs(mu_len - mu_hex) else mu_len
                height = mu_len if width == mu_wid else mu_wid
                area_hex = (3 * np.sqrt(3) / 8) * mu_hex ** 2
                volume = area_hex * height
                n_min = int(min(n_hex, n_len, n_wid))

                fig_hex.update_layout(title=f"Hexagon diagonal (nm): μ={mu_hex:.2f}, Vol={volume:.2f} nm³, n≥{n_min}")
                fig_len.update_layout(title=f"Length (nm): μ={mu_len:.2f}, Vol={volume:.2f} nm³, n≥{n_min}")
                fig_wid.update_layout(title=f"Width (nm): μ={mu_wid:.2f}, Vol={volume:.2f} nm³, n≥{n_min}")

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
                    "Length range (nm)",
                    float(all_len.min()),
                    float(all_len.max()),
                    (float(all_len.min()), float(all_len.max())),
                )
                range_wid = st.slider(
                    "Width range (nm)",
                    float(all_wid.min()),
                    float(all_wid.max()),
                    (float(all_wid.min()), float(all_wid.max())),
                )
                range_hgt = st.slider(
                    "Height range (nm)",
                    float(all_hgt.min()),
                    float(all_hgt.max()),
                    (float(all_hgt.min()), float(all_hgt.max())),
                )
                nbins = st.slider("Histogram bins", 10, 150, 50, step=5)

                fig_len, mu_len, n_len = histogram_with_fit(all_len, nbins, range_len, "Length (nm)")
                fig_wid, mu_wid, n_wid = histogram_with_fit(all_wid, nbins, range_wid, "Width (nm)")
                fig_hgt, mu_hgt, n_hgt = histogram_with_fit(all_hgt, nbins, range_hgt, "Height (nm)")

                volume = mu_len * mu_wid * mu_hgt
                n_min = int(min(n_len, n_wid, n_hgt))

                fig_len.update_layout(title=f"Length (nm): μ={mu_len:.2f}, Vol={volume:.2f} nm³, n≥{n_min}")
                fig_wid.update_layout(title=f"Width (nm): μ={mu_wid:.2f}, Vol={volume:.2f} nm³, n≥{n_min}")
                fig_hgt.update_layout(title=f"Height (nm): μ={mu_hgt:.2f}, Vol={volume:.2f} nm³, n≥{n_min}")

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

