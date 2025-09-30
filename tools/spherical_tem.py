"""
Interactive TEM particle analysis tool (Streamlit) — pyDM3reader edition.

Key changes:
- Swap ncempy for pyDM3reader (dm3_lib/DM3lib).
- Read image via DM3.imagedata and pixel size via DM3.pxsize.
- Robust unit normalization to nm/px (handles bytes/str and common variants).
- Keep watershedding/segmentation exactly as in your original code.
"""

from __future__ import annotations
import io
import math
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.stats import norm
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.feature import peak_local_max
from skimage.measure import regionprops, label
from skimage.morphology import (
    remove_small_objects,
    binary_closing,
    binary_opening,
    disk,
)
from skimage.segmentation import watershed

from streamlit_plotly_events import plotly_events
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import hashlib


# Optional import of pyDM3reader (for reading DM3/DM4 files)
try:  # pragma: no cover
    import dm3_lib as pydm3
except Exception:  # pragma: no cover
    try:
        import DM3lib as pydm3
    except Exception:
        pydm3 = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DM3Image:
    data: np.ndarray
    nm_per_px: float


# ---------------------------------------------------------------------------
# DM3 helpers (pyDM3reader)
# ---------------------------------------------------------------------------

def _text(x: Any) -> Optional[str]:
    """Best-effort conversion to plain str (handles bytes, arrays, codepoints)."""
    if x is None:
        return None
    try:
        if hasattr(x, "dtype"):  # numpy scalar/array
            try:
                x = x.item()
            except Exception:
                try:
                    if x.size == 1:
                        x = x.reshape(-1)[0]
                except Exception:
                    pass
        if isinstance(x, (bytes, bytearray)):
            return x.decode(errors="ignore")
        if isinstance(x, (list, tuple)) and x and all(isinstance(t, (int, np.integer)) for t in x):
            try:
                return bytes(x).decode(errors="ignore")
            except Exception:
                return "".join(chr(int(t)) for t in x if 0 <= int(t) < 0x110000)
        return str(x)
    except Exception:
        try:
            return str(x)
        except Exception:
            return None


def _nm_per_px_from_pydm3_pxsize(pxsize) -> float:
    """pxsize is commonly (value, unit). Convert to nm/px."""
    if not pxsize or (isinstance(pxsize, (list, tuple)) and len(pxsize) < 2):
        return float("nan")

    if isinstance(pxsize, dict):  # just in case
        val = pxsize.get("value") or pxsize.get("Value")
        unit = pxsize.get("unit") or pxsize.get("Unit")
    else:
        val = pxsize[0] if isinstance(pxsize, (list, tuple)) else getattr(pxsize, 0, None)
        unit = pxsize[1] if isinstance(pxsize, (list, tuple)) else getattr(pxsize, 1, None)

    try:
        v = float(val)
    except Exception:
        return float("nan")

    u = _text(unit) or ""
    u = u.strip().lower().replace("μ", "u").replace("µ", "u")
    u = u.replace("ångström", "angstrom").replace("å", "a")
    u = re.sub(r"\s+", "", u)

    # Direct
    if u in ("nm", "nanometer", "nanometers"):
        return v
    if u in ("um", "micrometer", "micrometre", "micron", "microns"):
        return v * 1000.0
    if u in ("a", "angstrom", "angstroms"):
        return v * 0.1
    if u in ("pm", "picometer", "picometre"):
        return v * 0.001
    if u == "mm":
        return v * 1_000_000.0
    if u == "cm":
        return v * 10_000_000.0
    if u in ("m", "meter", "metre"):
        return v * 1e9

    # 'm/pixel' etc. → assume meters
    if "m" in u and not u.startswith("nm"):
        return v * 1e9

    # Unknown: assume already nm
    return v


def try_read_dm3(file_bytes: bytes) -> DM3Image:
    """Read a DM3/DM4 file with pyDM3reader and extract image + nm/px."""
    if pydm3 is None:  # pragma: no cover
        raise RuntimeError("pyDM3reader is not installed. Install with: pip install pyDM3reader")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dm3")
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()

        dm3f = pydm3.DM3(tmp.name)  # parse
        data = np.asarray(dm3f.imagedata, dtype=np.float32)

        nm_per_px = _nm_per_px_from_pydm3_pxsize(getattr(dm3f, "pxsize", None))
        if not (isinstance(nm_per_px, (int, float, np.floating)) and np.isfinite(nm_per_px) and nm_per_px > 0):
            nm_per_px = float("nan")

        return DM3Image(data=data, nm_per_px=float(nm_per_px))
    finally:  # pragma: no cover
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------

def _as_float(x) -> Optional[float]:
    try:
        return float(x) if np.isfinite(float(x)) else None
    except Exception:
        return None


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
    return _as_float(min(c1_max, c2_max))


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
        return _as_float((left_mu + right_mu) / 2)
    sub_centers = centers[in_range]
    sub_counts = counts[in_range]
    return _as_float(sub_centers[np.argmin(sub_counts)])


# ---------------------------------------------------------------------------
# Display prep
# ---------------------------------------------------------------------------

def _prep_background_for_display(img: np.ndarray, method: str = "clahe") -> np.ndarray:
    """Create a high-contrast 8-bit background image for overlay only."""
    img = np.asarray(img, dtype=np.float32)
    vmin, vmax = np.percentile(img, (1, 99))
    img = np.clip(img, vmin, vmax)
    img = rescale_intensity(img, in_range=(vmin, vmax), out_range=(0.0, 1.0))
    if method == "clahe":
        img = equalize_adapthist(img, clip_limit=0.02)
    return (img * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Segmentation and measurement (UNCHANGED)
# ---------------------------------------------------------------------------

def segment_and_measure_shapes(
    data: np.ndarray,
    threshold: float,
    nm_per_px: float,
    shape_type: str,
    min_size_value: float,
    measurement_unit: str,
    min_area_px: int = 5,
    smooth_sigma: float = 0.8,
    h_max: int = 2,
    min_peak_distance: int = 5,
) -> Dict[str, np.ndarray]:
    """Segment particles and measure their dimensions.

    De-clumping improvements:
      - optional Gaussian smoothing prior to thresholding
      - distance-transform markers via peak_local_max with tunable min_distance
      - watershed from robust markers
    """

    # Smooth to suppress noise that causes bridges between dark particles
    if smooth_sigma > 0:
        data_s = gaussian_filter(data, sigma=_as_float(smooth_sigma))
    else:
        data_s = data

    # Binary image (dark particles)
    im_bi = data_s < threshold

    # Clean up small bridges/noise
    im_bi = binary_opening(im_bi, disk(1))
    im_bi = binary_closing(im_bi, disk(2))

    # Distance transform inside particles
    dist = distance_transform_edt(im_bi)

    # Robust markers: h-max suppression + local peaks
    # Enforce minimum spacing between seeds so close particles split
    coords = peak_local_max(
        dist,
        footprint=np.ones((3, 3)),
        min_distance=int(max(1, min_peak_distance)),
        labels=im_bi,
    )
    local_max = np.zeros_like(dist, dtype=bool)
    if coords.size > 0:
        local_max[tuple(coords.T)] = True
    markers = label(local_max)
    # Fallback if markers are too few (e.g., very small particles)
    if markers.max() < 1:
        # Seed everything non-zero
        markers = label(dist > max(0, h_max))

    # Watershed on the inverted distance
    labels_ws = watershed(-dist, markers=markers, mask=im_bi)

    # Remove border-touching & specks
    im_bi[labels_ws == 0] = 0
    im_bi = remove_small_objects(im_bi, min_size=max(3, min_area_px))
    labels_ws = label(im_bi)

    # Prepare measurement containers
    diameters_nm: List[float] = []
    hex_axes_nm: List[float] = []
    lengths_nm: List[float] = []
    widths_nm: List[float] = []

    # Scaling and exclusion zone
    if measurement_unit == "nm" and np.isfinite(nm_per_px) and nm_per_px > 0:
        scale_factor = nm_per_px
        exclusion_zone_px = 2 / nm_per_px
    else:
        scale_factor = 1.0
        exclusion_zone_px = 0.0
    img_h, img_w = data.shape

    # High-contrast background for readability
    bg = _prep_background_for_display(data, method="clahe")

    fig, ax = plt.subplots()
    ax.imshow(bg, cmap="gray")
    ax.axis("off")

    # Overlay style (with white stroke to pop on dark backgrounds)
    outline = [pe.Stroke(linewidth=1.0, foreground="white"), pe.Normal()]

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
        area = _as_float(p.area)
        perim = _as_float(getattr(p, "perimeter", 0.0)) or 0.0
        circ = 4 * np.pi * area / (perim ** 2) if perim > 0 else 0.0
        aspect = maj / minr_axis if minr_axis > 0 else 0.0
        extent = area / ((maxr - minr) * (maxc - minc)) if (maxr - minr) * (maxc - minc) > 0 else 0.0
        solidity = getattr(p, "solidity", 0.0)
        cy, cx = p.centroid  # (row, col)

        if shape_type == "Sphere":
            diam_px = (maj + minr_axis) / 2
            d_val = diam_px * scale_factor
            if d_val < min_size_value:
                continue
            diameters_nm.append(d_val)
            circ_patch = plt.Circle((cx, cy), diam_px / 2, fill=False, color="#e41a1c", linewidth=0.9)
            circ_patch.set_path_effects(outline)
            ax.add_patch(circ_patch)

        else:  # Hexagon or Cube (rectangular)
            if 1.2 < aspect < 1.8 and solidity > 0.8:
                # Rectangle-like
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
                    color="#377eb8",
                    linewidth=0.9,
                )
                rect.set_path_effects(outline)
                ax.add_patch(rect)
            elif solidity > 0.85 and extent > 0.6:
                # Potential hexagon (draw regular hex for preview)
                d = (maj + minr_axis) / 2
                d_val = d * scale_factor
                if d_val < min_size_value:
                    continue
                hex_axes_nm.append(d_val)
                radius = d / 2
                theta = np.linspace(0, 2 * np.pi, 7)
                x = cx + radius * np.cos(theta)
                y = cy + radius * np.sin(theta)
                line, = ax.plot(x, y, "-", linewidth=0.9, color="#ff7f00")
                line.set_path_effects(outline)
            else:
                # Other shapes (light preview)
                diam_px = (maj + minr_axis) / 2
                d_val = diam_px * scale_factor
                if d_val >= min_size_value:
                    circ_patch = plt.Circle((cx, cy), diam_px / 2, fill=False, color="#4daf4a", linewidth=0.9, alpha=0.8)
                    circ_patch.set_path_effects(outline)
                    ax.add_patch(circ_patch)

    buf_ann = io.BytesIO()
    fig.savefig(buf_ann, format="png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    fig_ws, ax_ws = plt.subplots()
    ax_ws.imshow(labels_ws > 0, cmap="gray")  # binary mask, no rainbow
    ax_ws.axis("off")
    buf_ws = io.BytesIO()
    fig_ws.savefig(buf_ws, format="png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig_ws)

    out: Dict[str, np.ndarray] = {
        "diameters_nm": np.array(diameters_nm, dtype=np.float32),
        "hex_axes_nm": np.array(hex_axes_nm, dtype=np.float32),
        "lengths_nm": np.array(lengths_nm, dtype=np.float32),
        "widths_nm": np.array(widths_nm, dtype=np.float32),
        "annotated_png": buf_ann.getvalue(),
        "watershed_png": buf_ws.getvalue(),
    }
    out["watershed_png_gray"] = buf_ws.getvalue()
    out["watershed_png"] = out["watershed_png_gray"]

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
    vals = values[(values >= xrange[0]) & (values <= xrange[1])]
    n = len(vals)
    counts, edges = np.histogram(vals, bins=nbins)
    centers = edges[:-1] + np.diff(edges) / 2
    fig = go.Figure(data=[go.Bar(x=centers, y=counts, name=title)])

    if n >= 3:
        mu, std = norm.fit(vals)
        xline = np.linspace(xrange[0], xrange[1], 300)
        yline = norm.pdf(xline, mu, std) * n * (centers[1] - centers[0])
        fig.add_trace(go.Scatter(x=xline, y=yline, mode="lines"))
    else:
        mu = _as_float("nan")

    fig.update_layout(title=title, xaxis_title=f"Size ({unit_label})", yaxis_title="Count")
    fig.update_xaxes(range=[float(xrange[0]), float(xrange[1])])

    return fig, _as_float(mu), n


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def run() -> None:  # pragma: no cover
    st.title("v2.1 TEM Particle Characterization (.dm3)")

    if pydm3 is None:
        st.error("`pyDM3reader` is not installed. Please run: `pip install pyDM3reader`")
        return

    st.caption(
        "Upload one or more `.dm3` images, choose a thresholding method and the particle shape, "
        "then view size distributions."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        files = st.file_uploader("Upload .dm3 file(s)", accept_multiple_files=True, type=["dm3", "dm4"])

    with col_right:
        method = st.selectbox("Threshold method", ["GMM", "K-means", "Manual"], index=0)
        shape_type = st.selectbox("Shape type", ["Sphere", "Hexagon", "Cube"], index=0)
        nbins_int = st.slider("Intensity histogram bins", 20, 200, 60, step=5)
        min_shape_size_input = st.number_input(
            "Minimum shape size (nm, or pixels if calibration missing)",
            min_value=0.0, max_value=10_000.0, value=4.0, step=0.5,
        )
        # Anti-clumping controls (unchanged)
        smooth_sigma = st.slider("Smoothing σ (px)", 0.0, 2.5, 0.8, step=0.1,
                                 help="Gaussian blur before thresholding; increases separation of nearby particles.")
        min_peak_distance = st.slider("Min seed distance (px)", 1, 20, 6, step=1,
                                      help="Minimum spacing between watershed markers from distance peaks.")
        h_max = st.slider("Marker fallback threshold", 0, 10, 2, step=1,
                          help="Used only if too few distance peaks are found.")

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

                if isinstance(nm_per_px, (int, float, np.floating)) and np.isfinite(nm_per_px) and nm_per_px > 0:
                    measurement_unit = "nm"
                    min_size_value = _as_float(min_shape_size_input)
                else:
                    measurement_unit = "px"
                    min_size_value = _as_float(min_shape_size_input)
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
                        st.session_state.manual_threshold = _as_float(clicked[-1]["x"])
                    st.session_state.manual_threshold = st.number_input(
                        "Manual threshold (intensity)",
                        value=_as_float(st.session_state.manual_threshold)
                        if st.session_state.manual_threshold is not None
                        else float(np.median(data)),
                        format="%.6f",
                        key=f"manual_thr_{i}",
                    )
                    chosen_threshold = _as_float(st.session_state.manual_threshold)
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
                    smooth_sigma=_as_float(smooth_sigma),
                    h_max=int(h_max),
                    min_peak_distance=int(min_peak_distance),
                )
                seg["name"] = f.name
                results.append(seg)

        # Display selected image results
        if results:
            names = [r["name"] for r in results]
            sel_name = st.selectbox("Select image to display", names)
            sel = next(r for r in results if r["name"] == sel_name)

            colA, colB = st.columns(2, gap="small")
            with colA:
                st.image(sel["annotated_png"], caption=f"Annotated segmentation", use_container_width=True)
            with colB:
                ws_img = sel.get("watershed_png_gray") or sel.get("watershed_png")
                st.image(ws_img, caption="Watershed labels (mask)", use_container_width=True)
            if missing_scale_files:
                missing_list = "\n".join(f"• {name}" for name in sorted(set(missing_scale_files)))
                st.warning(
                    "Pixel size metadata was not found for the following file(s). "
                    "All measurements (including the minimum shape size filter) are reported in pixels:\n"
                    f"{missing_list}"
                )

        # Combined histograms
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
                    float(all_d.min()), float(all_d.max()),
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
                    float(all_hex.min()), float(all_hex.max()),
                    (float(all_hex.min()), float(all_hex.max())),
                )
                range_len = st.slider(
                    f"Length range ({unit_label})",
                    float(all_len.min()), float(all_len.max()),
                    (float(all_len.min()), float(all_len.max())),
                )
                range_wid = st.slider(
                    f"Width range ({unit_label})",
                    float(all_wid.min()), float(all_wid.max()),
                    (float(all_wid.min()), float(all_wid.max())),
                )
                nbins = st.slider("Histogram bins", 10, 150, 50, step=5)

                fig_hex, mu_hex, n_hex = histogram_with_fit(all_hex, nbins, range_hex, f"Hexagon diagonal ({unit_label})", unit_label)
                fig_len, mu_len, n_len = histogram_with_fit(all_len, nbins, range_len, f"Length ({unit_label})", unit_label)
                fig_wid, mu_wid, n_wid = histogram_with_fit(all_wid, nbins, range_wid, f"Width ({unit_label})", unit_label)

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
                    float(all_len.min()), float(all_len.max()),
                    (float(all_len.min()), float(all_len.max())),
                )
                range_wid = st.slider(
                    f"Width range ({unit_label})",
                    float(all_wid.min()), float(all_wid.max()),
                    (float(all_wid.min()), float(all_wid.max())),
                )
                range_hgt = st.slider(
                    f"Height range ({unit_label})",
                    float(all_hgt.min()), float(all_hgt.max()),
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
        st.info("Upload one or more `.dm3` (or `.dm4`) files to begin.")

    # keep variable around to prevent Streamlit from re-running needlessly
    _ = (tuple((f.name, f.size, f.type) for f in files) if files else (),
         method, shape_type, nbins_int, float(min_shape_size_input))


if __name__ == "__main__":  # pragma: no cover
    run()
