# tools/spherical_tem.py
import math
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import tempfile

import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from skimage.segmentation import watershed
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects
from scipy.ndimage import distance_transform_edt

import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events  # pip install streamlit-plotly-events

# Try to import ncempy once, module-wide
try:
    from ncempy.io import dm as ncem_dm
except Exception:
    ncem_dm = None  # we'll warn in the UI if it's missing


# ---------- Types ----------
@dataclass
class DM3Image:
    data: np.ndarray
    nm_per_px: float


# ---------- Core utilities ----------
def try_read_dm3(file_bytes: bytes) -> DM3Image:
    if ncem_dm is None:
        raise RuntimeError("ncempy is not installed. Please install it (pip install ncempy).")

    # Write the uploaded bytes to a temp .dm3 file, then let ncempy open it.
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dm3")
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()

        # Use fileDM as a context manager to avoid the __del__/fid errors
        with ncem_dm.fileDM(tmp.name, verbose=False) as rdr:
            im = rdr.getDataset(0)           # dict with keys like 'data'
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
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass



def robust_percentile_cut(data: np.ndarray, p=99.5) -> np.ndarray:
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


def kmeans_threshold(data: np.ndarray, sample=200_000) -> float:
    flat = data.reshape(-1)
    if len(flat) > sample:
        idx = np.random.choice(len(flat), sample, replace=False)
        flat = flat[idx]
    km = KMeans(n_clusters=2, n_init="auto", random_state=42)
    labels = km.fit_predict(flat.reshape(-1, 1))
    c1_max = flat[labels == 0].max()
    c2_max = flat[labels == 1].max()
    return float(min(c1_max, c2_max))


def gmm_threshold(data: np.ndarray, nbins: Optional[int] = None, sample=200_000) -> float:
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


def segment_and_measure(
    data: np.ndarray,
    threshold: float,
    nm_per_px: float,
    min_area_px: int = 5,
    circ_lo: float = 0.05,
    circ_hi: float = 3.0,
) -> np.ndarray:
    im_bi = data < threshold
    im_bi = remove_small_objects(im_bi, min_size=min_area_px)
    dist = distance_transform_edt(im_bi)
    labels_ws = watershed(-dist, mask=im_bi)
    labels_ws = label(labels_ws > 0)

    props = regionprops(labels_ws)
    diam_px = []
    circ = []
    for p in props:
        maj = getattr(p, "major_axis_length", 0.0) or 0.0
        minr = getattr(p, "minor_axis_length", 0.0) or 0.0
        if maj == 0 and minr == 0:
            continue
        d = (maj + minr) / 2.0
        area = float(p.area)
        perim = float(getattr(p, "perimeter", 0.0)) or 0.0
        c = 4 * np.pi * area / (perim ** 2) if perim > 0 else 0.0
        diam_px.append(d)
        circ.append(c)

    if not diam_px:
        return np.array([], dtype=np.float32)

    diam_px = np.array(diam_px, dtype=np.float32)
    circ = np.array(circ, dtype=np.float32)
    keep = (circ >= circ_lo) & (circ <= circ_hi)
    diam_px = diam_px[keep]
    diam_nm = diam_px * nm_per_px
    return diam_nm[diam_nm >= 2.0]


def fig_histogram(values_nm: np.ndarray, nbins: int = 50, title: str = "Particle size distribution (nm)"):
    counts, edges = np.histogram(values_nm, bins=nbins)
    centers = edges[:-1] + np.diff(edges) / 2
    fig.update_layout(
    title=title,
    xaxis_title="Diameter (nm)",
    yaxis_title="Count",
    bargap=0.05,
    modebar_add=["toImage"]  # ensures the “Download as PNG” button shows
)

    return go.Figure(
        data=[go.Bar(x=centers, y=counts, name="Diameters (nm)")],
        layout=go.Layout(title=title, xaxis_title="Diameter (nm)", yaxis_title="Count", bargap=0.05),
    )


def fig_intensity_histogram(data: np.ndarray, nbins: Optional[int] = None, threshold: Optional[float] = None):
    centers, counts = histogram_for_intensity(data, nbins)
    fig = go.Figure(
        data=[go.Bar(x=centers, y=counts, name="Intensity")],
        layout=go.Layout(
            title="Intensity histogram (click to set threshold in Manual mode)",
            xaxis_title="Intensity",
            yaxis_title="Frequency",
            bargap=0.03,
        ),
    )
    fig.update_layout(
    title=title,
    xaxis_title="Diameter (nm)",
    yaxis_title="Count",
    bargap=0.05,
    modebar_add=["toImage"]  # ensures the “Download as PNG” button shows
)

    if threshold is not None:
        fig.add_shape(
            type="line",
            x0=threshold, x1=threshold, y0=0, y1=max(counts) if len(counts) else 1, line=dict(width=2, dash="dash")
        )
    return fig




# ---------- Streamlit entrypoint ----------
def run():
    st.title("TEM Spherical Particle Characterization (.dm3)")

    if ncem_dm is None:
        st.error("`ncempy` is not installed. Please run: `pip install ncempy`")
        return

    st.caption(
        "Upload one or more `.dm3` images, choose a thresholding method (Manual, K-means, or GMM), "
        "then view and export the particle diameter histogram."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        files = st.file_uploader("Upload .dm3 file(s)", accept_multiple_files=True, type=["dm3"])

    with col_right:
        method = st.selectbox(
            "Threshold method",
            ["GMM", "K-means", "Manual"],
            index=0,
            help="GMM: 2-component Gaussian mixture on intensity. K-means: 2 clusters of intensity. Manual: click the intensity histogram.",
        )
        nbins_int = st.slider("Intensity histogram bins (for Manual/GMM valley search)", 20, 200, 60, step=5)
        default_nm_per_px = st.number_input(
            "Fallback pixel size (nm per px) if absent in metadata",
            min_value=0.0001, max_value=1000.0, value=1.0, step=0.1,
            help="Used only when a file has no pixel size in its metadata.",
        )

    if "manual_threshold" not in st.session_state:
        st.session_state.manual_threshold = None

    diameters_all: List[float] = []

    if files:
        with st.spinner("Processing…"):
            for i, f in enumerate(files, start=1):
                dm3 = try_read_dm3(f.read())
                data = dm3.data
                nm_per_px = dm3.nm_per_px if np.isfinite(dm3.nm_per_px) else default_nm_per_px

                st.subheader(f"Image {i}")
                st.write(f"Shape: `{data.shape}`, nm/px: **{nm_per_px:.4g}**")

                # Threshold selection
                if method == "Manual":
                    fig_h = fig_intensity_histogram(data, nbins_int, st.session_state.manual_threshold)
                    clicked = plotly_events(fig_h, click_event=True, hover_event=False, select_event=False, key=f"click_{i}")
                    if clicked:
                        st.session_state.manual_threshold = float(clicked[-1]["x"])
                    st.session_state.manual_threshold = st.number_input(
                        "Manual threshold (intensity)",
                        value=float(st.session_state.manual_threshold)
                        if st.session_state.manual_threshold is not None else float(np.median(data)),
                        format="%.6f",
                        key=f"manual_thr_{i}",
                    )
                    chosen_threshold = float(st.session_state.manual_threshold)

                elif method == "K-means":
                    chosen_threshold = kmeans_threshold(data)
                    st.info(f"K-means threshold = **{chosen_threshold:.4f}**")
                    st.plotly_chart(fig_intensity_histogram(data, nbins_int, chosen_threshold), use_container_width=True)

                else:  # GMM
                    chosen_threshold = gmm_threshold(data, nbins_int)
                    st.info(f"GMM threshold = **{chosen_threshold:.4f}**")
                    st.plotly_chart(fig_intensity_histogram(data, nbins_int, chosen_threshold), use_container_width=True)

                # Segment + measure
                diam_nm = segment_and_measure(
                    data=data,
                    threshold=chosen_threshold,
                    nm_per_px=nm_per_px,
                    min_area_px=5,
                    circ_lo=0.05,
                    circ_hi=3.0,
                )

                if diam_nm.size == 0:
                    st.warning("No particles detected after filtering.")
                else:
                    diameters_all.extend(diam_nm.tolist())
                    st.success(f"Detected {len(diam_nm)} particles (≥ 2 nm).")

        st.markdown("---")
        st.subheader("Combined diameter histogram")

        if diameters_all:
            diam_arr = np.array(diameters_all, dtype=np.float32)

            with st.expander("Optional post-filter (reject mis-identified)"):
                enable_post = st.checkbox("Enable post-filter range", value=False)
                upper_slider_max = max(20.0, float(np.percentile(diam_arr, 99) * 1.5))
                min_nm, max_nm = st.slider("Diameter range (nm)", 0.0, upper_slider_max, (8.0, 13.0))
                if enable_post:
                    diam_arr = diam_arr[(diam_arr >= min_nm) & (diam_arr <= max_nm)]

            nbins_size = st.slider("Diameter histogram bins", 10, 150, 50, step=5)
            fig_sizes = fig_histogram(diam_arr, nbins=nbins_size)
            st.plotly_chart(fig_sizes, use_container_width=True)

            try:
                today = dt.datetime.now().strftime("%Y%m%d")
                st.download_button(
                    label="Download histogram as PNG",
                    data=png_bytes,
                    file_name=f"diameter_hist_{today}.png",
                    mime="image/png",
                )
            except Exception as e:
                st.error("To export PNGs, install `kaleido` (pip install -U kaleido). " + f"Export failed with: {e}")

            csv = "diameter_nm\n" + "\n".join(f"{x:.6f}" for x in diam_arr)
            st.download_button(
                label="Download diameters as CSV",
                data=csv.encode("utf-8"),
                file_name=f"diameters_nm_{dt.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No diameters to plot yet.")
    else:
        st.info("Upload one or more `.dm3` files to begin.")
