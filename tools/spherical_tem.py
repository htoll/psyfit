import io
import math
import os
import re  # Added for pairing logic
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist  # Added for colocalization
from scipy.stats import norm
from skimage.measure import regionprops, label
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    disk,
    h_maxima,
)
from skimage.segmentation import watershed
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from matplotlib.colors import ListedColormap
import pandas as pd

try:
    import dm3_lib as pyDM3reader
except ImportError:
    pyDM3reader = None




# ---------------------------------------------------------------------------
# Pairing Logic (From Previous Request)
# ---------------------------------------------------------------------------

def _split_ucnp_dye(files: List[Any], ucnp_id="976", dye_id="638") -> Tuple[List[Any], List[Any]]:
    """Split uploaded files into UCNP vs Dye sets based on filename tokens."""
    u_tok = str(ucnp_id).lower().strip()
    d_tok = str(dye_id).lower().strip()
    ucnp, dye = [], []

    for f in files:
        name = f.name if hasattr(f, "name") else str(f)
        lname = name.lower()
        has_ucnp = u_tok in lname if u_tok else False
        has_dye = d_tok in lname if d_tok else False

        if has_ucnp and not has_dye:
            ucnp.append(f)
        elif has_dye and not has_ucnp:
            dye.append(f)
        elif has_ucnp and has_dye:
            st.warning(f"Filename matches both tokens — skipping: {name}")
        else:
            # If strictly neither, we can treat as generic or skip. 
            # For this specific workflow, we skip to avoid pairing errors.
            pass 
            
    return ucnp, dye

def _match_ucnp_dye_files(ucnps: List[Any], dyes: List[Any]) -> List[Tuple[Any, Any]]:
    """Robustly matches UCNP and Dye files by treating them as a single timeline."""
    def get_seq_index(file_obj) -> int:
        name = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        matches = re.findall(r'\d+', name)
        return int(matches[-1]) if matches else -1

    all_files = []
    for f in ucnps:
        all_files.append({'file': f, 'type': 'u', 'idx': get_seq_index(f)})
    for f in dyes:
        all_files.append({'file': f, 'type': 'd', 'idx': get_seq_index(f)})

    all_files.sort(key=lambda x: x['idx'])

    pairs: List[Tuple[Any, Any]] = []
    i = 0
    while i < len(all_files) - 1:
        current = all_files[i]
        next_f = all_files[i+1]
        idx_diff = abs(current['idx'] - next_f['idx'])
        
        if (current['type'] != next_f['type']) and (idx_diff <= 1):
            u_file = current['file'] if current['type'] == 'u' else next_f['file']
            d_file = next_f['file'] if next_f['type'] == 'd' else current['file']
            pairs.append((u_file, d_file))
            i += 2 
        else:
            i += 1
    return pairs


# ---------------------------------------------------------------------------
# Data structures & File Reading
# ---------------------------------------------------------------------------
@dataclass
class DM3Image:
    data: np.ndarray
    nm_per_px: float

def _find_dimension_tags(tags: dict) -> list:
    found = []
    if isinstance(tags, dict):
        for key, value in tags.items():
            if key == "Dimension" and isinstance(value, dict):
                found.append(value)
            elif isinstance(value, dict):
                found.extend(_find_dimension_tags(value))
            elif isinstance(value, list):
                for item in value:
                    found.extend(_find_dimension_tags(item))
    return found

def try_read_dm3(file_bytes: bytes) -> DM3Image:
    if pyDM3reader is None:
        raise RuntimeError("dm3_lib is not installed.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dm3")
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()
        dm3_file = pyDM3reader.DM3(tmp.name)
        data = np.array(dm3_file.imagedata, dtype=np.float32)
        pixel_size = 0
        pixel_unit = ""
        if hasattr(dm3_file, "pxsize"):
            pixel_size, pixel_unit = dm3_file.pxsize
        if pixel_size == 0 and hasattr(dm3_file, "tags"):
            dim_groups = _find_dimension_tags(dm3_file.tags)
            for group in dim_groups:
                if '0' in group and isinstance(group['0'], dict):
                    dim_data = group['0']
                    scale = dim_data.get('Scale', 0)
                    units = dim_data.get('Units', '')
                    if scale > 0:
                        pixel_size = scale
                        pixel_unit = units
                        break
        if pixel_size == 0 and hasattr(dm3_file, "tags"):
             root_tags = dm3_file.tags
             if 'ImageTags' in root_tags:
                 img_tags = root_tags['ImageTags']
                 if 'Pixel Size (um)' in img_tags:
                     pixel_size = img_tags['Pixel Size (um)']
                     pixel_unit = 'µm'

        nm_per_px = float("nan")
        if pixel_size > 0:
            pixel_unit = str(pixel_unit).strip().replace('\x00', '')
            if pixel_unit == "m":
                nm_per_px = pixel_size * 1e9
            elif pixel_unit == "nm":
                nm_per_px = pixel_size
            elif pixel_unit in ["µm", "um", "micron", "microns"]:
                nm_per_px = pixel_size * 1e3
            else:
                nm_per_px = pixel_size 

        return DM3Image(data=data, nm_per_px=nm_per_px)
    except Exception as e:
        return DM3Image(data=np.array([[]]), nm_per_px=float("nan"))
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Segmentation and measurement
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

def segment_and_measure_shapes(
    data: np.ndarray,
    threshold: float,
    nm_per_px: float,
    shape_type: str,
    min_size_value: float,
    measurement_unit: str,
    min_area_px: int = 5,
) -> Dict[str, any]:
    """Segment particles and measure dimensions. Now exposes Centroids."""

    im_bi = data < threshold
    im_bi = remove_small_holes(im_bi, area_threshold=min_area_px)
    im_bi = binary_closing(im_bi, disk(3))

    dist = distance_transform_edt(im_bi)
    hmax = h_maxima(dist, 2)
    markers = label(hmax)
    labels_ws = watershed(-dist, markers=markers, mask=im_bi)
    im_bi[labels_ws == 0] = 0
    im_bi = remove_small_objects(im_bi, min_size=min_area_px)
    labels_ws = label(im_bi)

    diameters_nm: List[float] = []
    hex_axes_nm: List[float] = []
    lengths_nm: List[float] = []
    widths_nm: List[float] = []
    
    # Store centroids as (x, y) for plotting
    centroids_px: List[Tuple[float, float]] = []

    if measurement_unit == "nm" and np.isfinite(nm_per_px) and nm_per_px > 0:
        scale_factor = nm_per_px
        exclusion_zone_px = 2 / nm_per_px
    else:
        scale_factor = 1.0
        exclusion_zone_px = 0.0
    img_h, img_w = data.shape

    fig = go.Figure()
    fig.add_trace(go.Image(z=data))

    fig_shapes = []

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
        # ext = getattr(p, "extent", 0.0)
        solidity = getattr(p, "solidity", 0.0)
        extent = float(p.area) / ((maxr - minr) * (maxc - minc)) if (maxr - minr) * (maxc - minc) > 0 else 0.0
        aspect = maj / minr_axis if minr_axis > 0 else 0.0
        cy, cx = p.centroid

        # Determine if valid based on size/shape
        is_valid = False
        
        if shape_type == "Sphere":
            diam_px = (maj + minr_axis) / 2
            d_val = diam_px * scale_factor
            if d_val >= min_size_value:
                diameters_nm.append(d_val)
                is_valid = True
                fig_shapes.append(
                    dict(type="circle", x0=cx-diam_px/2, y0=cy-diam_px/2, x1=cx+diam_px/2, y1=cy+diam_px/2, 
                         line=dict(color="rgba(255, 0, 0, 0.5)", width=2))
                )
        else:
            # Hex/Cube Logic
            is_hex = solidity > 0.85 and extent > 0.6
            is_rect = 1.2 < aspect < 1.8 and solidity > 0.8

            if is_rect:
                length_val = maj * scale_factor
                width_val = minr_axis * scale_factor
                if length_val >= min_size_value and width_val >= min_size_value:
                    lengths_nm.append(length_val)
                    widths_nm.append(width_val)
                    is_valid = True
                    fig_shapes.append(
                        dict(type="rect", x0=minc, y0=minr, x1=maxc, y1=maxr,
                             line=dict(color="rgba(255, 0, 0, 0.5)", width=2))
                    )
            elif is_hex:
                d = (maj + minr_axis) / 2
                d_val = d * scale_factor
                if d_val >= min_size_value:
                    hex_axes_nm.append(d_val)
                    is_valid = True
                    radius = d / 2
                    path = f"M {cx+radius} {cy} " + " ".join([f"L {cx+radius*np.cos(t)} {cy+radius*np.sin(t)}" for t in np.linspace(np.pi/3, 2*np.pi-np.pi/3, 5)]) + " Z"
                    fig_shapes.append(
                        dict(type="path", path=path, line=dict(color="rgba(255, 255, 0, 0.5)", width=2))
                    )
            else:
                # Fallback to circle
                diam_px = (maj + minr_axis) / 2
                d_val = diam_px * scale_factor
                if d_val >= min_size_value:
                    # Generic catch-all
                    fig_shapes.append(
                        dict(type="circle", x0=cx-diam_px/2, y0=cy-diam_px/2, x1=cx+diam_px/2, y1=cy+diam_px/2,
                             line=dict(color="rgba(0, 0, 255, 0.5)", width=2))
                    )
                    is_valid = True # Treat as valid for "reconstruction" visualization

        if is_valid:
            centroids_px.append((cx, cy))

    fig.update_layout(
        shapes=fig_shapes,
        margin=dict(l=0, r=0, b=0, t=0),
    )
    fig.update_xaxes(visible=False, range=[0, img_w])
    fig.update_yaxes(visible=False, range=[img_h, 0]) 

    # Watershed Fig
    rand_cmap = np.random.rand(labels_ws.max() + 1, 3)
    rand_cmap[0] = [1, 1, 1]
    fig_ws = go.Figure(data=go.Heatmap(
        z=labels_ws, colorscale='Rainbow', zmin=0, zmax=labels_ws.max(), showscale=False
    ))
    fig_ws.update_layout(width=img_w, height=img_h, margin=dict(l=0, r=0, b=0, t=0))
    fig_ws.update_xaxes(visible=False)
    fig_ws.update_yaxes(visible=False, autorange="reversed")

    out: Dict[str, any] = {
        "diameters_nm": np.array(diameters_nm, dtype=np.float32),
        "hex_axes_nm": np.array(hex_axes_nm, dtype=np.float32),
        "lengths_nm": np.array(lengths_nm, dtype=np.float32),
        "widths_nm": np.array(widths_nm, dtype=np.float32),
        "annotated_fig": fig,
        "watershed_fig": fig_ws,
        "centroids_px": np.array(centroids_px), # Added for reconstruction
        "img_shape": (img_h, img_w)
    }

    if shape_type == "Cube":
        out["heights_nm"] = np.array(widths_nm, dtype=np.float32)

    out["unit"] = measurement_unit
    out["nm_per_px"] = float(nm_per_px) if measurement_unit == "nm" else float("nan")

    return out

def get_common_prefix(names: List[str]) -> str:
    if not names: return "analysis"
    shortest = min(names, key=len)
    for i, char in enumerate(shortest):
        if any(name[i] != char for name in names):
            return shortest[:i].rstrip(" _-") or "analysis"
    return shortest.rstrip(" _-") or "analysis"


@st.cache_data
def analyze_image(file_bytes: bytes, filename: str, method: str, shape_type: str, min_shape_size_input: float, nbins_int: int, manual_threshold: float | None) -> dict:
    dm3 = try_read_dm3(file_bytes)
    data = dm3.data
    nm_per_px = dm3.nm_per_px

    if data.size == 0:
        return {"error": "Image data is empty."}

    measurement_unit = "nm" if (np.isfinite(nm_per_px) and nm_per_px > 0) else "px"
    if not np.isfinite(nm_per_px) or nm_per_px <= 0:
        nm_per_px = float("nan")

    if method == "Manual":
        chosen_threshold = manual_threshold if manual_threshold is not None else float(np.median(data))
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
    )
    seg["chosen_threshold"] = chosen_threshold
    seg["name"] = filename
    return seg

# ---------------------------------------------------------------------------
# Plotting: Histogram
# ---------------------------------------------------------------------------
def histogram_with_fit(values, nbins, xrange, title, unit_label):
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
# Main Streamlit App
# ---------------------------------------------------------------------------
def run() -> None:
    st.set_page_config(layout="wide")
    st.title("Paired Particle Analysis (976 vs 638)")

    if pyDM3reader is None:
        st.error("`dm3_lib` is not installed. Please run: `pip install dm3_lib`")
        return

    # Sidebar / Controls
    with st.sidebar:
        st.header("Settings")
        shape_type = st.selectbox("Shape type", ["Sphere", "Hexagon", "Cube"], index=0)
        nbins_int = st.slider("Intensity histogram bins", 20, 200, 60, step=5)
        min_shape_size_input = st.number_input(
            "Min shape size (nm or px)", value=4.0, step=0.5
        )
        coloc_tolerance = st.slider("Colocalization Tolerance (px)", 1.0, 50.0, 10.0, step=0.5, 
                                    help="Max distance between centroids to consider them colocalized.")
        
        st.divider()
        st.info("Files are matched by filename index. (e.g. image_07.dm3)")

    if "manual_threshold" not in st.session_state:
        st.session_state.manual_threshold = None

    files = st.file_uploader("Upload .dm3 files (both 976 and 638)", accept_multiple_files=True, type=["dm3"])

    if not files:
        st.info("Please upload files.")
        return

    # 1. Split and Pair
    ucnp_files, dye_files = _split_ucnp_dye(files)
    pairs = _match_ucnp_dye_files(ucnp_files, dye_files)

    st.write(f"Found {len(ucnp_files)} UCNP (976) files and {len(dye_files)} Dye (638) files.")
    st.write(f"**Identified {len(pairs)} matched pairs.**")

    # 2. Process Pairs
    all_results = []

    if pairs:
        st.divider()
        for i, (f_u, f_d) in enumerate(pairs, 1):
            
            # Use columns for processing status to save vertical space
            st.markdown(f"#### Pair {i}: `{f_u.name}` & `{f_d.name}`")
            
            # Analyze UCNP
            res_u = analyze_image(f_u.read(), f_u.name, "GMM", shape_type, min_shape_size_input, nbins_int, st.session_state.manual_threshold)
            f_u.seek(0) # Reset stream
            
            # Analyze Dye
            res_d = analyze_image(f_d.read(), f_d.name, "GMM", shape_type, min_shape_size_input, nbins_int, st.session_state.manual_threshold)
            f_d.seek(0) # Reset stream

            all_results.append(res_u)
            all_results.append(res_d)

            # --- VISUALIZATION COLUMNS ---
            col1, col2, col3 = st.columns(3)
            
            # COLUMN 1: UCNP (976)
            with col1:
                st.caption(f"976 (UCNP): {f_u.name}")
                if "annotated_fig" in res_u:
                    st.plotly_chart(res_u["annotated_fig"], use_container_width=True)
                else:
                    st.error("Analysis failed")

            # COLUMN 2: Dye (638)
            with col2:
                st.caption(f"638 (Dye): {f_d.name}")
                if "annotated_fig" in res_d:
                    st.plotly_chart(res_d["annotated_fig"], use_container_width=True)
                else:
                    st.error("Analysis failed")

            # COLUMN 3: Reconstruction
            with col3:
                st.caption("Reconstruction (Colocalization)")
                
                pts_u = res_u.get("centroids_px", np.array([]))
                pts_d = res_d.get("centroids_px", np.array([]))
                
                fig_recon = go.Figure()
                
                # Determine image bounds for the reconstruction plot
                h, w = res_u.get("img_shape", (1024, 1024))
                
                # Plot UCNP (Grey)
                if len(pts_u) > 0:
                    fig_recon.add_trace(go.Scatter(
                        x=pts_u[:, 0], y=pts_u[:, 1],
                        mode='markers',
                        marker=dict(color='gray', size=6, opacity=0.6),
                        name='976 (UCNP)'
                    ))
                
                # Plot Dye (Red)
                if len(pts_d) > 0:
                    fig_recon.add_trace(go.Scatter(
                        x=pts_d[:, 0], y=pts_d[:, 1],
                        mode='markers',
                        marker=dict(color='red', size=6, opacity=0.6),
                        name='638 (Dye)'
                    ))

                # Calculate Colocalization (X)
                coloc_count = 0
                if len(pts_u) > 0 and len(pts_d) > 0:
                    # Calculate distance matrix (Euclidean)
                    dists = cdist(pts_u, pts_d, metric='euclidean')
                    # Find indices where dist < tolerance
                    u_indices, d_indices = np.where(dists < coloc_tolerance)
                    
                    if len(u_indices) > 0:
                        # Get coords for colocalized spots (take avg position)
                        coloc_x = (pts_u[u_indices, 0] + pts_d[d_indices, 0]) / 2
                        coloc_y = (pts_u[u_indices, 1] + pts_d[d_indices, 1]) / 2
                        
                        fig_recon.add_trace(go.Scatter(
                            x=coloc_x, y=coloc_y,
                            mode='markers',
                            marker=dict(symbol='x', color='black', size=8, line=dict(width=2)),
                            name='Colocalized'
                        ))
                        coloc_count = len(u_indices)
                
                # Match image coordinate system (0,0 at top left)
                fig_recon.update_xaxes(range=[0, w], showgrid=False)
                fig_recon.update_yaxes(range=[h, 0], showgrid=False) # Inverted Y
                fig_recon.update_layout(
                    margin=dict(l=0, r=0, b=0, t=30),
                    height=300, # Matches generic image aspect roughly
                    title=f"Matches: {coloc_count}",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_recon, use_container_width=True)

            st.divider()

    # --- Aggregate Statistics (Optional, kept from original logic) ---
    if all_results:
        with st.expander("Aggregate Statistics"):
            # Reuse existing histogram logic for the aggregated data
            units_present = {r.get("unit", "nm") for r in all_results}
            if len(units_present) == 1:
                unit_label = list(units_present)[0]
                if shape_type == "Sphere":
                    all_d = np.concatenate([r["diameters_nm"] for r in all_results])
                    if all_d.size:
                        fig, mu, n = histogram_with_fit(all_d, 50, (all_d.min(), all_d.max()), f"All Diameters ({unit_label})", unit_label)
                        st.plotly_chart(fig)
            else:
                st.warning("Mixed units detected, cannot aggregate histograms.")

if __name__ == "__main__":
    run()