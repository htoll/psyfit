"""Interactive TEM FFT Analysis Tool - Compact Dashboard."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Optional import of ncempy
try:
    from ncempy.io import dm as ncem_dm
    from ncempy.io import emd as ncem_emd
except ImportError:
    ncem_dm = None
    ncem_emd = None


# --- DATA STRUCTURES ---
@dataclass
class TEMImage:
    data: np.ndarray
    nm_per_px: float
    filename: str


# --- CACHED FUNCTIONS ---

@st.cache_data(show_spinner=False)
def get_file_content(file_bytes: bytes, filename: str) -> TEMImage:
    """Reads file bytes and extracts metadata."""
    if ncem_dm is None:
        raise RuntimeError("ncempy is not installed. Please install it (pip install ncempy).")

    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        data = None
        nm_per_px = np.nan
        
        if suffix.lower() == '.dm3':
            with ncem_dm.fileDM(tmp_path, verbose=False) as rdr:
                im = rdr.getDataset(0)
                data = np.array(im["data"], dtype=np.float32)
                
                # Metadata Strategy
                if 'pixelSize' in im and len(im['pixelSize']) > 0:
                     val = im['pixelSize'][0]
                     if val < 1e-6: nm_per_px = val * 1e9
                     else: nm_per_px = val
                
                if np.isnan(nm_per_px):
                    md = rdr.allTags
                    candidates = [
                        ("ImageList.1.ImageData.Calibrations.Dimension.0.Scale", 1e9),
                        ("pixelSize.x", 1e9),
                        ("xscale", 1e9),
                        ("root.ImageList.1.ImageData.Calibrations.Dimension.0.Scale", 1)
                    ]
                    for key, factor in candidates:
                        try:
                            val = md
                            for k in key.split("."):
                                val = val[k]
                            if isinstance(val, (int, float)) and val > 0:
                                nm_per_px = float(val) * factor
                                break
                        except Exception:
                            continue

        elif suffix.lower() == '.emd':
            with ncem_emd.fileEMD(tmp_path, readonly=True) as f:
                for group in f.list_groups():
                    try:
                        ds = f.get_dataset(group)
                        if isinstance(ds, tuple) and len(ds) >= 1:
                            data = np.array(ds[0], dtype=np.float32)
                            break
                    except Exception:
                        continue
        
        if data is None:
            raise ValueError("Could not extract image data.")
            
        if np.isnan(nm_per_px) or nm_per_px == 0:
            nm_per_px = 1.0

        return TEMImage(data=data, nm_per_px=nm_per_px, filename=filename)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@st.cache_data(show_spinner=False)
def process_fft_image(roi_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes FFT and generates 'ImageJ Style' Display Image."""
    roi_clean = np.nan_to_num(roi_data)
    
    h, w = roi_clean.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    fft_res = np.fft.fft2(roi_clean * window)
    fft_shifted = np.fft.fftshift(fft_res)
    fft_mag = np.abs(fft_shifted)
    
    # ImageJ Style Display: Log + Gaussian Smooth
    fft_log = np.log10(fft_mag + 1)
    fft_display = gaussian_filter(fft_log, sigma=1.0)
    
    # Contrast Stretch
    cy, cx = h // 2, w // 2
    mask = np.ones_like(fft_display, dtype=bool)
    mask[cy-2:cy+3, cx-2:cx+3] = False
    stats_data = fft_display[mask]
    
    v_min = np.percentile(stats_data, 1) 
    v_max = np.percentile(stats_data, 99.9) 
    if v_max == v_min: v_max = v_min + 1e-6

    fft_norm = (fft_display - v_min) / (v_max - v_min)
    fft_norm = np.clip(fft_norm, 0, 1)
    
    cmap = plt.get_cmap('gray')
    fft_rgb = (cmap(fft_norm)[:, :, :3] * 255).astype(np.uint8)
    
    return fft_mag, fft_rgb


# --- MAIN APP ---
def run():
    st.set_page_config(layout="wide", page_title="TEM FFT Analysis")
    st.title("Estimating Lattice Spacing from TEM Images")
    
    if "last_file_id" not in st.session_state: st.session_state.last_file_id = None
    
    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload TEM Image", type=["dm3", "emd"])
        st.divider()
        manual_scale_container = st.container()
        st.divider()
        st.write("**Peak Detection**")
        peak_thresh = st.slider("Sensitivity", 0.01, 1.0, 0.1, 0.01)
        min_dist_px = st.slider("Min Spacing (px)", 1, 50, 10)

    if not uploaded_file:
        st.info("Please upload a .dm3 or .emd file to begin.")
        return

    # Reset Logic
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.last_file_id != file_id:
        st.session_state.last_file_id = file_id
        uploaded_file.seek(0)
    
    try:
        tem_img_raw = get_file_content(uploaded_file.getvalue(), uploaded_file.name)
        with manual_scale_container:
            val_to_show = float(tem_img_raw.nm_per_px)
            if val_to_show == 1.0:
                st.warning("⚠️ Enter pixel size:")
            actual_scale = st.number_input("Pixel Size (nm/px)", value=val_to_show, format="%.5f")
            tem_img = TEMImage(tem_img_raw.data, actual_scale, tem_img_raw.filename)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # --- LAYOUT PREP ---
    # Prepare Display Image
    p2, p98 = np.percentile(tem_img.data, (2, 98))
    img_norm = np.clip((tem_img.data - p2) / (p98 - p2), 0, 1) * 255
    img_rgb = np.stack((img_norm.astype(np.uint8),)*3, axis=-1)
    
    # FIXED WIDTHS prevent offset issues
    # A width of 350px fits 3 columns comfortably on a 1080p screen
    CANVAS_WIDTH = 350 
    CANVAS_HEIGHT = int(CANVAS_WIDTH * (img_rgb.shape[0] / img_rgb.shape[1]))
    
    # --- FIX: Resize image for display performance on Cloud ---
    # We keep the raw data in 'tem_img.data' for the math, 
    # but we downsample the visual background to save bandwidth.
    pil_image_full = Image.fromarray(img_rgb)
    pil_image_display = pil_image_full.resize((CANVAS_WIDTH, CANVAS_HEIGHT), resample=Image.Resampling.LANCZOS)

    # --- 3-COLUMN DASHBOARD ---
    c_left, c_mid, c_right = st.columns([1, 1, 1])

    # --- 1. LEFT: INPUT IMAGE ---
    with c_left:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.2)",
            stroke_width=2,
            stroke_color="#FFFFFF",
            background_image=pil_image_display, # <--- Pass the resized image here
            update_streamlit=True,
            height=CANVAS_HEIGHT,
            width=CANVAS_WIDTH,
            drawing_mode="rect",
            key="roi_canvas",
        )

    # --- 2. MIDDLE: FFT ---
    with c_mid:
        
        roi_data = None
        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            obj = canvas_result.json_data["objects"][-1]
            
            # Robust Coordinate Scaling
            scale_x = tem_img.data.shape[1] / CANVAS_WIDTH
            scale_y = tem_img.data.shape[0] / CANVAS_HEIGHT
            
            x, y = int(obj["left"] * scale_x), int(obj["top"] * scale_y)
            w, h = int(obj["width"] * scale_x), int(obj["height"] * scale_y)
            
            # Clip & Extract
            y_start, y_end = max(0, y), min(y+h, tem_img.data.shape[0])
            x_start, x_end = max(0, x), min(x+w, tem_img.data.shape[1])
            if y_end > y_start and x_end > x_start:
                roi_data = tem_img.data[y_start:y_end, x_start:x_end]

        if roi_data is not None:
            # Process
            fft_mag, fft_rgb = process_fft_image(roi_data)
            
            # Peak Finding
            cy, cx = fft_mag.shape[0]//2, fft_mag.shape[1]//2
            fft_search = fft_mag.copy()
            fft_search[cy-8:cy+8, cx-8:cx+8] = 0 # Mask DC
            
            coordinates = peak_local_max(
                fft_search, 
                min_distance=min_dist_px, 
                threshold_rel=peak_thresh,
                num_peaks=20
            )
            
            # Calculate Physics
            d_spacing = 0.0
            avg_dist_px = 0.0
            found_peaks = []
            
            if len(coordinates) > 0:
                distances = np.sqrt((coordinates[:, 0] - cy)**2 + (coordinates[:, 1] - cx)**2)
                sorted_idx = np.argsort(distances)
                sorted_coords = coordinates[sorted_idx]
                sorted_dists = distances[sorted_idx]
                
                # Filter Primary Ring
                if len(sorted_dists) > 0:
                    nearest = sorted_dists[0]
                    mask = sorted_dists < (nearest * 1.25)
                    found_peaks = sorted_coords[mask]
                    avg_dist_px = np.mean(sorted_dists[mask])
                    
                    roi_avg_px = (roi_data.shape[0] + roi_data.shape[1]) / 2
                    recip_step = 1.0 / (roi_avg_px * tem_img.nm_per_px)
                    if avg_dist_px > 0:
                        d_spacing = 1.0 / (avg_dist_px * recip_step)

            # Display FFT
            fig = px.imshow(fft_rgb)
            if len(found_peaks) > 0:
                fig.add_trace(go.Scatter(
                    x=found_peaks[:, 1], y=found_peaks[:, 0],
                    mode='markers',
                    marker=dict(color='lime', size=10, symbol='circle-open', line=dict(width=2)),
                    name='Spots'
                ))
            fig.add_trace(go.Scatter(x=[cx], y=[cy], mode='markers', marker=dict(color='red', symbol='x'), showlegend=False))

            # Compact Layout
            fig.update_layout(
                width=350, height=350, # Matches Canvas Width
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
                showlegend=False
            )
            st.plotly_chart(fig)

    # --- 3. RIGHT: RESULTS ---
    with c_right:
        
        if roi_data is None:
            st.info("Draw a box on the left image.")
        elif d_spacing > 0:
            st.success(f"**d = {d_spacing:.4f} nm**")
            st.metric("Detected Spots", f"{len(found_peaks)}")
            st.metric("Avg Distance", f"{avg_dist_px:.1f} px")
            st.caption(f"Scale: {tem_img.nm_per_px:.4f} nm/px")
            
        else:
            st.warning("No clear peaks found.")
            st.caption("Try lowering the sensitivity threshold or increasing the box size.")

if __name__ == "__main__":
    run()
