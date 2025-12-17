import streamlit as st
import numpy as np
import pandas as pd
import io
import os
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
import re

from utils import plot_brightness, plot_histogram

def read_dat_image(file_buffer):
    """
    Parses a confocal .dat file.
    Handles lines with '' prefixes and comma/whitespace delimiters.
    """
    try:
        file_buffer.seek(0)
        lines = file_buffer.readlines()
        data_rows = []
        
        for line in lines:
            # Decode if bytes, strip whitespace
            s = line.decode('utf-8') if isinstance(line, bytes) else line
            s = s.strip()
            
            if not s:
                continue
            
            # Remove prefix using a raw string for the regex
            # r'' ensures backslashes are treated literally
            clean_line = re.sub(r'\\', '', s)
            
            # Split by comma or whitespace and convert to floats
            try:
                row = [float(x) for x in re.split(r'[,\s]+', clean_line) if x]
                if row:
                    data_rows.append(row)
            except ValueError:
                continue
        
        if not data_rows:
            return None
            
        data = np.array(data_rows)
        # Check validity
        if data.ndim == 2 and data.size > 0:
            return data
            
        return None
    except Exception as e:
        st.warning(f"Error parsing file: {e}")
        return None

def fit_gaussian_2d(sub_img, x0_guess, y0_guess, amp_guess, offset_guess, pix_size_um):
    """
    Performs a 2D Gaussian fit on a sub-image.
    Note: Fitting is done in microns (um) to maintain stability with the provided bounds.
    """
    h, w = sub_img.shape
    y_idx, x_idx = np.indices((h, w))
    
    # Scale indices to physical units (microns)
    x_flat = x_idx.ravel() * pix_size_um
    y_flat = y_idx.ravel() * pix_size_um
    z_flat = sub_img.ravel()

    # Initial guess vector: [Amplitude, x0_um, sigma_x_um, y0_um, sigma_y_um, offset]
    # We guess sigma ~ 150nm (0.15um)
    p0 = [
        amp_guess, 
        x0_guess * pix_size_um, 
        0.15, 
        y0_guess * pix_size_um, 
        0.15, 
        offset_guess
    ]

    # Bounds
    # Amp: 0 to 2x max
    # Pos: +/- 2 pixels from guess
    # Sigma: 0.05um to 0.5um (50nm to 500nm)
    lb = [0, (x0_guess - 2)*pix_size_um, 0.05, (y0_guess - 2)*pix_size_um, 0.05, 0]
    ub = [amp_guess * 2.0, (x0_guess + 2)*pix_size_um, 0.5, (y0_guess + 2)*pix_size_um, 0.5, np.max(sub_img) + 1e-6]

    def residuals(params, x, y, z):
        A, x0, sx, y0, sy, offset = params
        model = A * np.exp(-((x - x0)**2 / (2 * sx**2) + (y - y0)**2 / (2 * sy**2))) + offset
        return model - z

    try:
        res = least_squares(residuals, p0, args=(x_flat, y_flat, z_flat), bounds=(lb, ub))
        return res.x # [A, x0, sx, y0, sy, offset]
    except Exception:
        return None

def integrate_dat(
    image_data, 
    dwell_val, 
    line_acc, 
    filename,
    threshold_std=5, 
    pix_size_um=0.1, 
    min_fit_r2=0.85
):
    """
    Analyzes a generic confocal image array.
    """
    # 1. Peak Detection
    smoothed = gaussian_filter(image_data, sigma=1)
    bg_mean = np.mean(smoothed)
    bg_std = np.std(smoothed)
    abs_threshold = bg_mean + (threshold_std * bg_std)

    coords = peak_local_max(smoothed, min_distance=3, threshold_abs=abs_threshold)

    results = []
    r_fit = 6  # Radius for fitting window (pixels)

    for y_peak, x_peak in coords:
        # 2. Extract Subregion
        y_min = max(0, y_peak - r_fit)
        y_max = min(image_data.shape[0], y_peak + r_fit + 1)
        x_min = max(0, x_peak - r_fit)
        x_max = min(image_data.shape[1], x_peak + r_fit + 1)
        
        sub_img = image_data[y_min:y_max, x_min:x_max]
        
        if sub_img.shape[0] < 3 or sub_img.shape[1] < 3:
            continue

        local_y = y_peak - y_min
        local_x = x_peak - x_min
        
        amp_guess = float(sub_img.max() - sub_img.min())
        offset_guess = float(sub_img.min())

        # 3. Fit Gaussian
        popt = fit_gaussian_2d(
            sub_img, local_x, local_y, amp_guess, offset_guess, pix_size_um
        )
        
        if popt is None:
            continue

        amp_fit, x0_um, sx_um, y0_um, sy_um, offset_fit = popt
        
        # 4. Check Goodness of Fit (R^2)
        h_sub, w_sub = sub_img.shape
        yi, xi = np.indices((h_sub, w_sub))
        xf = xi.ravel() * pix_size_um
        yf = yi.ravel() * pix_size_um
        zf = sub_img.ravel()
        
        model = amp_fit * np.exp(-((xf - x0_um)**2 / (2 * sx_um**2) + (yf - y0_um)**2 / (2 * sy_um**2))) + offset_fit
        ss_res = np.sum((zf - model) ** 2)
        ss_tot = np.sum((zf - np.mean(zf)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        if r2 < min_fit_r2:
            continue

        # 5. Calculate Brightness
        # Formula: Amplitude / Dwell / Acc
        denom = (dwell_val * line_acc) if (dwell_val * line_acc) > 0 else 1.0
        calculated_brightness = amp_fit / denom

        # Store results (convert fit centers back to global pixels)
        global_x_pix = x_min + (x0_um / pix_size_um)
        global_y_pix = y_min + (y0_um / pix_size_um)

        results.append({
            'filename': filename,
            'x_pix': global_x_pix,
            'y_pix': global_y_pix,
            'sigx_fit': sx_um, # microns
            'sigy_fit': sy_um, # microns
            'amp_fit': amp_fit,
            'offset_fit': offset_fit,
            'brightness_integrated': calculated_brightness, 
            'r2': r2
        })

    return pd.DataFrame(results)

def run():
    st.header("Analyze Confocal .dat Files")
    st.markdown(r"""
    $Brightness = \frac{Amplitude}{Dwell \times Accumulation}$
    """)

    # --- Sidebar Controls ---
    with st.sidebar:
        st.subheader("Files")
        uploaded_files = st.file_uploader("Upload .dat files", type=["dat", "txt", "csv"], accept_multiple_files=True)
        
        st.markdown("---")
        st.subheader("Acquisition Settings")
        
        # --- Autodetect Logic ---
        autodetect = st.checkbox("Autodetect imaging conditions", value=False, 
                                 help="Attempt to parse pixel size, dwell time, and accumulation from the filename.")
        
        # Default tags
        tag_pix = "nmpx"
        tag_dwell = "msDwell"
        tag_acc = "lineaccum"
        
        if autodetect:
            st.caption("Enter the unique text suffixes used in your filenames:")
            c1, c2, c3 = st.columns(3)
            with c1:
                tag_pix = st.text_input("Px Tag", value="nmpx", help="e.g. for '75nmpx', enter 'nmpx'")
            with c2:
                tag_dwell = st.text_input("Dwell Tag", value="msDwell", help="e.g. for '1msDwell', enter 'msDwell'")
            with c3:
                tag_acc = st.text_input("Acc Tag", value="lineaccum", help="e.g. for '10lineaccum', enter 'lineaccum'")
            st.info("Note: Parsed dwell time is assumed to be **ms**.")
        
        # Global Defaults (used if autodetect is off or fails)
        pix_size_nm_def = st.number_input("Pixel Size (nm)", value=100.0, min_value=1.0, step=10.0, format="%.1f")
        dwell_us_def = st.number_input("Dwell Time (µs)", value=1000.0, min_value=0.1, step=10.0, format="%.1f")
        line_acc_def = st.number_input("Line Accumulation", value=10, min_value=1, step=1)
        
        st.markdown("---")
        st.subheader("Fitting Thresholds")
        threshold_std = st.slider("Detection Threshold (σ above mean)", 1.0, 10.0, 5.0)
        min_r2 = st.slider("Min R²", 0.0, 1.0, 0.85)
        
        st.markdown("---")
        cmap = st.selectbox("Colormap", ['hot',"magma", "viridis", "plasma", "inferno", "gray"], index=0)
        show_fits = st.checkbox("Show Fit Circles", value=True)
        log_scale = st.checkbox("Log Scale Image", value=False)

    # --- Main Processing ---
    if uploaded_files:
        all_results = []
        processed_images = {} # Store images for display

        # Process each file
        progress_bar = st.progress(0)
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # 1. Determine Parameters for this file
                current_pix_nm = pix_size_nm_def
                current_dwell_us = dwell_us_def
                current_acc = line_acc_def
                
                if autodetect:
                    # Regex looks for: (number) followed strictly by (tag)
                    # Parsing Pixel Size (e.g., 75nmpx)
                    m_pix = re.search(r"(\d+(?:\.\d+)?)" + re.escape(tag_pix), uploaded_file.name)
                    if m_pix:
                        current_pix_nm = float(m_pix.group(1))

                    # Parsing Dwell (e.g., 1msDwell -> assumed ms -> convert to us)
                    m_dwell = re.search(r"(\d+(?:\.\d+)?)" + re.escape(tag_dwell), uploaded_file.name)
                    if m_dwell:
                        current_dwell_us = float(m_dwell.group(1)) * 1000.0 

                    # Parsing Accumulation (e.g., 10lineaccum)
                    m_acc = re.search(r"(\d+)" + re.escape(tag_acc), uploaded_file.name)
                    if m_acc:
                        current_acc = int(m_acc.group(1))

                # 2. Parse Image
                image_data = read_dat_image(uploaded_file)
                if image_data is None:
                    st.warning(f"Skipping {uploaded_file.name}: Empty or invalid format.")
                    continue
                
                # 3. Prepare units for integration
                # Dwell: us -> seconds
                dwell_s = current_dwell_us / 1e6
                # Pixel: nm -> microns
                pix_size_um = current_pix_nm / 1000.0

                # 4. Analyze
                df_file = integrate_dat(
                    image_data, 
                    dwell_s,  
                    current_acc, 
                    uploaded_file.name,
                    threshold_std=threshold_std, 
                    pix_size_um=pix_size_um,
                    min_fit_r2=min_r2
                )
                
                if not df_file.empty:
                    all_results.append(df_file)
                
                # Store for visualization
                processed_images[uploaded_file.name] = {
                    "image": image_data,
                    "df": df_file,
                    "params": (current_pix_nm, current_dwell_us, current_acc)
                }
            
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))

        progress_bar.empty()

        # Combine results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # --- Layout ---
            col_viz, col_data = st.columns([2, 1])

            with col_viz:
                # File Selector for Display
                file_options = list(processed_images.keys())
                selected_file = st.selectbox("Select file to view:", file_options)
                
                if selected_file:
                    data = processed_images[selected_file]
                    img = data["image"]
                    df = data["df"]
                    # Retrieve params stored during the loop
                    p_nm, p_us, p_acc = data.get("params", (0,0,0))
                    
                    st.caption(f"Image: {selected_file}")
                    st.caption(f"Params: {p_nm}nm | {p_us}µs | {p_acc} acc | Spots: {len(df)}")
                    
                    # Recalculate um for plotting scale
                    plot_pix_um = p_nm / 1000.0
                    
                    fig = plot_brightness(
                        img, 
                        df, 
                        show_fits=show_fits, 
                        normalization=log_scale, 
                        pix_size_um=plot_pix_um, 
                        cmap=cmap, 
                        interactive=True
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col_data:
                st.subheader("Combined Analysis")
                if not combined_df.empty:
                    st.metric("Total Spots", len(combined_df))
                    st.metric("Mean Brightness", f"{combined_df['brightness_integrated'].mean():.0f} pps")
                    
                    fig_hist, mu, sigma = plot_histogram(
                        combined_df,
                        min_val=combined_df['brightness_integrated'].min(),
                        max_val=combined_df['brightness_integrated'].max(),
                        num_bins=30
                    )
                    st.pyplot(fig_hist, use_container_width=True)
                    
                    csv = combined_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download All Results (CSV)",
                        csv,
                        "combined_confocal_results.csv",
                        "text/csv"
                    )
        else:
            st.warning("No spots detected in any of the uploaded files.")

if __name__ == "__main__":
    run()