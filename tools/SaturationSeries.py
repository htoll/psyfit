import streamlit as st
import os
import io
from utils import integrate_sif, plot_brightness, plot_histogram
from tools.process_files import process_files
from matplotlib.colors import LogNorm
import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_histogram(df, min_val, max_val, num_bins):
    """
    Plots a histogram of brightness data, calculates its mean and std directly,
    and overlays a Gaussian curve based on those calculated parameters.
    Written by Hephaestus, a Gemini Gem tweaked by JFS
    """
    fig, ax = plt.subplots()
    
    # Filter the data based on the slider range
    brightness_data = df['brightness_fit']
    filtered_data = brightness_data[(brightness_data >= min_val) & (brightness_data <= max_val)]

    if filtered_data.empty:
        ax.text(0.5, 0.5, "No data in selected range.", ha='center')
        return fig, 0, 0

    # Step 1: Plot the histogram
    counts, bin_edges, _ = ax.hist(filtered_data, bins=num_bins, color='skyblue', edgecolor='black', alpha=0.7, label='Data')
    
    # Step 2: Calculate mean and standard deviation directly from the data
    mean_val = filtered_data.mean()
    std_val = filtered_data.std()

    # Step 3: Plot the Gaussian curve using the calculated parameters
    x_axis = np.linspace(bin_edges[0], bin_edges[-1], 100)
    
    # Scale the PDF to match the histogram's height
    bin_width = bin_edges[1] - bin_edges[0]
    scaling_factor = len(filtered_data) * bin_width
    gaussian_pdf = norm.pdf(x_axis, mean_val, std_val) * scaling_factor
    
    ax.plot(x_axis, gaussian_pdf, color='red', linestyle='--', linewidth=2, label=f'Gaussian (μ={mean_val:.1f}, σ={std_val:.1f})')

    ax.set_title("Brightness Distribution")
    ax.set_xlabel("Brightness (pps)")
    ax.set_ylabel("Counts")
    ax.legend()
    fig.tight_layout()

    return fig, mean_val, std_val
def build_brightness_heatmap(processed_data, weight_col="brightness_fit", shape_hint=None):
    """
    Aggregates brightness by pixel location across all processed files.
    - Tries to auto-detect coordinate columns from common names.
    - Returns a 2D numpy array heatmap with summed brightness.
    """
    # Candidate column names for x/y in pixels
    x_candidates = ["x", "x_px", "col", "column", "x_pix", "x_idx"]
    y_candidates = ["y", "y_px", "row", "line", "y_pix", "y_idx"]

    # Derive a shape from the first image if possible
    if shape_hint is not None:
        img_h, img_w = shape_hint
    else:
        first_img = None
        for v in processed_data.values():
            if "image" in v and isinstance(v["image"], np.ndarray):
                first_img = v["image"]
                break
        if first_img is None:
            raise ValueError("No image arrays found to infer heatmap shape.")
        img_h, img_w = first_img.shape

    heatmap = np.zeros((img_h, img_w), dtype=np.float64)

    for item in processed_data.values():
        df = item.get("df", None)
        if df is None or df.empty:
            continue

        # Find coordinate columns
        x_col = next((c for c in x_candidates if c in df.columns), None)
        y_col = next((c for c in y_candidates if c in df.columns), None)
        if x_col is None or y_col is None:
            # Skip this file if coords are missing
            continue

        if weight_col not in df.columns:
            # Skip if brightness column missing
            continue

        xs = df[x_col].to_numpy()
        ys = df[y_col].to_numpy()
        ws = df[weight_col].to_numpy()

        # Round to nearest pixel and clamp into image bounds
        xi = np.clip(np.rint(xs).astype(int), 0, img_w - 1)
        yi = np.clip(np.rint(ys).astype(int), 0, img_h - 1)

        # Accumulate brightness at pixel locations
        np.add.at(heatmap, (yi, xi), ws)

    return heatmap 
def plot_brightness_vs_current(df):
    """
    Calculates the mean and standard deviation of all particle brightness values
    for each current. Plots the mean vs. current with error bars showing the
    standard deviation of the particle distribution.
    Written by Hephaestus, a Gemini Gem tweaked by JFS
    """
    # Check if the initial dataframe is valid.
    if df is None or df.empty or 'filename' not in df.columns or 'brightness_fit' not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data available to plot.", ha='center', va='center')
        return fig

    df_copy = df.copy()

    # Step 1: Extract the current from the filename for every particle.
    df_copy['current'] = df_copy['filename'].str.extract(r'^(\d+)_').astype(int)

    # Step 2: Group all particles directly by current.
    # This calculates the mean and std dev from the entire population of
    # particles at each current value.
    agg_data = df_copy.groupby('current')['brightness_fit'].agg(['mean', 'std']).reset_index()
    agg_data = agg_data.sort_values('current')
    
    # If a group has only one particle, its std dev will be NaN. Fill with 0.
    agg_data['std'] = agg_data['std'].fillna(0)

    # Step 3: Create the plot.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        agg_data['current'],
        agg_data['mean'],      # This is the mean of all particles.
        yerr=agg_data['std'],  # This is the std dev of all particles.
        fmt='o-',
        capsize=5,
        ecolor='red',
        markerfacecolor='blue',
        markeredgecolor='blue'
    )

    ax.set_yscale('log')
    ax.set_xlabel("Current (mA)")
    ax.set_ylabel("Mean Particle Brightness (pps)")
    ax.set_title("Mean Particle Brightness vs. Current")
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    fig.tight_layout()

    return fig
@st.cache_data
def plot_quadrant_histograms_for_max_current(_uploaded_files, threshold, signal):
    """
    Processes files for all 4 quadrants, finds the max current, and
    plots a 2x2 grid of brightness histograms for that current.
    Written by Hephaestus, a Gemini Gem tweaked by JFS
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.flatten()  # Flatten the 2x2 array for easy iteration
    all_dfs = []
    
    # Process data for each quadrant to build a complete dataframe
    for i in range(1, 5):
        quadrant = str(i)
        # We only need the dataframe part of the output
        _, df_quad = process_files(list(_uploaded_files), quadrant, threshold=threshold, signal=signal)
        if df_quad is not None and not df_quad.empty:
            df_quad['quadrant'] = quadrant
            all_dfs.append(df_quad)
            
    if not all_dfs:
        fig.text(0.5, 0.5, "No data found in any quadrant.", ha='center')
        return fig

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['current'] = combined_df['filename'].str.extract(r'^(\d+)_').astype(int)
    max_current = combined_df['current'].max()

    fig.suptitle(f"Brightness Histograms for Max Current: {max_current} mA", fontsize=16)

    for i in range(4):
        ax = axes[i]
        quadrant = str(i + 1)
        
        # Filter data for the current quadrant and the max current
        quad_data = combined_df[(combined_df['quadrant'] == quadrant) & (combined_df['current'] == max_current)]
        
        if not quad_data.empty:
            brightness_data = quad_data['brightness_fit']
            ax.hist(brightness_data, bins=50, color='skyblue', edgecolor='black')
            ax.set_title(f"Quadrant {quadrant}")
            ax.set_xlabel("Brightness (pps)")
            ax.set_ylabel("Counts")
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        else:
            ax.text(0.5, 0.5, "No data", ha='center')
            ax.set_title(f"Quadrant {quadrant}")

    return fig

# --- Keep your build_brightness_heatmap function here ---
# --- Add the new plot_brightness_vs_current function here ---
def run():
    col1, col2 = st.columns([1, 2])
    
    @st.cache_data
    def df_to_csv_bytes(df):
        return df.to_csv(index=False).encode("utf-8")

    with col1:
        st.header("Analyze SIF Files")
        
        uploaded_files = st.file_uploader(
            "Upload .sif files (e.g., 100_1.sif, 120_1.sif)", 
            type=["sif"], 
            accept_multiple_files=True
        )
        
        threshold = st.number_input("Threshold", min_value=0, value=2, help='''
        Stringency of fit, higher value is more selective:  
        -UCNP signal sets absolute peak cut off  
        -Dye signal sets sensitivity of blob detection
        ''')

        # New section for threshold overrides
        with st.expander("Override Thresholds (Optional)"):
            override_text = st.text_area(
                "Enter overrides, one per line:",
                placeholder="100_1.sif: 2.5\n120_2.sif: 3.0",
                height=100
            )

        diagram = """ Splits sif into quadrants (256x256 px):  
        ┌─┬─┐  
        │ 1 │ 2 │  
        ├─┼─┤  
        │ 3 │ 4 │  
        └─┴─┘
        """
        region = st.selectbox("Region (for individual analysis)", options=["1", "2", "3", "4", "all"], help=diagram)

        signal = st.selectbox("Signal", options=["UCNP", "dye"], help='''Changes detection method:  
                                - UCNP for high SNR (sklearn peakfinder)  
                                - dye for low SNR (sklearn blob detection)''')
        cmap = st.selectbox("Colormap", options=["magma", 'viridis', 'plasma', 'hot', 'gray', 'hsv'])

    with col2:
        if "analyze_clicked" not in st.session_state:
            st.session_state.analyze_clicked = False

        if st.button("Analyze"):
            st.session_state.analyze_clicked = True
            
            # Parse the threshold overrides from the text area
            threshold_overrides = {}
            if override_text:
                for line in override_text.strip().split('\n'):
                    if ':' in line:
                        parts = line.split(':', 1)
                        filename = parts[0].strip()
                        try:
                            value = float(parts[1].strip())
                            threshold_overrides[filename] = value
                        except ValueError:
                            st.warning(f"Could not parse override for '{filename}'.")
            
            # You must update your process_files function to accept this new argument
            try:
                processed_data, _ = process_files(
                    uploaded_files, 
                    region, 
                    threshold=threshold, 
                    signal=signal,
                    threshold_overrides=threshold_overrides # Pass overrides to your function
                )

                # FIX for KeyError: Ensure each dataframe has a 'filename' column
                all_dfs_corrected = []
                for filename, data in processed_data.items():
                    df = data.get("df")
                    if df is not None and not df.empty:
                        df['filename'] = filename
                        all_dfs_corrected.append(df)
                
                # Re-create the combined_df from the corrected data
                if all_dfs_corrected:
                    combined_df = pd.concat(all_dfs_corrected, ignore_index=True)
                else:
                    combined_df = pd.DataFrame()

                st.session_state.processed_data = processed_data
                st.session_state.combined_df = combined_df

            except Exception as e:
                st.error(f"Error processing files: {e}")
                st.session_state.analyze_clicked = False

        if 'processed_data' in st.session_state:
            processed_data = st.session_state.processed_data
            combined_df = st.session_state.combined_df

            # The rest of your UI logic for tabs remains here...
            # This part is unchanged from the previous version.
            tab_analysis, tab_current, tab_max_current = st.tabs(["Image Analysis", "Current Dependency", "Max Current Analysis"])

            with tab_analysis:
                # ... content of tab_analysis
                pass # Placeholder for your existing code

            with tab_current:
                # ... content of tab_current
                pass # Placeholder for your existing code

            with tab_max_current:
                # ... content of tab_max_current
                pass # Placeholder for your existing code
