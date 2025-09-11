import streamlit as st
import os
import io
from utilsJFS import plot_histogram
from utils import integrate_sif, plot_brightness
from tools.process_files import process_files
from matplotlib.colors import LogNorm
import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd
import matplotlib.pyplot as plt

@st.cache_data
def plot_quadrant_dependencies(_uploaded_files, threshold, signal, threshold_overrides):
    """
    Processes data for each of the 4 quadrants and overlays their brightness vs.
    current dependency curves on a single plot.
    Written by Hephaestus, a Gemini Gem tweaked by JFS
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['Blue', 'Green', 'Red', 'Black']
    quadrants = ["1", "2", "3", "4"]

    for i, quadrant in enumerate(quadrants):
        # Process files for the current quadrant
        processed_data, _ = process_files(
            list(_uploaded_files),
            quadrant,
            threshold=threshold,
            signal=signal,
            threshold_overrides=threshold_overrides
        )
        
        # Rebuild the combined_df correctly to include the 'filename'
        all_dfs_corrected = []
        for filename, data in processed_data.items():
            df = data.get("df")
            if df is not None and not df.empty:
                df['filename'] = filename
                all_dfs_corrected.append(df)
        
        if not all_dfs_corrected:
            continue # Skip to the next quadrant if no data was found

        quadrant_df = pd.concat(all_dfs_corrected, ignore_index=True)
        
        # --- Perform the brightness vs. current calculation ---
        image_means = quadrant_df.groupby('filename')['brightness_fit'].mean().reset_index()
        image_means.rename(columns={'brightness_fit': 'mean_brightness'}, inplace=True)
        image_means['current'] = image_means['filename'].str.extract(r'^(\d+)_').astype(int)
        
        agg_data = image_means.groupby('current')['mean_brightness'].agg(['mean', 'std']).reset_index()
        agg_data = agg_data.sort_values('current')
        agg_data['std'] = agg_data['std'].fillna(0)
        # --- Calculation End ---

        # Plot this quadrant's data on the shared axes
        ax.errorbar(
            agg_data['current'],
            agg_data['mean'],
            yerr=agg_data['std'],
            fmt='o-',
            capsize=5,
            color=colors[i],
            label=f'Quadrant {quadrant}'
        )

    ax.set_yscale('log')
    ax.set_xlabel("Current (mA)")
    ax.set_ylabel("Mean of Image Means (pps)")
    ax.set_title("Quadrant Comparison of Brightness vs. Current")
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    return fig
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
    Calculates mean brightness per image, then aggregates these means by current.
    Plots the mean of image means vs. current, with error bars showing
    the standard deviation of the image means for each current.
    Written by Hephaestus, a Gemini Gem tweaked by JFS
    """
    if df is None or df.empty or 'brightness_fit' not in df.columns or 'filename' not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "filename error", ha='center', va='center')
        return fig

    # Step 1: Calculate the mean brightness for each individual image (FOV).
    image_means = df.groupby('filename')['brightness_fit'].mean().reset_index()
    image_means.rename(columns={'brightness_fit': 'mean_brightness'}, inplace=True)

    # Step 2: Extract the current from the filename in our new dataframe of means.
    image_means['current'] = image_means['filename'].str.extract(r'^(\d+)_').astype(int)

    # Step 3: Group the image means by current and calculate the final aggregate statistics.
    # The result is the mean of image means and the standard deviation of image means.
    agg_data = image_means.groupby('current')['mean_brightness'].agg(['mean', 'std']).reset_index()
    agg_data = agg_data.sort_values('current')
    
    # If a current has only one FOV, its std dev will be NaN. Set it to 0.
    agg_data['std'] = agg_data['std'].fillna(0)

    # Step 4: Create the plot (this part remains the same).
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        agg_data['current'],
        agg_data['mean'],  # This is the mean of the image means
        yerr=agg_data['std'],    # This is the std dev of the image means
        fmt='o-',
        capsize=5,
        ecolor='red',
        markerfacecolor='blue',
        markeredgecolor='blue'
    )

    ax.set_yscale('log')
    ax.set_xlabel("Current (mA)")
    ax.set_ylabel("Mean of Image Means (pps)")
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

    # Process data for each quadrant. We will capture the detailed dictionary
    # (processed_data_quad) to retain the filename for each particle.
    for i in range(1, 5):
        quadrant = str(i)
        
        # We use the first return value from process_files, which is a
        # dictionary mapping filenames to their data.
        processed_data_quad, _ = process_files(list(_uploaded_files), quadrant, threshold=threshold, signal=signal)

        # Iterate through each file's results for the current quadrant
        for filename, data in processed_data_quad.items():
            df = data.get("df")
            if df is not None and not df.empty:
                # Create a copy and add the filename and quadrant as new columns
                df_with_meta = df.copy()
                df_with_meta['filename'] = filename
                df_with_meta['quadrant'] = quadrant
                all_dfs.append(df_with_meta)
                
    if not all_dfs:
        fig.text(0.5, 0.5, "No data found in any quadrant.", ha='center')
        return fig

    # This new DataFrame now contains the 'filename' and 'quadrant' columns
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # This line will now execute successfully
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
        
        threshold = st.number_input("Threshold", min_value=0, value=2)
        
        with st.expander("Override Thresholds (Optional)"):
            override_text = st.text_area(
                "Enter overrides, one per line:",
                placeholder="100_1.sif: 2.5\n120_2.sif: 3.0",
                height=100
            )
            
        diagram = "..." # Your diagram string
        region = st.selectbox("Region (for individual analysis)", options=["1", "2", "3", "4", "all"], help=diagram)
        signal = st.selectbox("Signal", options=["UCNP", "dye"])
        cmap = st.selectbox("Colormap", options=["magma", 'viridis', 'plasma', 'hot', 'gray', 'hsv'])

    with col2:
        if "analyze_clicked" not in st.session_state:
            st.session_state.analyze_clicked = False

        if st.button("Analyze"):
            st.session_state.analyze_clicked = True
            
            threshold_overrides = {}
            if override_text:
                for line in override_text.strip().split('\n'):
                    if ':' in line:
                        parts = line.split(':', 1)
                        try:
                            threshold_overrides[parts[0].strip()] = float(parts[1].strip())
                        except (ValueError, IndexError):
                            st.warning(f"Could not parse override: {line}")
            st.session_state['threshold_overrides'] = threshold_overrides

            try:
                processed_data, _ = process_files(
                    uploaded_files, 
                    region, 
                    threshold=threshold, 
                    signal=signal,
                    threshold_overrides=threshold_overrides
                )
                
                all_dfs_corrected = []
                for filename, data in processed_data.items():
                    df = data.get("df")
                    if df is not None and not df.empty:
                        df['filename'] = filename
                        all_dfs_corrected.append(df)
                
                combined_df = pd.concat(all_dfs_corrected, ignore_index=True) if all_dfs_corrected else pd.DataFrame()

                st.session_state.processed_data = processed_data
                st.session_state.combined_df = combined_df

            except Exception as e:
                st.error(f"Error processing files: {e}")
                st.session_state.analyze_clicked = False

        if 'processed_data' in st.session_state:
            processed_data = st.session_state.processed_data
            combined_df = st.session_state.combined_df
            threshold_overrides = st.session_state.get('threshold_overrides', {})

            # Added "Quadrant Comparison" tab
            tab1, tab2, tab3, tab4 = st.tabs(["Image Analysis", "Current Dependency", "Quadrant Comparison", "Max Current Analysis"])

            with tab1:
                # Your existing code for Image Analysis tab
                pass

            with tab2:
                # Your existing code for Current Dependency tab
                pass
            
            # New tab to display the quadrant overlay plot
            with tab3:
                st.markdown("### Quadrant Comparison: Brightness vs. Current")
                fig_quad_compare = plot_quadrant_dependencies(
                    tuple(uploaded_files), 
                    threshold, 
                    signal, 
                    threshold_overrides
                )
                st.pyplot(fig_quad_compare)
                
                svg_buffer = io.StringIO()
                fig_quad_compare.savefig(svg_buffer, format='svg')
                st.download_button("Download Plot (SVG)", svg_buffer.getvalue(), "quadrant_comparison.svg")

            with tab4:
                # Your existing code for Max Current Analysis tab
                pass
