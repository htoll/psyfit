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
    Plots mean brightness vs. current with std dev error bars.
    Assumes df has 'filename' and 'brightness_fit' columns.
    Filename format is expected to be 'CURRENT_FOV.sif'.
    Written by Hephaestus, a Gemini Gem tweaked by JFS
    """
    if df is None or df.empty or 'filename' not in df.columns or 'brightness_fit' not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data available to plot.", ha='center', va='center')
        return fig

    # Use a regular expression to safely extract the current (integer part) from the filename
    df['current'] = df['filename'].str.extract(r'^(\d+)_').astype(int)

    # Group data by the extracted current and calculate the mean and standard deviation
    aggData = df.groupby('current')['brightness_fit'].agg(['mean', 'std']).reset_index()
    aggData = aggData.sort_values('current')
    
    # If a group has only one data point, its standard deviation will be NaN. Replace with 0.
    aggData['std'] = aggData['std'].fillna(0)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        aggData['current'],
        aggData['mean'],
        yerr=aggData['std'],
        fmt='o-',
        capsize=5,
        ecolor='red',
        markerfacecolor='blue',
        markeredgecolor='blue'
    )



# --- Keep your build_brightness_heatmap function here ---
# --- Add the new plot_brightness_vs_current function here ---

def run():
    col1, col2 = st.columns([1, 2])
    
    @st.cache_data
    def df_to_csv_bytes(df):
        return df.to_csv(index=False).encode("utf-8")

    with col1:
        st.header("Analyze SIF Files")
        
        # New UI for directory input instead of file uploader
        sif_directory = st.text_input("Path to SIF file directory", help="Enter the full path to the folder containing your .sif files.")
        
        threshold = st.number_input("Threshold", min_value=0, value=2, help='''
        Stringency of fit, higher value is more selective:  
        -UCNP signal sets absolute peak cut off  
        -Dye signal sets sensitivity of blob detection
        ''')
        diagram = """ Splits sif into quadrants (256x256 px):  
        ┌─┬─┐  
        │ 1 │ 2 │  
        ├─┼─┤  
        │ 3 │ 4 │  
        └─┴─┘
        """
        region = st.selectbox("Region", options=["1", "2", "3", "4", "all"], help=diagram)

        signal = st.selectbox("Signal", options=["UCNP", "dye"], help='''Changes detection method:  
                                     - UCNP for high SNR (sklearn peakfinder)  
                                     - dye for low SNR (sklearn blob detection)''')
        cmap = st.selectbox("Colormap", options=["magma", 'viridis', 'plasma', 'hot', 'gray', 'hsv'])

    with col2:
        if "analyze_clicked" not in st.session_state:
            st.session_state.analyze_clicked = False

        if st.button("Analyze"):
            st.session_state.analyze_clicked = True
            # Clear previous data if re-analyzing
            if 'processed_data' in st.session_state:
                del st.session_state.processed_data
            if 'combined_df' in st.session_state:
                del st.session_state.combined_df

        if st.session_state.analyze_clicked and sif_directory:
            if not os.path.isdir(sif_directory):
                st.error(f"Directory not found: {sif_directory}")
                st.session_state.analyze_clicked = False
                return

            # Find and load files matching the specified pattern
            sif_files_to_process = []
            file_pattern = re.compile(r"^\d+_\d+\.sif$")
            
            # Using st.cache_data to avoid reloading files on every rerun
            @st.cache_data
            def load_sif_files(directory):
                loaded_files = []
                filenames = sorted(os.listdir(directory), key=lambda x: int(x.split('_')[0]))
                for filename in filenames:
                    if file_pattern.match(filename):
                        file_path = os.path.join(directory, filename)
                        try:
                            with open(file_path, 'rb') as f:
                                file_bytes = io.BytesIO(f.read())
                                file_bytes.name = filename  # Mock UploadedFile object
                                loaded_files.append(file_bytes)
                        except IOError as e:
                            st.warning(f"Could not read file {filename}: {e}")
                return loaded_files

            sif_files_to_process = load_sif_files(sif_directory)

            if not sif_files_to_process:
                st.warning("No files matching the 'integer_FOV.sif' format found.")
                st.session_state.analyze_clicked = False
                return

            try:
                # Process all found files
                processed_data, combined_df = process_files(sif_files_to_process, region, threshold=threshold, signal=signal)
                st.session_state.processed_data = processed_data
                st.session_state.combined_df = combined_df

            except Exception as e:
                st.error(f"Error processing files: {e}")
                st.session_state.analyze_clicked = False

        # Display results if they exist in session state
        if 'processed_data' in st.session_state:
            processed_data = st.session_state.processed_data
            combined_df = st.session_state.combined_df

            # Create tabs for different plots
            tab1, tab2, tab3 = st.tabs(["Individual Image", "Brightness Histogram", "Current Dependency"])

            with tab1:
                file_options = list(processed_data.keys())
                selected_file_name = st.selectbox("Select SIF to display:", options=file_options)
                
                show_fits = st.checkbox("Show fits")
                normalization = st.checkbox("Log Image Scaling")
                normalization_to_use = LogNorm() if normalization else None

                if selected_file_name in processed_data:
                    data_to_plot = processed_data[selected_file_name]
                    fig_image = plot_brightness(
                        data_to_plot["image"], data_to_plot["df"],
                        show_fits=show_fits, normalization=normalization_to_use,
                        pix_size_um=0.1, cmap=cmap
                    )
                    st.pyplot(fig_image)
                    
                    # Add download buttons
                    svg_buffer = io.StringIO()
                    fig_image.savefig(svg_buffer, format='svg')
                    st.download_button("Download Image (SVG)", svg_buffer.getvalue(), f"{selected_file_name}.svg", "image/svg+xml")

            with tab2:
                if combined_df is not None and not combined_df.empty:
                    st.markdown("### Combined Brightness Histogram")
                    brightness_vals = combined_df['brightness_fit'].values
                    min_val, max_val = st.slider("Select brightness range (pps):", 
                                                 float(np.min(brightness_vals)), float(np.max(brightness_vals)), 
                                                 (float(np.min(brightness_vals)), float(np.max(brightness_vals))))
                    num_bins = st.number_input("# Bins:", value=50, key="hist_bins")
                    
                    fig_hist, _, _ = plot_histogram(combined_df, min_val=min_val, max_val=max_val, num_bins=num_bins)
                    st.pyplot(fig_hist)

                    svg_buffer_hist = io.StringIO()
                    fig_hist.savefig(svg_buffer_hist, format='svg')
                    st.download_button("Download Histogram (SVG)", svg_buffer_hist.getvalue(), "histogram.svg", "image/svg+xml")
                    
                    csv_bytes = df_to_csv_bytes(combined_df)
                    st.download_button("Download All Data (CSV)", csv_bytes, "combined_data.csv", "text/csv")
                else:
                    st.info("No data for histogram.")

            with tab3:
                st.markdown(f"### Mean Brightness vs. Current (Region: {region})")
                if combined_df is not None and not combined_df.empty:
                    fig_current = plot_brightness_vs_current(combined_df)
                    st.pyplot(fig_current)
                    
                    svg_buffer_current = io.StringIO()
                    fig_current.savefig(svg_buffer_current, format='svg')
                    st.download_button("Download Plot (SVG)", svg_buffer_current.getvalue(), "brightness_vs_current.svg", "image/svg+xml")
                else:
                    st.info("No data to plot current dependency.")
