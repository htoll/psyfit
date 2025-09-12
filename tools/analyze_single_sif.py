import streamlit as st
import os
import io
from utils import integrate_sif, plot_brightness, plot_histogram
from tools.process_files import process_files
from matplotlib.colors import LogNorm
import numpy as np
from scipy.ndimage import gaussian_filter

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



def run():
    col1, col2 = st.columns([1, 2])
    @st.cache_data
    def df_to_csv_bytes(df):
        return df.to_csv(index=False).encode("utf-8")

    with col1:
        st.header("Analyze SIF Files")
        uploaded_files = st.file_uploader("Upload .sif file", type=["sif"], accept_multiple_files=True)
        threshold = st.number_input("Threshold", min_value=0, value=2, help = '''
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
        region = st.selectbox("Region", options=["1", "2", "3", "4", "all"], help = diagram)

        signal = st.selectbox("Signal", options=["UCNP", "dye"], help= '''Changes detection method:  
                                                                - UCNP for high SNR (sklearn peakfinder)  
                                                                - dye for low SNR (sklearn blob detection)''')
        cmap = st.selectbox("Colormap", options = ["magma", 'viridis', 'plasma', 'hot', 'gray', 'hsv'])


    with col2:
        if "analyze_clicked" not in st.session_state:
            st.session_state.analyze_clicked = False
    
        plot_col1, plot_col2 = st.columns(2)
    
        with plot_col1:
            show_fits = st.checkbox("Show fits")
            plot_brightness_histogram = True
            normalization = st.checkbox("Log Image Scaling")
    
            if st.button("Analyze"):
                st.session_state.analyze_clicked = True
    
        if st.session_state.analyze_clicked and uploaded_files:
            try:
                processed_data, combined_df = process_files(uploaded_files, region, threshold = threshold, signal=signal)
    
                if len(uploaded_files) > 1:
                    file_options = [f.name for f in uploaded_files]
                    selected_file_name = st.selectbox("Select sif to display:", options=file_options)
                else:
                    selected_file_name = uploaded_files[0].name
    
                if selected_file_name in processed_data:
                    data_to_plot = processed_data[selected_file_name]
                    df_selected = data_to_plot["df"]
                    image_data_cps = data_to_plot["image"]
    
                    normalization_to_use = LogNorm() if normalization else None
                    fig_image = plot_brightness(
                        image_data_cps,
                        df_selected,
                        show_fits=show_fits,
                        normalization=normalization_to_use,
                        pix_size_um=0.1,
                        cmap=cmap
                    )
    
                    with plot_col1:
                        st.pyplot(fig_image)
                        st.markdown("#### Zoom & Inspect")
                        zoom_mode = st.toggle("🔍 Magnifying glass", value=True,
                                              help="Turn on to box-zoom by click-and-drag. Turn off to pan.")
                        fig_interactive = plot_brightness(
                            image_data_cps,
                            df_selected if show_fits else None,
                            show_fits=show_fits,
                            normalization=normalization_to_use,
                            pix_size_um=0.1,
                            cmap=cmap,
                            interactive=True,
                        )
                        # Set drag mode based on toggle
                        fig_interactive.update_layout(dragmode="zoom" if zoom_mode else "pan")
                    
                        st.plotly_chart(
                            fig_interactive,
                            use_container_width=True,
                            config={
                                "displaylogo": False,
                                "modeBarButtonsToRemove": [
                                    "select2d", "lasso2d", "toggleSpikelines"
                                ],
                                # leave zoom, pan, autoscale, reset unchanged (they’ll show in the toolbar)
                            },
                        )
                        svg_buffer = io.StringIO()
                        fig_image.savefig(svg_buffer, format='svg')
                        svg_data = svg_buffer.getvalue()
                        svg_buffer.close()
                        st.download_button(
                            label="Download PSFs",
                            data=svg_data,
                            file_name=f"{selected_file_name}.svg",
                            mime="image/svg+xml"
                        )
                        if combined_df is not None and not combined_df.empty:
                            csv_bytes = df_to_csv_bytes(combined_df)
                            st.download_button(
                                label="Download as CSV",
                                data=csv_bytes,
                                file_name=f"{os.path.splitext(selected_file_name)[0]}_compiled.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No compiled data available to download yet.")
    
                    with plot_col2:
                        if plot_brightness_histogram and not combined_df.empty:
                            brightness_vals = combined_df['brightness_fit'].values
                            default_min_val = float(np.min(brightness_vals))
                            default_max_val = float(np.max(brightness_vals))
    
                            user_min_val_str = st.text_input("Min Brightness (pps)", value=f"{default_min_val:.2e}")
                            user_max_val_str = st.text_input("Max Brightness (pps)", value=f"{default_max_val:.2e}")
    
                            try:
                                user_min = float(user_min_val_str)
                                user_max = float(user_max_val_str)
                            except ValueError:
                                st.warning("Please enter valid numbers (you can use scientific notation like 1e6).")
                                return
    
                            num_bins = st.number_input("# Bins:", value=50)
    
                            if user_min < user_max:
                                fig_hist, _, _ = plot_histogram(
                                    combined_df,
                                    min_val=user_min,
                                    max_val=user_max,
                                    num_bins=num_bins
                                )
                                st.pyplot(fig_hist)
    
                                svg_buffer_hist = io.StringIO()
                                fig_hist.savefig(svg_buffer_hist, format='svg')
                                svg_data_hist = svg_buffer_hist.getvalue()
                                svg_buffer_hist.close()
    
                                st.download_button(
                                    label="Download histogram",
                                    data=svg_data_hist,
                                    file_name="combined_histogram.svg",
                                    mime="image/svg+xml"
                                )
                            else:
                                st.warning("Min greater than max.")
                else:
                    st.error(f"Data for file '{selected_file_name}' not found.")
    
            except Exception as e:
                st.error(f"Error processing files: {e}")
                st.session_state.analyze_clicked = False

    # --- Global Brightness Heatmap (across all SIFs) ---
    with plot_col2:
        st.markdown("### Global Brightness Heatmap")
        show_heatmap = st.toggle("Show heatmap (all SIFs)", value=False, help="Aggregates brightness across all detections from all uploaded .sif files.")
        if show_heatmap:
            # Smoothing controls
            smooth_sigma = st.slider("Smoothing (σ, px)", min_value=0.0, max_value=8.0, value=2.0, step=0.5,
                                     help="Apply Gaussian smoothing to reduce patchy coverage. Set to 0 for no smoothing.")
            heat_cmap = st.selectbox("Heatmap colormap", options=["magma", "inferno", "plasma", "viridis", "hot", "cividis"], index=0)
    
            try:
                # Use current image shape as a hint for consistent sizing
                shape_hint = image_data_cps.shape if isinstance(image_data_cps, np.ndarray) else None
                heatmap = build_brightness_heatmap(processed_data, weight_col="brightness_fit", shape_hint=shape_hint)
    
                # Optional smoothing
                if smooth_sigma > 0:
                    if gaussian_filter is not None:
                        heatmap = gaussian_filter(heatmap, sigma=smooth_sigma, mode="nearest")
                    else:
                        # Lightweight fallback: simple box blur via convolution
                        k = int(max(1, round(smooth_sigma * 3)))
                        kernel = np.ones((k, k), dtype=np.float64)
                        kernel /= kernel.sum()
                        # Pad and convolve (valid for small kernels and moderate sizes)
                        from numpy.lib.stride_tricks import sliding_window_view
                        if heatmap.shape[0] >= k and heatmap.shape[1] >= k:
                            windows = sliding_window_view(
                                np.pad(heatmap, ((k//2, k-1-k//2), (k//2, k-1-k//2)), mode="edge"),
                                (k, k)
                            )
                            heatmap = (windows * kernel).sum(axis=(-1, -2))
    
                # Plot
                import matplotlib.pyplot as plt
                fig_hm, ax_hm = plt.subplots()
                im = ax_hm.imshow(heatmap, origin="lower", cmap=heat_cmap, norm=None)
                ax_hm.set_title("Brightness Heatmap (All SIFs)")
                ax_hm.set_xlabel("X (px)")
                ax_hm.set_ylabel("Y (px)")
                cbar = fig_hm.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
                cbar.set_label("Summed brightness (pps)")
    
                st.pyplot(fig_hm)
    
                # Download SVG
                hm_svg_buf = io.StringIO()
                fig_hm.savefig(hm_svg_buf, format="svg", bbox_inches="tight")
                hm_svg_data = hm_svg_buf.getvalue()
                hm_svg_buf.close()
    
                st.download_button(
                    label="Download heatmap (SVG)",
                    data=hm_svg_data,
                    file_name="global_brightness_heatmap.svg",
                    mime="image/svg+xml"
                )
    
            except Exception as e_hm:
                st.warning(f"Couldn't build heatmap: {e_hm}")

    

