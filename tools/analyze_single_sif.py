import streamlit as st
import os
import io
from utils import integrate_sif, plot_brightness, plot_histogram
from tools.process_files import process_files
from matplotlib.colors import LogNorm
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import norm
from scipy.ndimage import gaussian_filter


from sklearn.mixture import GaussianMixture

from mpl_toolkits.axes_grid1 import make_axes_locatable


blue_ch_color = 'dodgerblue'
green_ch_color = 'forestgreen'
red_ch_color = 'tomato'
nir_ch_color = 'darkorange'


def build_brightness_heatmap(processed_data, weight_col="brightness_integrated", shape_hint=None):
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
    @st.cache_data
    def df_to_csv_bytes(df):
        return df.to_csv(index=False).encode("utf-8")

    if "analyze_clicked" not in st.session_state:
        st.session_state.analyze_clicked = False

    processed_data = None
    image_data_cps = None

    with st.sidebar:
        st.header("Analyze SIF Files")
        uploaded_files = st.file_uploader("Upload .sif file", type=["sif"], accept_multiple_files=True)
        threshold = st.number_input("Threshold", min_value=0, value=10, help='''
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
        gmm_components = st.sidebar.number_input("GMM Components", min_value=1, value=2)


        signal = st.selectbox("Signal", options=["UCNP", "dye"], help='''Changes detection method:
                                                                - UCNP for high SNR (sklearn peakfinder)
                                                                - dye for low SNR (sklearn blob detection)''')
        min_distance = st.number_input("Minimum Distance", min_value=1, value=5, help='Min distance between PSFs (px)')
        pix_size_um = st.number_input("Pixel Size (µm)", min_value = 0.01, value = 0.1)

        cmap = st.selectbox("Colormap", options=[ 'gray', 'plasma', "magma", 'viridis', 'hot', 'hsv'])
        show_fits = st.checkbox("Show fits", value=True)
        normalization = st.checkbox("Log Image Scaling")
        save_format = st.selectbox("Download format", options=["svg", "png", "jpeg"]).lower()
        show_heatmap = st.toggle(
            "Show heatmap (all SIFs)",
            value=False,
            help="Aggregates brightness across all detections from all uploaded .sif files.",
        )
        mcl_toggle = st.toggle("All channel MCL brightness", help="Overrides region to 'all' and splits analysis by quadrant.")

    if st.button("Analyze"):
        st.session_state.analyze_clicked = True

    if mcl_toggle:
        region = "all"

    brightness_col, hist_col = st.columns([3, 1])
    mime_map = {"svg": "image/svg+xml", "png": "image/png", "jpeg": "image/jpeg"}

    if st.session_state.analyze_clicked and uploaded_files:
        try:
            processed_data, combined_df = process_files(uploaded_files, 
                                                        region, 
                                                        threshold=threshold, 
                                                        signal=signal,
                                                       min_distance = min_distance,
                                                       pix_size_um = pix_size_um)
            if mcl_toggle and combined_df is not None and not combined_df.empty:
                # Assign quadrants based on pixel coordinates
                conditions = [
                    (combined_df['y_pix'] > 256) & (combined_df['x_pix'] < 256),  # Blue
                    (combined_df['y_pix'] > 256) & (combined_df['x_pix'] > 256),  # Green
                    (combined_df['y_pix'] < 256) & (combined_df['x_pix'] < 256),  # Red
                    (combined_df['y_pix'] < 256) & (combined_df['x_pix'] > 256)   # NIR
                ]
                choices = ['Blue', 'Green', 'Red', 'NIR']
                combined_df['quadrant'] = np.select(conditions, choices, default='Border')


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
                    cmap=cmap,
                    interactive=True,
                )
                if mcl_toggle:

                    if hasattr(fig_image, "savefig"):
                        # Matplotlib annotations
                        ax = fig_image.gca()
                        ax.axhline(256, color='white', linestyle='--', alpha=0.6)
                        ax.axvline(256, color='white', linestyle='--', alpha=0.6)
                        ax.text(128, 384, 'Blue', color='cyan', fontsize=14, ha='center', weight='bold')
                        ax.text(384, 384, 'Green', color='lime', fontsize=14, ha='center', weight='bold')
                        ax.text(128, 128, 'Red', color='red', fontsize=14, ha='center', weight='bold')
                        ax.text(384, 128, 'NIR', color='pink', fontsize=14, ha='center', weight='bold')
                    else:
                        # Plotly annotations
                        fig_image.add_hline(y=256, line_dash="dash", line_color="white", opacity=0.6)
                        fig_image.add_vline(x=256, line_dash="dash", line_color="white", opacity=0.6)
                        fig_image.add_annotation(x=128, y=500, text="Blue", showarrow=False, font=dict(color=blue_ch_color, size=20, weight='bold'))
                        fig_image.add_annotation(x=384, y=500, text="Green", showarrow=False, font=dict(color=green_ch_color, size=20, weight='bold'))
                        fig_image.add_annotation(x=128, y=245, text="Red", showarrow=False, font=dict(color=red_ch_color, size=20, weight='bold'))
                        fig_image.add_annotation(x=384, y=245, text="NIR", showarrow=False, font=dict(color=nir_ch_color, size=20, weight='bold'))

                with brightness_col:
                    if hasattr(fig_image, "savefig"):
                        fig_image.set_size_inches(8, 8)
                        st.pyplot(fig_image, use_container_width=True)
                        buffer = io.BytesIO()
                        fig_image.savefig(buffer, format=save_format)
                        st.download_button(
                            label=f"Download PSFs ({save_format})",
                            data=buffer.getvalue(),
                            file_name=f"{selected_file_name}.{save_format}",
                            mime=mime_map[save_format],
                        )
                    else:
                        fig_image.update_layout(height=640)
                        fmt = save_format.lower()
                        if fmt not in {"png", "jpeg", "jpg", "svg", "webp"}:
                            fmt = "png"
                        st.plotly_chart(
                            fig_image,
                            use_container_width=True,
                            config={
                                "displaylogo": False,
                                "modeBarButtonsToRemove": ["select2d", "lasso2d", "toggleSpikelines"],
                                "toImageButtonOptions": {"format": fmt},
                            },
                        )
                        html_bytes = fig_image.to_html().encode("utf-8")
                        st.download_button(
                            label="Download PSFs (HTML)",
                            data=html_bytes,
                            file_name=f"{selected_file_name}.html",
                            mime="text/html",
                        )

                    if combined_df is not None and not combined_df.empty:
                        csv_bytes = df_to_csv_bytes(combined_df)
                        st.download_button(
                            label="Download as CSV",
                            data=csv_bytes,
                            file_name=f"{os.path.splitext(selected_file_name)[0]}_compiled.csv",
                            mime="text/csv",
                        )
                    else:
                        st.info("No compiled data available to download yet.")

                with hist_col:
                    if not combined_df.empty:
                        brightness_vals = combined_df['brightness_integrated'].values
                        default_min_val = float(np.min(brightness_vals))
                        default_max_val = float(np.max(brightness_vals))



                        try:
                            user_min = float(default_min_val)
                            user_max = float(default_max_val)
                        except ValueError:
                            st.warning("Please enter valid numbers (you can use scientific notation like 1e6).")
                            return

                        

                        if user_min < user_max:
                            if mcl_toggle:
                                import matplotlib.pyplot as plt
                                # Create a 4-panel subplot for the channels
                                fig_hist, axes = plt.subplots(4, 1, figsize=(4, 8), sharex=True)
                                channels = ['Blue', 'Green', 'Red', 'NIR']
                                colors = [blue_ch_color, green_ch_color, red_ch_color, nir_ch_color]
                                
                                for ax, channel, color in zip(axes, channels, colors):
                                    # Filter data for this specific channel
                                    chan_data = combined_df[(combined_df['quadrant'] == channel) & 
                                                            (combined_df['brightness_integrated'] >= user_min) & 
                                                            (combined_df['brightness_integrated'] <= user_max)]['brightness_integrated'].values
                                    
                                    if len(chan_data) == 0:
                                        ax.text(0.5, 0.5, f"No {channel} emission", ha='center', va='center', transform=ax.transAxes)
                                        ax.set_yticks([])
                                      #  st.warning(f"No emission detected in the {channel} channel.")
                                        continue
                                        
                                    # Plot histogram (removed density=True to show raw counts)
                                    # We capture 'bins' to calculate the bin width for scaling the fit
                                    counts, bins, _ = ax.hist(chan_data, bins='auto', color=color, alpha=0.6, density=False)
                                    ax.set_ylabel(channel, color=color, weight='bold')
                                    
                                    # Gaussian fit
                                    if len(chan_data) > int(gmm_components):
                                        # Use chan_data here, NOT brightness_vals
                                        X_chan = chan_data.reshape(-1, 1)
                                        gmm = GaussianMixture(n_components=int(gmm_components), random_state=42).fit(X_chan)
                                        
                                        # 1. Generate plot data using 'bins' (the variable returned by ax.hist)
                                        x_fit = np.linspace(bins[0], bins[-1], 500).reshape(-1, 1)
                                        pdf = np.exp(gmm.score_samples(x_fit))
                                        
                                        # Scale PDF to match raw counts
                                        bin_width = bins[1] - bins[0]
                                        y_fit = pdf * len(chan_data) * bin_width
                                        
                                        # Use n_components from the sidebar
                                        ax.plot(x_fit, y_fit, color='black', linewidth=1, label=f"{gmm_components}-comp GMM")
                                    
                                        # 2. Extract Primary Peak Stats for Title
                                        idx = np.argmax(gmm.weights_)
                                        mu_primary = gmm.means_.flatten()[idx]
                                        sigma_primary = np.sqrt(gmm.covariances_.flatten()[idx])
                                        sigma_over_mu = (sigma_primary / mu_primary * 100) if mu_primary != 0 else 0
                                        n_points = len(chan_data)
                                    
                                        # 3. Set Title
                                        ax.set_title(
                                            f"μ={mu_primary:.2e} ± {sigma_primary:.2e} pps \nσ/μ={sigma_over_mu:.1f}%| n={n_points}",
                                            fontsize=10 * scale, 
                                            pad=10
                                        )
                                        
                                        # Parameters for return (all components)
                                        mu = gmm.means_.flatten()
                                        sigma = np.sqrt(gmm.covariances_.flatten())
                                        ax.legend(fontsize=8 * scale)
                                
                                axes[-1].set_xlabel('Brightness (pps)')
                                fig_hist.tight_layout()
                                st.pyplot(fig_hist, use_container_width=True)
                                
                                # Download button for the MCL histogram
                                hist_buffer = io.BytesIO()
                                fig_hist.savefig(hist_buffer, format=save_format)
                                st.download_button(
                                    label=f"Download MCL hist ({save_format})",
                                    data=hist_buffer.getvalue(),
                                    file_name=f"mcl_histogram.{save_format}",
                                    mime=mime_map[save_format],
                                )
                                
                            else:
                                fig_hist, _, _ = plot_histogram(
                                    combined_df,
                                    min_val=user_min,
                                    max_val=user_max,
                                    num_bins='auto',
                                    n_components = int(gmm_components)
                                    
                                )
                                st.pyplot(fig_hist, use_container_width=True)
                        else:
                            st.warning("Min greater than max.")
                        
                        user_min_val_str = st.text_input("Min Brightness (pps)", value=f"{default_min_val:.2e}")
                        user_max_val_str = st.text_input("Max Brightness (pps)", value=f"{default_max_val:.2e}")
                       # num_bins = st.number_input("# Bins:", value=20)
            else:
                st.error(f"Data for file '{selected_file_name}' not found.")

        except Exception as e:
            st.error(f"Error processing files: {e}")
            st.session_state.analyze_clicked = False

    # --- Global Brightness Heatmap (across all SIFs) ---
    if show_heatmap:
        with hist_col:
            if processed_data:
                smooth_sigma = st.slider(
                    "Smoothing (σ, px)",
                    min_value=0.0,
                    max_value=8.0,
                    value=2.0,
                    step=0.5,
                    help="Apply Gaussian smoothing to reduce patchy coverage. Set to 0 for no smoothing.",
                )
                heat_cmap = st.selectbox(
                    "Heatmap colormap",
                    options=["hot", "magma", "inferno", "plasma", "viridis", "cividis"],
                    index=0,
                )

                try:
                    shape_hint = image_data_cps.shape if isinstance(image_data_cps, np.ndarray) else None
                    heatmap = build_brightness_heatmap(processed_data, weight_col="brightness_integrated", shape_hint=shape_hint)

                    if smooth_sigma > 0:
                        if gaussian_filter is not None:
                            heatmap = gaussian_filter(heatmap, sigma=smooth_sigma, mode="nearest")
                        else:
                            k = int(max(1, round(smooth_sigma * 3)))
                            kernel = np.ones((k, k), dtype=np.float64)
                            kernel /= kernel.sum()
                            from numpy.lib.stride_tricks import sliding_window_view
                            if heatmap.shape[0] >= k and heatmap.shape[1] >= k:
                                windows = sliding_window_view(
                                    np.pad(heatmap, ((k//2, k-1-k//2), (k//2, k-1-k//2)), mode="edge"),
                                    (k, k)
                                )
                                heatmap = (windows * kernel).sum(axis=(-1, -2))

                    import matplotlib.pyplot as plt
                    fig_hm, ax_hm = plt.subplots()
                    im = ax_hm.imshow(heatmap, origin="lower", cmap=heat_cmap, norm=None)
                    ax_hm.set_title("Brightness Heatmap (All SIFs)")
                    ax_hm.set_xlabel("X (px)")
                    ax_hm.set_ylabel("Y (px)")

                    divider = make_axes_locatable(ax_hm)
                    cax = divider.append_axes("right", size="1%", pad=0.01)


                    cbar = fig_hm.colorbar(im, cax=cax)
                    cbar.set_label("Summed brightness (pps)")

                    st.pyplot(fig_hm)

                    hm_svg_buf = io.StringIO()
                    fig_hm.savefig(hm_svg_buf, format="svg")
                    st.download_button(
                        label="Download heatmap",
                        data=hm_svg_buf.getvalue(),
                        file_name="brightness_heatmap.svg",
                        mime="image/svg+xml",
                    )

                except Exception as e_hm:
                    st.warning(f"Couldn't build heatmap: {e_hm}")
            else:
                st.info("Run analysis to build heatmap.")
