import streamlit as st
import os, io, tempfile, hashlib
import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px

from utils import integrate_sif, plot_brightness, plot_histogram, HWT_aesthetic
from tools.process_files import process_files

def _hash_file(uploaded_file):
    # stable cache key for content
    uploaded_file.seek(0)
    h = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return h

@st.cache_data(show_spinner=False)
def _process_files_cached(file_paths, region):
    # Do the heavy work ONCE per content/region combo
    # (You can include other params if they affect processing)
    return process_files(file_paths, region)

def run():
    col1, col2 = st.columns([1, 2])

    # hold paths and keys between reruns
    if "saved_files" not in st.session_state:
        st.session_state.saved_files = {}   # {display_name: local_path}
    if "processed" not in st.session_state:
        st.session_state.processed = None   # (processed_data, combined_df)

    with col1:
        uploaded_files = st.file_uploader(
            "Upload .sif file", type=["sif"], accept_multiple_files=True
        )

        # QUICK SAVE PHASE (do NOT process yet)
        if uploaded_files:
            for f in uploaded_files:
                key = f"{f.name}:{_hash_file(f)}"
                if key not in st.session_state.saved_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".sif") as tmp:
                        tmp.write(f.getbuffer())
                        st.session_state.saved_files[key] = (f.name, tmp.name)

            file_options = [display for (display, _) in st.session_state.saved_files.values()]
            selected_file_name = st.selectbox("Select sif to display:", options=file_options)

            threshold = st.number_input("Threshold", min_value=0, value=2, help='''
                Stringency of fit, higher value is more selective:  
                - UCNP signal sets absolute peak cut off  
                - Dye signal sets sensitivity of blob detection
            ''')
            diagram = """ Splits sif into quadrants (256x256 px):  
            ┌─┬─┐  
            │ 1 │ 2 │  
            ├─┼─┤  
            │ 3 │ 4 │  
            └─┴─┘
            """
            region = st.selectbox("Region", options=["1", "2", "3", "4", "all"], help=diagram)
            signal = st.selectbox("Signal", options=["UCNP", "dye"], help='''
                Changes detection method:  
                - UCNP for high SNR (sklearn peakfinder)  
                - dye for low SNR (sklearn blob detection)
            ''')
            cmap = st.selectbox("Colormap", options=["magma", 'viridis', 'plasma', 'hot', 'gray', 'hsv'])

            # PROCESS PHASE (explicit)
            to_process_paths = [p for (_, p) in st.session_state.saved_files.values()]
            if st.button("Process uploaded files"):
                with st.spinner("Processing…"):
                    # Use cached heavy function — returns fast on rerun
                    processed_data, combined_df = _process_files_cached(tuple(to_process_paths), region)
                    st.session_state.processed = (processed_data, combined_df)

    # DISPLAY PHASE (safe to render anytime)
    if st.session_state.get("processed"):
        processed_data, combined_df = st.session_state.processed

        plot_col1, plot_col2 = col2.columns(2)

        with plot_col1:
            show_fits = st.checkbox("Show fits")
            normalization = st.checkbox("Log Image Scaling")

            # map selected display name back to the processed key
            if uploaded_files:
                selected_display = st.session_state.get("selected_display")  # optional if you store it
                # your original lookup by name still works if keys match
                # but if process_files keyed by base filename only, ensure it matches here:
                # e.g., selected_file_name = st.selectbox(...) above can be kept in session_state

            # If your processed_data keys are the original basenames:
            # selected_file_name comes from the selectbox above
            if uploaded_files:
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
                    st.pyplot(fig_image)

                    svg_buffer = io.StringIO()
                    fig_image.savefig(svg_buffer, format='svg')
                    st.download_button(
                        label="Download PSFs",
                        data=svg_buffer.getvalue(),
                        file_name=f"{selected_file_name}.svg",
                        mime="image/svg+xml"
                    )
                else:
                    st.error(f"Data for file '{selected_file_name}' not found.")

        with plot_col2:
            if not combined_df.empty:
                brightness_vals = combined_df['brightness_fit'].values
                default_min_val = float(np.min(brightness_vals))
                default_max_val = float(np.max(brightness_vals))

                user_min_val_str = st.text_input("Min Brightness (pps)", value=f"{default_min_val:.2e}")
                user_max_val_str = st.text_input("Max Brightness (pps)", value=f"{default_max_val:.2e}")

                try:
                    user_min_val = float(user_min_val_str); user_max_val = float(user_max_val_str)
                except ValueError:
                    st.warning("Please enter valid numbers (you can use scientific notation like 1e6).")
                    st.stop()

                if user_min_val >= user_max_val:
                    st.warning("Min brightness must be less than max brightness.")
                else:
                    thresholding_method = st.selectbox("Choose thresholding method:", options=["Automatic (Mu/Sigma)", "Manual"], help="Automatic sets thresholds at 1.5μ, 2.5μ, 3.5μ")
                    num_bins = st.number_input("# Bins:", value=50)

                    fig_hist, mu, sigma = plot_histogram(
                        combined_df,
                        min_val=user_min_val,
                        max_val=user_max_val,
                        num_bins=num_bins
                    )

                    thresholds = []
                    if thresholding_method == "Automatic (Mu/Sigma)":
                        if mu is not None and sigma is not None:
                            for k in range(1, 4):
                                t = (2 * k + 1) / 2 * mu
                                if user_min_val < t < user_max_val:
                                    thresholds.append(t)
                        else:
                            st.warning("Gaussian fit failed to converge. Cannot perform automatic thresholding.")
                    else:
                        t1 = st.number_input("Threshold 1", min_value=user_min_val, max_value=user_max_val, value=(user_max_val + user_min_val) / 2)
                        t2 = st.number_input("Threshold 2", min_value=user_min_val, max_value=user_max_val, value=user_max_val * 0.75)
                        t3 = st.number_input("Threshold 3", min_value=user_min_val, max_value=user_max_val, value=user_max_val * 0.9)
                        thresholds = sorted([t1, t2, t3])

                    if thresholds:
                        fig_hist_final, _, _ = plot_histogram(
                            combined_df,
                            min_val=user_min_val,
                            max_val=user_max_val,
                            num_bins=num_bins,
                            thresholds=thresholds
                        )
                        st.pyplot(fig_hist_final)

                        with plot_col1:
                            thresholds = sorted(set(thresholds))
                            bins_for_pie = [user_min_val] + thresholds + [user_max_val]
                            num_bins_pie = len(bins_for_pie) - 1
                            base_labels = ["Monomers", "Dimers", "Trimers", "Multimers"]
                            labels_for_pie = base_labels[:num_bins_pie] if num_bins_pie <= len(base_labels) else base_labels + [f"Group {i+1}" for i in range(len(base_labels), num_bins_pie)]

                            if len(labels_for_pie) != num_bins_pie:
                                st.warning(f"Label/bin mismatch: {len(labels_for_pie)} labels for {num_bins_pie} bins.")
                            else:
                                import pandas as pd, matplotlib.colors as mcolors, plotly.express as px
                                categories = pd.cut(
                                    combined_df['brightness_fit'],
                                    bins=bins_for_pie,
                                    right=False,
                                    include_lowest=True,
                                    labels=labels_for_pie
                                )
                                category_counts = categories.value_counts().reset_index()
                                category_counts.columns = ['Category', 'Count']
                                palette = HWT_aesthetic()
                                region_colors = palette[:len(category_counts)]
                                plotly_colors = [mcolors.to_hex(c) for c in region_colors]

                                fig_pie = px.pie(
                                    category_counts,
                                    values='Count',
                                    names='Category',
                                    color_discrete_sequence=plotly_colors
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.pyplot(fig_hist)

        # Summary PSF count bar plot in left column (col1)
        with col1:
            if st.session_state.get("processed"):
                psf_counts = {
                    os.path.basename(name): len(processed_data[name]["df"])
                    for name in processed_data.keys()
                }
                import re
                def extract_sif_number(filename):
                    m = re.search(r'_([0-9]+)\.sif$', filename)
                    return m.group(1) if m else filename

                file_names = [extract_sif_number(n) for n in psf_counts.keys()]
                counts = list(psf_counts.values())
                mean_count = np.mean(counts)

                fig_count, ax_count = plt.subplots(figsize=(5, 3))
                ax_count.bar(file_names, counts)
                ax_count.axhline(mean_count, color='black', linestyle='--', label=f'Avg = {mean_count:.1f}', linewidth=0.5)
                ax_count.set_ylabel("# Fit PSFs", fontsize=10)
                ax_count.set_xlabel("SIF #", fontsize=10)
                ax_count.legend(fontsize=10)
                ax_count.tick_params(axis='x', labelsize=8)
                ax_count.tick_params(axis='y', labelsize=8)
                HWT_aesthetic()
                st.pyplot(fig_count)
