import streamlit as st
import os
import io
from utils import integrate_sif, plot_brightness, plot_histogram, sort_UCNP_dye_sifs, coloc_subplots, match_ucnp_dye_files
from tools.process_files import process_files
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import pandas as pd


def run():
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Colocalize ##Beta##")
        uploaded_files = st.file_uploader("Upload .sif file", type=["sif"], accept_multiple_files=True)
        ucnp_threshold = st.number_input("UCNP threshold", min_value=0, value=2, key="ucnp_threshold_input")
        dye_threshold = st.number_input("Dye threshold", min_value=0, value=5, key="dye_threshold_input")

        diagram = """ Splits sif into quadrants (256x256 px):
        ┌─┬─┐
        │ 1 │ 2 │
        ├─╌─┤
        │ 3 │ 4 │
        └─┴─┘
        """
        ucnp_region = st.selectbox("UCNP Region", options=["1", "2", "3", "4", "all"], help=diagram)
        dye_region = st.selectbox("Dye Region", options=["1", "2", "3", "4", "all"], help=diagram)

        coloc_radius = st.number_input("Colocalization Radius", min_value=1, value=2, help='Max radius to associate two PSFs')
        export_format = st.selectbox("Export Format", options=["SVG", "TIFF", "PNG", "JPEG"])
        ucnp_id = st.text_input("UCNP ID:", value="976")
        dye_id = st.text_input("Dye ID:", value="638")

        single_ucnp_brightness = st.number_input("Single UCNP Brightness (pps)", min_value=1.0, value=5000.0)
        single_dye_brightness = st.number_input("Single Dye Brightness (pps)", min_value=1.0, value=1000.0)

    with col2:
        show_fits = st.checkbox("Show fits")
        use_log_norm = st.checkbox("Log Image Scaling")
        univ_minmax = st.checkbox("Universal Scaling")

        if "Analyze" not in st.session_state:
            st.session_state.convert = False

        if st.button("Analyze"):
            st.session_state.convert = True

        if st.session_state.convert and uploaded_files:
            ucnp_files, dye_files = sort_UCNP_dye_sifs(uploaded_files, ucnp_id, dye_id)
            df_dict = {}

            for f in ucnp_files + dye_files:
                try:
                    signal = 'UCNP' if f in ucnp_files else 'dye'
                    region = ucnp_region if signal == 'UCNP' else dye_region
                    threshold = int(ucnp_threshold) if signal == 'UCNP' else int(dye_threshold)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".sif") as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name

                    df, image = integrate_sif(tmp_path, threshold=threshold, region=region, signal=signal)
                    df_dict[f.name] = (df, image)
                except Exception as e:
                    st.error(f"Failed to parse {f.name}: {e}")

            pairs = match_ucnp_dye_files(ucnp_files, dye_files)

            if not pairs:
                st.warning("No matched UCNP/dye file pairs.")
                return

            all_results = []
            pair_labels = []
            for i, (uf, df_) in enumerate(pairs):
                if uf.name not in df_dict or df_.name not in df_dict:
                    st.warning(f"Skipping: Missing data for {uf.name} or {df_.name}")
                    continue
                coloc_df = coloc_subplots(uf, df_, df_dict, colocalization_radius=coloc_radius, show_fits=show_fits, pix_size_um=0.1)
                all_results.append(coloc_df)
                pair_labels.append(f"{uf.name} ↔ {df_.name}")

            if not all_results:
                st.warning("No data processed.")
                return

            compiled_df = pd.concat(all_results, ignore_index=True)
            compiled_df['num_ucnps'] = compiled_df['ucnp_brightness'] / single_ucnp_brightness
            compiled_df['num_dyes'] = compiled_df['dye_brightness'] / single_dye_brightness

            thresholded_df = compiled_df[compiled_df['ucnp_brightness'] >= 0.3 * single_ucnp_brightness]

            selected_pair = st.selectbox("Select a matched pair to view", pair_labels)
            st.dataframe(thresholded_df)

            # Scatter plots
            x = thresholded_df['num_ucnps'].values.reshape(-1, 1)
            y = thresholded_df['num_dyes'].values

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            ax1.scatter(x, y, alpha=0.6)
            ax1.set_xlabel('Number of UCNPs per PSF')
            ax1.set_ylabel('Number of Dyes per PSF')

            ax2.scatter(x, y, alpha=0.6)
            ax2.set_xlabel('Number of UCNPs per PSF')
            ax2.set_ylabel('Number of Dyes per PSF')
            ax2.set_xlim([0, 2])
            ax2.set_ylim([0, 400])

            st.pyplot(fig)

            mask = (thresholded_df['num_ucnps'] >= 0) & (thresholded_df['num_ucnps'] <= 2)
            y_subset = thresholded_df.loc[mask, 'num_dyes']
            mean_val = y_subset.mean()

            fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
            ax_hist.hist(y_subset, bins=20, edgecolor='black', color='#bc5090')
            ax_hist.set_xlabel('Number of Dyes per Single UCNP')
            ax_hist.set_ylabel('Count')
            ax_hist.set_title(f'Mean = {mean_val:.1f}')
            st.pyplot(fig_hist)

            csv = thresholded_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Thresholded Colocalization Data", csv, "thresholded_colocalization.csv", "text/csv")

# def run():
#     st.header("Colocalize Beta")
#     uploaded_files = st.file_uploader("Upload .sif files", type="sif", accept_multiple_files=True)
#     if not uploaded_files:
#         st.info("Please upload .sif files to continue.")
#         return

#     ucnp_id = st.text_input("UCNP ID", value="976")
#     dye_id = st.text_input("Dye ID", value="638")
#     coloc_radius = st.number_input("Colocalization Radius (pixels)", min_value=1, value=2)
#     threshold_ucnp = st.number_input("UCNP Threshold", min_value=0, value=2)
#     threshold_dye = st.number_input("Dye Threshold", min_value=0, value=5)
#     show_fits = st.checkbox("Show Fits", value=True)
#     export_format = st.selectbox("Export Format", ["SVG", "TIFF", "PNG", "JPEG"])

#     ucnp_files, dye_files = sort_UCNP_dye_sifs(uploaded_files, ucnp_id, dye_id)
#     df_dict = {}

#     for f in ucnp_files + dye_files:
#         try:
#             signal = 'UCNP' if f in ucnp_files else 'dye'
#             region = "all"
#             threshold = threshold_ucnp if signal == 'UCNP' else threshold_dye

#             with tempfile.NamedTemporaryFile(delete=False, suffix=".sif") as tmp:
#                 tmp.write(f.read())
#                 tmp_path = tmp.name

#             df, image = integrate_sif(tmp_path, threshold=threshold, region=region, signal=signal)
#             df_dict[f.name] = (df, image)
#         except Exception as e:
#             st.error(f"Failed to parse {f.name}: {e}")

#     pairs = match_ucnp_dye_files(ucnp_files, dye_files)

#     if not pairs:
#         st.warning("No matched UCNP/dye file pairs.")
#         return

#     for i, (uf, df_) in enumerate(pairs):
#         st.subheader(f"Pair {i+1}: {uf.name} and {df_.name}")
#         if uf.name not in df_dict or df_.name not in df_dict:
#             st.warning(f"Skipping: Missing data for {uf.name} or {df_.name}")
#             continue
#         coloc_df = coloc_subplots(uf, df_, df_dict, colocalization_radius=coloc_radius, show_fits=show_fits, pix_size_um=0.1)
#         st.dataframe(coloc_df)
