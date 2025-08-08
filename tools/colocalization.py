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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def run():
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Colocalize ##Beta##")
        uploaded_files = st.file_uploader("Upload .sif file", type=["sif"], accept_multiple_files=True,
                                          help='''This function assumes dye and UCNP images are taken  
                                          sequentially and uses the Andor Solis automatic numbering to sort order''')
        ucnp_threshold = st.number_input("UCNP threshold", min_value=0, value=2, key="ucnp_threshold_input")
        dye_threshold = st.number_input("Dye threshold", min_value=0, value=25, key="dye_threshold_input")

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
        ucnp_id = st.text_input("UCNP ID:", value="976", help="Unique characters to identify UCNP sifs")
        dye_id = st.text_input("Dye ID:", value="638", help="Unique characters to identify UCNP sifs")

        single_ucnp_brightness = st.number_input("Single UCNP Brightness (pps)", min_value=1.0, value=25000.0)
        single_dye_brightness = st.number_input("Single Dye Brightness (pps)", min_value=1.0, value=200.0)

    with col2:
        show_fits = st.checkbox("Show fits")
        use_log_norm = st.checkbox("Log Image Scaling")
        univ_minmax = st.checkbox("Universal Scaling")

        if "Analyze" not in st.session_state:
            st.session_state.convert = False

        if st.button("Analyze"):
            st.session_state.convert = True
            st.session_state.coloc_data = None  # reset cache

        if st.session_state.convert and uploaded_files and 'coloc_data' not in st.session_state:
            ucnp_files, dye_files = sort_UCNP_dye_sifs(uploaded_files, ucnp_id, dye_id)
            df_dict = {}

            for f in ucnp_files + dye_files:
                try:
                    signal = 'UCNP' if f in ucnp_files else 'dye'
                    region = ucnp_region if signal == 'UCNP' else dye_region
                    threshold = int(ucnp_threshold) if signal == 'UCNP' else int(dye_threshold)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".sif") as tmp:
                        f.seek(0)
                        tmp.write(f.read())
                        tmp_path = tmp.name

                    df, image = integrate_sif(tmp_path, threshold=threshold, region=region, signal=signal)
                    df_dict[f.name] = (df, image)
                except Exception as e:
                    st.error(f"Failed to parse {f.name}: {e}")

            pairs = match_ucnp_dye_files(ucnp_files, dye_files)

            if not pairs:
                st.warning("No matched UCNP/dye file pairs found.")
                return

            compiled_results = {}
            pair_labels = []
            for i, (uf, df_) in enumerate(pairs):
                if uf.name not in df_dict or df_.name not in df_dict:
                    st.warning(f"Skipping: Missing data for {uf.name} or {df_.name}")
                    continue
                # Colocalization with improved matching
                ucnp_df, ucnp_img = df_dict[uf.name]
                dye_df, dye_img = df_dict[df_.name]

                required_cols = ['x_pix', 'y_pix', 'sigx_fit', 'sigy_fit', 'brightness_fit']
                if not all(col in ucnp_df.columns for col in required_cols) or not all(col in dye_df.columns for col in required_cols):
                    continue

                coloc_records = []
                matched_dye = set()

                for idx_ucnp, row_ucnp in ucnp_df.iterrows():
                    x_ucnp, y_ucnp = row_ucnp['x_pix'], row_ucnp['y_pix']
                    dx = dye_df['x_pix'] - x_ucnp
                    dy = dye_df['y_pix'] - y_ucnp
                    distances = np.hypot(dx, dy)
                    within_radius = distances <= coloc_radius
                    candidate_idxs = dye_df.index[within_radius & ~dye_df.index.isin(matched_dye)]

                    if len(candidate_idxs) == 0:
                        continue

                    closest_idx = candidate_idxs[np.argmin(distances[within_radius & ~dye_df.index.isin(matched_dye)])]
                    row_dye = dye_df.loc[closest_idx]
                    matched_dye.add(closest_idx)

                    coloc_records.append({
                        'x_pix': row_ucnp['x_pix'],
                        'y_pix': row_ucnp['y_pix'],
                        'ucnp_brightness': row_ucnp['brightness_fit'],
                        'dye_brightness': row_dye['brightness_fit']
                    })

                coloc_df = pd.DataFrame(coloc_records)
                st.write(f"{uf.name} & {df_.name}: {len(coloc_df)} colocalized PSFs")
                st.write(f"{uf.name} & {df_.name}: {len(coloc_df)} colocalized PSFs")
                if not coloc_df.empty:
                    pair_key = f"{uf.name} ↔ {df_.name}"
                    compiled_results[pair_key] = {
                        "df": coloc_df,
                        "ucnp_img": df_dict[uf.name][1],
                        "ucnp_df": df_dict[uf.name][0]
                    }
                    pair_labels.append(pair_key)

            if not compiled_results:
                st.warning("No colocalized points found.")
                return

            st.session_state.coloc_data = {
                "results": compiled_results,
                "labels": pair_labels
            }

            if 'coloc_data' not in st.session_state:
                st.warning("Run analysis first.")
                return

            compiled_results = st.session_state.coloc_data['results']
            pair_labels = st.session_state.coloc_data['labels']

            selected_pair = st.selectbox("Select a matched pair to view", pair_labels)
            selected_result = compiled_results[selected_pair]
            selected_df = selected_result["df"].copy()

            # Optional: show dye image too
            st.markdown("### Dye Image")
            dye_file = selected_pair.split(" ↔ ")[1]
            dye_img, dye_df = df_dict[dye_file][1], df_dict[dye_file][0]
            fig_dye = plot_brightness(dye_img, dye_df, show_fits=show_fits, normalization=use_log_norm, pix_size_um=0.1, cmap="magma")
            st.pyplot(fig_dye)

            st.markdown("### Image View")
            fig_image = plot_brightness(
                selected_result["ucnp_img"],
                selected_result["ucnp_df"],
                show_fits=show_fits,
                normalization=use_log_norm,
                pix_size_um=0.1,
                cmap="magma"
            )
            st.pyplot(fig_image)

            selected_df['num_ucnps'] = selected_df['ucnp_brightness'] / single_ucnp_brightness
            selected_df['num_dyes'] = selected_df['dye_brightness'] / single_dye_brightness

            thresholded_df = selected_df[selected_df['ucnp_brightness'] >= 0.3 * single_ucnp_brightness]

            st.dataframe(thresholded_df)

            # Scatter + Histogram across ALL results
            full_df = pd.concat([d['df'] for d in compiled_results.values()], ignore_index=True)
            full_df['num_ucnps'] = full_df['ucnp_brightness'] / single_ucnp_brightness
            full_df['num_dyes'] = full_df['dye_brightness'] / single_dye_brightness
            filtered_df = full_df[full_df['ucnp_brightness'] >= 0.3 * single_ucnp_brightness]

            x = filtered_df['num_ucnps'].values.reshape(-1, 1)
            y = filtered_df['num_dyes'].values

            model = LinearRegression().fit(x, y)
            y_pred = model.predict(x)
            r2 = r2_score(y, y_pred)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            ax1.scatter(x, y, alpha=0.6)
            ax1.plot(x, y_pred, color='red', label=f'y = {model.coef_[0]:.2f}x + {model.intercept_:.1f}\nR² = {r2:.2f}')
            ax1.set_xlabel('Number of UCNPs per PSF')
            ax1.set_ylabel('Number of Dyes per PSF')
            ax1.legend()

            mask = (filtered_df['num_ucnps'] >= 0) & (filtered_df['num_ucnps'] <= 2)
            y_subset = filtered_df.loc[mask, 'num_dyes']
            mean_val = y_subset.mean()

            ax2.hist(y_subset, bins=20, edgecolor='black', color='#bc5090')
            ax2.set_xlabel('Number of Dyes per Single UCNP')
            ax2.set_ylabel('Count')
            ax2.set_title(f'Mean = {mean_val:.1f}')

            st.pyplot(fig)

            csv = thresholded_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Thresholded Colocalization Data", csv, "thresholded_colocalization.csv", "text/csv")
