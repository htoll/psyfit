import streamlit as st
import os
import io
from utils import integrate_sif, plot_brightness, plot_histogram, sort_UCNP_dye_sifs, coloc_subplots, match_ucnp_dye_files
from tools.process_files import process_files
from matplotlib.colors import LogNorm
import numpy as np

# def run():
#     col1, col2 = st.columns([1, 2])

#     with col1:
#         st.header("Colocalize ##Beta##")
#         uploaded_files = st.file_uploader("Upload .sif file", type=["sif"], accept_multiple_files=True)
#         ucnp_threshold = st.number_input("UCNP threshold", min_value=0, value=2,
#                                          key="ucnp_threshold_input",
#                                          help='''
#                                          Stringency of fit, higher value is more selective:
#                                          -UCNP signal sets absolute peak cut off
#                                          -Dye signal sets sensitivity of blob detection
#                                          ''')
#         dye_threshold = st.number_input("Dye threshold", min_value=0, value=5,
#                                         key="dye_threshold_input",
#                                         help='''
#                                         Stringency of fit, higher value is more selective:
#                                         -UCNP signal sets absolute peak cut off
#                                         -Dye signal sets sensitivity of blob detection
#                                         ''')
#         diagram = """ Splits sif into quadrants (256x256 px):
#         ┌─┬─┐
#         │ 1 │ 2 │
#         ├─┼─┤
#         │ 3 │ 4 │
#         └─┴─┘
#         """
#         ucnp_region = st.selectbox("UCNP Region", options=["1", "2", "3", "4", "all"], help=diagram)
#         dye_region = st.selectbox("Dye Region", options=["1", "2", "3", "4", "all"], help=diagram)

#         coloc_radius = st.number_input("Colocalization Radius", min_value=1, value=2, help='Max radius to associate two PSFs')
#         export_format = st.selectbox("Export Format", options=["SVG", "TIFF", "PNG", "JPEG"])
#         ucnp_id = st.text_input("UCNP ID:", value="976")
#         dye_id = st.text_input("Dye ID:", value="638")

#     with col2:
#         show_fits = st.checkbox("Show fits")
#         use_log_norm = st.checkbox("Log Image Scaling")
#         norm = LogNorm() if use_log_norm else None

#         univ_minmax = st.checkbox("Universal Scaling")
#         if "Analyze" not in st.session_state:
#             st.session_state.convert = False

#         if st.button("Analyze"):
#             st.session_state.convert = True

#         if st.session_state.convert and uploaded_files:
#             ucnp_list, dye_list = sort_UCNP_dye_sifs(uploaded_files, ucnp_id=ucnp_id, dye_id=dye_id)
#             df_dict = {}

#             # Process UCNP files and use file.name as the key
#             for file in ucnp_list:
#                 try:
#                     file.seek(0)
#                     st.write(f"Reading dye file: {file.name} (size: {len(file.read())})")
#                     file.seek(0)
#                     df, cropped_img = integrate_sif(file, threshold=ucnp_threshold, region=ucnp_region, signal='UCNP')
#                     if df is not None:
#                         st.write(f"Success parsing: {file.name}")
#                         df_dict[file.name] = (df, cropped_img)
#                     else:
#                         st.warning(f"integrate_sif returned None for {file.name}")
#                 except Exception as e:
#                     st.error(f"Could not process dye file: {file.name}")
#                     st.exception(e)
#             # Process dye files and use file.name as the key
#             for file in dye_list:
#                 try:
#                     file.seek(0)
#                     st.write(f"Reading dye file: {file.name} (size: {len(file.read())})")
#                     file.seek(0)
#                     df, cropped_img = integrate_sif(file, threshold=dye_threshold, region=dye_region, signal='dye')
#                     if df is not None:
#                         st.write(f"Success parsing: {file.name}")
#                         df_dict[file.name] = (df, cropped_img)
#                     else:
#                         st.warning(f"integrate_sif returned None for {file.name}")
#                 except Exception as e:
#                     st.error(f"Could not process dye file: {file.name}")
#                     st.exception(e)
            
#             coloc_subplots(
#                 ucnp_list,
#                 dye_list,
#                 df_dict,
#                 show_fits=show_fits,
#                 export_format=export_format,
#                 colocalization_radius=coloc_radius,
#                 pix_size_um=0.1  
#             )


def run():
    st.header("Colocalize Beta")
    uploaded_files = st.file_uploader("Upload .sif files", type="sif", accept_multiple_files=True)
    if not uploaded_files:
        st.info("Please upload .sif files to continue.")
        return

    ucnp_id = st.text_input("UCNP ID", value="976")
    dye_id = st.text_input("Dye ID", value="638")
    coloc_radius = st.number_input("Colocalization Radius (pixels)", min_value=1, value=2)
    threshold_ucnp = st.number_input("UCNP Threshold", min_value=0, value=2)
    threshold_dye = st.number_input("Dye Threshold", min_value=0, value=5)
    show_fits = st.checkbox("Show Fits", value=True)
    export_format = st.selectbox("Export Format", ["SVG", "TIFF", "PNG", "JPEG"])

    ucnp_files, dye_files = sort_UCNP_dye_sifs(uploaded_files, ucnp_id, dye_id)
    df_dict = {}

    for f in ucnp_files + dye_files:
        try:
            data = f.read()
            signal = 'UCNP' if f in ucnp_files else 'dye'
            region = "all"
            threshold = threshold_ucnp if signal == 'UCNP' else threshold_dye
            df, image = integrate_sif(io.BytesIO(data), threshold=threshold, region=region, signal=signal)
            df_dict[f.name] = (df, image)
        except Exception as e:
            st.error(f"Failed to parse {f.name}: {e}")

    pairs = match_ucnp_dye_files(ucnp_files, dye_files)

    if not pairs:
        st.warning("No matched UCNP/dye file pairs.")
        return

    for i, (uf, df_) in enumerate(pairs):
        st.subheader(f"Pair {i+1}: {uf.name} and {df_.name}")
        if uf.name not in df_dict or df_.name not in df_dict:
            st.warning(f"Skipping: Missing data for {uf.name} or {df_.name}")
            continue
        coloc_df = coloc_subplots(uf, df_, df_dict, colocalization_radius=coloc_radius, show_fits=show_fits, pix_size_um=0.1)
        st.dataframe(coloc_df)



