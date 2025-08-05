import streamlit as st
import os
import io
from utils import integrate_sif, plot_brightness, plot_histogram, plot_all_sifs
from tools.process_files import process_files
from matplotlib.colors import LogNorm
import numpy as np

def run():
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Convert SIF Files")
        uploaded_files = st.file_uploader("Upload .sif file", type=["sif"], accept_multiple_files=True)
        threshold = st.number_input("Threshold", min_value=0, value=2)
        region = st.selectbox("Region", options=["1", "2", "3", "4", "all"])
        export_format = st.selectbox("Export Format", options=["SVG", "PNG", "JPEG"])

        st.markdown("""
        ┌─┬─┐<br>
        │ 1 │ 2 │<br>
        ├─┼─┤<br>
        │ 3 │ 4 │<br>
        └─┴─┘
        """, unsafe_allow_html=True)
        signal = st.selectbox("Signal", options=["UCNP", "dye"])


    with col2:
        show_fits = st.checkbox("Show fits")
        use_log_norm = st.checkbox("Log Image Scaling")
        norm = LogNorm() if use_log_norm else None

        univ_minmax = st.checkbox("Universal Scaling")
        if "Convert" not in st.session_state:
            st.session_state.convert = False

        if st.button("Convert"):
            st.session_state.convert = True
        
        if st.session_state.convert and uploaded_files:
            try:
                processed_data, combined_df = process_files(uploaded_files, region)
                
                plot_all_sifs(sif_files=uploaded_files, 
                            df_dict=processed_data,
                            show_fits=show_fits, 
                            save_format=export_format, 
                            normalization=norm,
                             univ_minmax = univ_minmax)
            except Exception as e:
                st.error(f"An error occurred: {e}")


