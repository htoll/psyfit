import streamlit as st
import os
import io
from utils import integrate_sif, plot_brightness, plot_histogram
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
        show_fits = st.checkbox("Show fits")
        normalization = st.checkbox("Log Image Scaling")

    with col2:
        st.header("placeholder")
