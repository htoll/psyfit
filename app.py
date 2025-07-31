import streamlit as st
from utils import integrate_sif, plot_brightness, sort_UCNP_dye_sifs, natural_sort_key, match_ucnp_dye_files, coloc_subplots, extract_subregion, gaussian2d, HWT_aesthetic
import sif_parser

from skimage.feature import peak_local_max
from skimage.feature import blob_log

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm


from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

from datetime import date
import os

# Region breakdown:
#    1 | 2
#    -----
#    3 | 4

import streamlit as st
import os

# Sidebar navigation
st.sidebar.title("Navigation")
tool = st.sidebar.radio("Select a tool:", [
    "Analyze single SIF",
    "Analyze Colocalization Set",
    "Batch Convert SIFs",
    "Visualize Data"
])
col1, col2 = st.columns([3, 1])  # 3 parts for left, 1 part for right (adjust ratio as you like)

# Tool: Analyze single SIF
if tool == "Analyze single SIF":
    with col1:
        st.header("Analyze Single SIF File")
        uploaded_file = st.file_uploader("Upload .sif file", type=["sif"])
        threshold = st.number_input("Threshold", min_value=0, value=2)
        region = st.selectbox("Signal", options=["1", "2", "3", "4", "all"])
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
        save_as_svg = st.checkbox("Save as SVG")
        plot_brightness_histogram = st.checkbox("Plot brightness histogram")

    if st.button("Fit PSFs"):
        if uploaded_file is not None:
            try:
                os.makedirs("temp", exist_ok=True)
                file_path = os.path.join("temp", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                ex_df, image_data = integrate_sif(file_path, threshold=threshold, region=region, signal=signal)
                plot_brightness(image_data, ex_df)

            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.warning("Please upload a .sif file.")

# Tool: Analyze Colocalization Set
elif tool == "Colocalization Set":
    st.header("Colocalization Set")
    st.info("This feature is under construction — implement logic here.")

# Tool: Batch Convert SIFs
elif tool == "Batch Convert SIFs":
    st.header("Batch Convert SIFs")
    st.info("This feature is under construction — implement logic here.")

# Tool: Visualize Data
elif tool == "Visualize Data":
    st.header("Visualize Data")
    st.info("This feature is under construction — implement logic here.")




