import streamlit as st
from utils import integrate_sif, plot_brightness, sort_UCNP_dye_sifs, natural_sort_key, match_ucnp_dye_files, coloc_subplots, extract_subregion, gaussian2d, HWT_aesthetic, plot_histogram
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
st.set_page_config(layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
tool = st.sidebar.radio("Select a tool:", [
    "Analyze single SIF",
    "Analyze Colocalization Set",
    "Batch Convert SIFs",
    "Visualize Data"
])
col1, col2 = st.columns([1, 2])

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
        show_fits = st.checkbox("Show fits")
        plot_brightness_histogram = st.checkbox("Plot brightness histogram")
        normalization = st.checkbox("Log Image Scaling")


    with col2:
        if st.button("Fit PSFs"):
            if uploaded_file is not None:
                try:
                    os.makedirs("temp", exist_ok=True)
                    file_path = os.path.join("temp", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    df, image_data_cps = integrate_sif(file_path, region = region)
                    plot_col1, plot_col2 = st.columns(2)
    
                    with plot_col1:
                        fig_image = plot_brightness(image_data_cps, df, show_fits=show_fits, normalization=normalization, pix_size_um=0.1, save_as_svg = save_as_svg)
                        st.pyplot(fig_image)
                                st.download_button(
                                label="Download PSfs",
                                data=fig_image,
                                file_name="figure.svg",
                                mime="image/svg+xml"
                                                    )
    
                    if plot_brightness_histogram:
                        with plot_col2:
                            fig_hist = plot_histogram(df, save_as_svg = save_as_svg)
                            st.pyplot(fig_hist)

    
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




