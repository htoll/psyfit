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


# Step control
if "analyze_started" not in st.session_state:
    st.session_state.analyze_started = False

# STEP 1: Initial button
if not st.session_state.analyze_started:
    if st.button("Analyze single SIF"):
        st.session_state.analyze_started = True
else:
    # STEP 2: Show file uploader and parameter inputs
    st.markdown("### Upload your `.sif` file and set parameters")
    
    uploaded_file = st.file_uploader("Choose a .sif file", type=["sif"])
    threshold = st.number_input("Threshold", min_value=0, value=2)
    region = st.text_input("Region", value="1")
    signal = st.text_input("Signal", value="UCNP")

    # STEP 3: Final analysis button
    if st.button("Run Analysis"):
        if uploaded_file is not None:
            try:
                # Save uploaded file
                os.makedirs("temp", exist_ok=True)
                file_path = os.path.join("temp", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Run your analysis
                ex_df, image_data = integrate_sif(file_path, threshold=threshold, region=region, signal=signal)
                plot_brightness(image_data, ex_df)

            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.warning("Please upload a .sif file before running analysis.")
