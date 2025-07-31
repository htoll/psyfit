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


# Create a session state flag
if "show_upload" not in st.session_state:
    st.session_state.show_upload = False

# First, show the Analyze button
if st.button("Analyze single SIF"):
    st.session_state.show_upload = True

# Only show file uploader and inputs after the button is clicked
if st.session_state.show_upload:
    st.markdown("### Upload SIF file")
    uploaded_file = st.file_uploader("Upload .sif file", type=["sif"])

    threshold = st.number_input("Threshold", min_value=0, value=2)
    region = st.text_input("Region", value="1")
    signal = st.text_input("Signal", value="UCNP")

    # When file is uploaded, process it immediately
    if uploaded_file is not None:
        try:
            # Save to disk
            file_path = os.path.join("temp", uploaded_file.name)
            os.makedirs("temp", exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Call your analysis
            ex_df, image_data = integrate_sif(file_path, threshold=threshold, region=region, signal=signal)
            plot_brightness(image_data, ex_df)

        except Exception as e:
            st.error(f"Error processing file: {e}")
    
