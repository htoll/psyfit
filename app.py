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

# Region breakdown:
#    1 | 2
#    -----
#    3 | 4

def main():
    st.title("psfit analysis")
    
    # File uploader widget
    uploaded_file = st.file_uploader("Upload a .sif file", type=["sif"])
    
   

if __name__ == "__main__":
    main()

if st.button("Analyze single sif"):
     if uploaded_file is not None:
        # Save the uploaded file to disk (Streamlit's file_uploader returns BytesIO)
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Optional: Add inputs for threshold, region, signal
        threshold = st.number_input("Threshold", min_value=0, value=2)
        region = st.text_input("Region", value="1")
        signal = st.text_input("Signal", value="UCNP")

        # Process file and plot
        try:
            ex_df, image_data = integrate_sif(uploaded_file.name, threshold=threshold, region=region, signal=signal)
            plot_brightness(image_data, ex_df)
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
