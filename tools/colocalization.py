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
      st.header("Colocalize")
      uploaded_files = st.file_uploader("Upload .sif file", type=["sif"], accept_multiple_files=True)
      threshold = st.number_input("Threshold", min_value=0, value=2, help = '''
      Stringency of fit, higher value is more selective:  
      -UCNP signal sets absolute peak cut off  
      -Dye signal sets sensitivity of blob detection
      ''')
      diagram = """ Splits sif into quadrants (256x256 px):  
      ┌─┬─┐  
      │ 1 │ 2 │  
      ├─┼─┤  
      │ 3 │ 4 │  
      └─┴─┘
      """
      region = st.selectbox("Region", options=["1", "2", "3", "4", "all"], help = diagram)
      coloc_radius = st.number_input("Colocalization Radius", min_value=1, value = 2, help = 'Max radius to associate two PSFs')
      export_format = st.selectbox("Export Format", options=["SVG","TIFF", "PNG", "JPEG"])

  
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
              ucnp_list, dye_list = sort_UCNP_dye_sifs(uploaded_files)
              df_dict = {}
            # Process UCNP files
              for file in ucnp_list:
                  df, cropped_img = integrate_sif(file, threshold=ucnp_threshold, region=ucnp_region, signal='UCNP')
                  df_dict[file] = (df, cropped_img)
              
              # Process dye  files
              for file in dye_list:
                  df, cropped_img = integrate_sif(file, threshold=dye_threshold, region=dye_region, signal='dye')
                  df_dict[file] = (df, cropped_img)

            
              processed_data, combined_df = process_files(uploaded_files, region)
              
              coloc_subplots(ucnp_list, dye_list, df_dict, show_fits=show_fits, 
                             export_format = export_format, colocalization_radius=coloc_radius)

        
          except Exception as e:
              st.error(f"An error occurred: {e}")



