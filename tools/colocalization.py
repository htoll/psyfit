import streamlit as st
import os
import io
from utils import integrate_sif, plot_brightness, plot_histogram, sort_UCNP_dye_sifs, coloc_subplots
from tools.process_files import process_files
from matplotlib.colors import LogNorm
import numpy as np

def run():
  col1, col2 = st.columns([1, 2])

  with col1:
      st.header("Colocalize ##Beta##")
      uploaded_files = st.file_uploader("Upload .sif file", type=["sif"], accept_multiple_files=True)
      ucnp_threshold = st.number_input("UCNP threshold", min_value=0, value=2,  
                                       key="ucnp_threshold_input", 
                                       help = '''
                                        Stringency of fit, higher value is more selective:  
                                        -UCNP signal sets absolute peak cut off  
                                        -Dye signal sets sensitivity of blob detection
                                        ''')
      dye_threshold = st.number_input("Dye threshold", min_value=0, value=5, 
                                      key="dye_threshold_input",
                                      help = '''
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
      ucnp_region = st.selectbox("UCNP Region", options=["1", "2", "3", "4", "all"], help = diagram)
      dye_region = st.selectbox("Dye Region", options=["1", "2", "3", "4", "all"], help = diagram)

      coloc_radius = st.number_input("Colocalization Radius", min_value=1, value = 2, help = 'Max radius to associate two PSFs')
      export_format = st.selectbox("Export Format", options=["SVG","TIFF", "PNG", "JPEG"])
      ucnp_id = st.text_input("UCNP ID:", value = "976")
      dye_id = st.text_input("Dye ID:", value = "638")

  
  with col2:
      show_fits = st.checkbox("Show fits")
      use_log_norm = st.checkbox("Log Image Scaling")
      norm = LogNorm() if use_log_norm else None
  
      univ_minmax = st.checkbox("Universal Scaling")
      if "Analyze" not in st.session_state:
          st.session_state.convert = False
  
      if st.button("Analyze"):
          st.session_state.convert = True
  
      if st.session_state.convert and uploaded_files:
          ucnp_list, dye_list = sort_UCNP_dye_sifs(uploaded_files, ucnp_id=ucnp_id, dye_id=dye_id)
          df_dict = {}
  
          # Process UCNP files
          for file in ucnp_list:
              try:
                  df, cropped_img = integrate_sif(file, threshold=ucnp_threshold, region=ucnp_region, signal='UCNP')
                  if df is not None:
                      df_dict[file.name] = (df, cropped_img)
              except Exception as e:
                  st.error(f"Could not process UCNP file: {file.name}")
                  st.exception(e)
  
          # Process dye files
          for file in dye_list:
              try:
                  df, cropped_img = integrate_sif(file, threshold=dye_threshold, region=dye_region, signal='dye')
                  if df is not None:
                      df_dict[file.name] = (df, cropped_img)
              except Exception as e:
                  st.error(f"Could not process dye file: {file.name}")
                  st.exception(e)
  
          processed_data, combined_df = process_files(uploaded_files, region = 'all')
          ucnp_names = [f.name for f in ucnp_list]
          dye_names = [f.name for f in dye_list]
          coloc_subplots(ucnp_names, dye_names, df_dict, show_fits=show_fits,
                         export_format=export_format, colocalization_radius=coloc_radius)




