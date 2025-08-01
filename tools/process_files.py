import streamlit as st
import os
import pandas as pd
from matplotlib.colors import LogNorm
# ... import other necessary libraries like plot_brightness, plot_histogram, etc.

# Assumes integrate_sif, plot_brightness, plot_histogram, etc. are defined elsewhere.
# These functions should be compatible with the new approach.
# For example, plot_brightness needs a single image and df.
# plot_histogram needs a single df (the combined one).

@st.cache_data
def process_files(uploaded_files, region):
    """
    Processes all uploaded .sif files and returns a dictionary of dataframes
    and images, plus a single combined dataframe for the histogram.
    """
    processed_data = {}
    all_dfs = []
    
    # Use a temporary directory to store files
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            df, image_data_cps = integrate_sif(file_path, region=region)
            processed_data[uploaded_file.name] = {
                "df": df,
                "image": image_data_cps,
            }
            all_dfs.append(df)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            
    # Combine all dataframes into a single one for the histogram
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
    else:
        combined_df = pd.DataFrame()
        
    return processed_data, combined_df
