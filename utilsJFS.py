
import streamlit as st
import os
import pandas as pd

# Assuming 'integrate_sif' is in your utils file
from utils import integrate_sif

@st.cache_data
def process_files(uploaded_files, region, threshold=1, signal="UCNP", pix_size_um=0.1, sig_threshold=0.3, threshold_overrides={}):
    """
    Processes all uploaded .sif files. Now handles file-specific threshold overrides.
    Written by Hephaestus, a Gemini Gem tweaked by JFS
    """
    processed_data = {}
    all_dfs = []
    
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # --- MODIFICATION START ---
        # Check the dictionary for a file-specific override.
        # If the filename is not found, it defaults to the global 'threshold'.
        current_threshold = threshold_overrides.get(filename, threshold)
        # --- MODIFICATION END ---

        try:
            # The 'current_threshold' variable is now used in the call below.
            df, image_data_cps = integrate_sif(
                file_path,
                region=region,
                threshold=current_threshold, # <-- Using the potentially overridden value
                signal=signal,
                pix_size_um=pix_size_um,
                sig_threshold=sig_threshold
            )
            processed_data[filename] = {
                "df": df,
                "image": image_data_cps,
            }
            # Only append if a dataframe was successfully created
            if df is not None:
                all_dfs.append(df)
            
        except Exception as e:
            st.error(f"Error processing {filename}: {e}")
            
    # Combine all dataframes into a single one
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
    else:
        combined_df = pd.DataFrame()
        
    return processed_data, combined_df
