import streamlit as st
import os
import io
from utils import integrate_sif, plot_brightness, plot_histogram
from tools.process_files import process_files
from matplotlib.colors import LogNorm
import numpy as np
from sklearn.mixture import GaussianMixture
import plotly.express as px
import pandas as pd

def run():
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Analyze SIF Files")
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

        signal = st.selectbox("Signal", options=["UCNP", "dye"], help= '''Changes detection method:  
                                            - UCNP for high SNR (sklearn peakfinder)  
                                            - dye for low SNR (sklearn blob detection)''')
        cmap = st.selectbox("Colormap", options = ["magma", 'viridis', 'plasma', 'hot', 'gray', 'hsv'])


    with col2:
        if "analyze_clicked" not in st.session_state:
            st.session_state.analyze_clicked = False

        show_fits = st.checkbox("Show fits")
        normalization = st.checkbox("Log Image Scaling")

        if st.button("Analyze"):
            st.session_state.analyze_clicked = True
            
        if st.session_state.analyze_clicked and uploaded_files:
            try:
                processed_data, combined_df = process_files(uploaded_files, region)

                plot_col1, plot_col2 = st.columns(2)

                with plot_col1:
                    if len(uploaded_files) > 1:
                        file_options = [f.name for f in uploaded_files]
                        selected_file_name = st.selectbox("Select sif to display:", options=file_options)
                    else:
                        selected_file_name = uploaded_files[0].name
                    
                    if selected_file_name in processed_data:
                        data_to_plot = processed_data[selected_file_name]
                        df_selected = data_to_plot["df"]
                        image_data_cps = data_to_plot["image"]
                        normalization_to_use = LogNorm() if normalization else None
                        fig_image = plot_brightness(
                            image_data_cps,
                            df_selected,
                            show_fits=show_fits,
                            normalization=normalization_to_use,
                            pix_size_um=0.1,
                            cmap = cmap
                        )
                        st.pyplot(fig_image)

                        svg_buffer = io.StringIO()
                        fig_image.savefig(svg_buffer, format='svg')
                        svg_data = svg_buffer.getvalue()
                        svg_buffer.close()
                        st.download_button(
                            label="Download PSFs",
                            data=svg_data,
                            file_name=f"{selected_file_name}.svg",
                            mime="image/svg+xml"
                        )
                        
                with plot_col2:
                    st.subheader("Brightness Histogram & Thresholding")
                    if not combined_df.empty:
                        brightness_vals = combined_df['brightness_fit'].values
                        
                        thresholding_method = st.radio("Choose thresholding method:", ("Automatic (Mu/Sigma)", "Manual"))
                
                        num_bins = st.number_input("# Bins:", value=20)
                        
                        min_val = float(np.min(brightness_vals))
                        max_val = float(np.max(brightness_vals))
                        
                        thresholds = []
                        if thresholding_method == "Automatic (Mu/Sigma)":
                            mu = np.mean(brightness_vals)
                            # Create three thresholds for Monomers, Dimers, and Trimers
                            threshold_1 = (mu + 2 * mu) / 2  # Threshold between Monomers and Dimers
                            threshold_2 = (2 * mu + 3 * mu) / 2  # Threshold between Dimers and Trimers
                            threshold_3 = (3 * mu + 4 * mu) / 2  # Threshold between Trimers and Multimers
                            
                            # Filter thresholds to ensure they are within the data range
                            thresholds = [t for t in [threshold_1, threshold_2, threshold_3] if min_val < t < max_val]
                            
                            st.write(f"Automatic Thresholds (based on Mu/Sigma): {', '.join([f'{t:.2f}' for t in thresholds])}")
                
                        else: # Manual thresholding
                            threshold1 = st.number_input("Threshold 1", min_value=min_val, max_value=max_val, value=(max_val + min_val) / 2, step=(max_val - min_val) / 1000)
                            threshold2 = st.number_input("Threshold 2", min_value=min_val, max_value=max_val, value=max_val * 0.75, step=(max_val - min_val) / 1000)
                            threshold3 = st.number_input("Threshold 3", min_value=min_val, max_value=max_val, value=max_val * 0.9, step=(max_val - min_val) / 1000)
                            thresholds = sorted([threshold1, threshold2, threshold3])
                
                        if min_val < max_val:
                            fig_hist = plot_histogram(combined_df, min_val=min_val, max_val=max_val, num_bins=num_bins, thresholds=thresholds)
                            st.pyplot(fig_hist)
                            
                            svg_buffer_hist = io.StringIO()
                            fig_hist.savefig(svg_buffer_hist, format='svg')
                            svg_data_hist = svg_buffer_hist.getvalue()
                            svg_buffer_hist.close()
                            
                            st.download_button(
                                label="Download histogram",
                                data=svg_data_hist,
                                file_name="combined_histogram.svg",
                                mime="image/svg+xml"
                            )
                
                            if thresholds:
                                # Define labels for the pie chart
                                labels_for_pie = ["Monomers", "Dimers", "Trimers", "Multimers"]
                
                                # Pad labels if more thresholds are added manually
                                num_categories = len(thresholds) + 1
                                if len(labels_for_pie) < num_categories:
                                     labels_for_pie = labels_for_pie + [f"Group {i+1}" for i in range(len(labels_for_pie), num_categories)]
                
                                bins_for_pie = [min_val] + sorted(thresholds) + [max_val]
                                
                                categories = pd.cut(combined_df['brightness_fit'], bins=bins_for_pie, right=False, include_lowest=True, labels=labels_for_pie)
                                category_counts = categories.value_counts().reset_index()
                                category_counts.columns = ['Category', 'Count']
                                
                                fig_pie = px.pie(category_counts, values='Count', names='Category', title='Estimated Monomer Proportion')
                                st.plotly_chart(fig_pie, use_container_width=True)
                
                        else:
                            st.warning("Min brightness is not less than max brightness.")
                    else:
                        st.info("No brightness data to display.")


            except Exception as e:
                st.error(f"Error processing files: {e}")
                st.session_state.analyze_clicked = False
