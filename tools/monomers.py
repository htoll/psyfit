import streamlit as st
import os
import io
from utils import integrate_sif, plot_brightness, plot_histogram, HWT_aesthetic
from tools.process_files import process_files
from matplotlib.colors import LogNorm
import numpy as np
from sklearn.mixture import GaussianMixture
import plotly.express as px
import pandas as pd
import matplotlib.colors as mcolors


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
                        
                        # User inputs for x-axis limits
                        default_min_val = float(np.min(brightness_vals))
                        default_max_val = float(np.max(brightness_vals))
                        user_min_val = st.number_input("Min Brightness (pps)", value=default_min_val)
                        user_max_val = st.number_input("Max Brightness (pps)", value=default_max_val)
                
                        if user_min_val >= user_max_val:
                            st.warning("Min brightness must be less than max brightness.")
                        else:
                            thresholding_method = st.radio("Choose thresholding method:", ("Automatic (Mu/Sigma)", "Manual"))
                            num_bins = st.number_input("# Bins:", value=20)
                
                            # Generate the histogram plot first to get mu and sigma
                            fig_hist, mu, sigma = plot_histogram(combined_df, min_val=user_min_val, max_val=user_max_val, num_bins=num_bins)
                
                            thresholds = []
                            if thresholding_method == "Automatic (Mu/Sigma)":
                                if mu is not None and sigma is not None:
                                    max_multiplicity = 4  # up to tetramers
                                    thresholds = []
                            
                                    for k in range(1, max_multiplicity):
                                        t = (2 * k + 1) / 2 * mu
                                        if user_min_val < t < user_max_val:
                                            thresholds.append(t)
                            
                                else:
                                    st.warning("Gaussian fit failed to converge. Cannot perform automatic thresholding.")

                            
                            else: # Manual thresholding
                                threshold1 = st.number_input("Threshold 1", min_value=user_min_val, max_value=user_max_val, value=(user_max_val + user_min_val) / 2, step=(user_max_val - user_min_val) / 1000)
                                threshold2 = st.number_input("Threshold 2", min_value=user_min_val, max_value=user_max_val, value=user_max_val * 0.75, step=(user_max_val - user_min_val) / 1000)
                                threshold3 = st.number_input("Threshold 3", min_value=user_min_val, max_value=user_max_val, value=user_max_val * 0.9, step=(user_max_val - user_min_val) / 1000)
                                thresholds = sorted([threshold1, threshold2, threshold3])
                

                
                            if thresholds:
                                fig_hist_final, _, _ = plot_histogram(combined_df, min_val=user_min_val, max_val=user_max_val, num_bins=num_bins, thresholds=thresholds)
                                st.pyplot(fig_hist_final)
                                # Prepare bins
                                thresholds = sorted(set(thresholds))
                                bins_for_pie = [user_min_val] + thresholds + [user_max_val]
                                num_bins = len(bins_for_pie) - 1
                                
                                # Generate correct number of labels
                                base_labels = ["Monomers", "Dimers", "Trimers", "Multimers"]
                                if num_bins <= len(base_labels):
                                    labels_for_pie = base_labels[:num_bins]
                                else:
                                    # Extend with generic names if more bins are created
                                    labels_for_pie = base_labels + [f"Group {i+1}" for i in range(len(base_labels), num_bins)]
                                
                                if len(labels_for_pie) != num_bins:
                                    st.warning(f"Label/bin mismatch: {len(labels_for_pie)} labels for {num_bins} bins. Cannot categorize.")
                                else:
                                    categories = pd.cut(
                                        combined_df['brightness_fit'],
                                        bins=bins_for_pie,
                                        right=False,
                                        include_lowest=True,
                                        labels=labels_for_pie
                                    )
                                    category_counts = categories.value_counts().reset_index()
                                    category_counts.columns = ['Category', 'Count']
                                    palette = HWT_aesthetic()
                                    region_colors = palette[:len(category_counts)]
                                    plotly_colors = [mcolors.to_hex(c) for c in region_colors]

                                    fig_pie = px.pie(category_counts, values='Count', names='Category', 
                                                     title='Percentage of Data Points by Threshold',
                                                    color_discrete_sequence=plotly_colors)
                                    st.plotly_chart(fig_pie, use_container_width=True)
                            else:
                                st.pyplot(fig_hist) # Plot the initial histogram without thresholds

            except Exception as e:
                st.error(f"Error processing files: {e}")
                st.session_state.analyze_clicked = False
