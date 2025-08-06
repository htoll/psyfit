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
            
        # All dynamic output logic is now within this `with col2:` block
        if st.session_state.analyze_clicked and uploaded_files:
            try:
                processed_data, combined_df = process_files(uploaded_files, region)

                # Now create the sub-columns for the plots below the selectbox
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
                        
                        # User selects thresholding method
                        thresholding_method = st.radio("Choose thresholding method:", ("Automatic (GMM)", "Manual"))

                        num_bins = st.number_input("# Bins:", value=20)
                        
                        min_val = float(np.min(brightness_vals))
                        max_val = float(np.max(brightness_vals))
                        
                        thresholds = []
                        if thresholding_method == "Automatic (GMM)":
                            # Fit GMM to data
                            gmm = GaussianMixture(n_components=2, random_state=0)
                            gmm.fit(brightness_vals.reshape(-1, 1))
                            thresholds = sorted(gmm.means_.flatten())

                            st.write(f"Automatic Thresholds (based on GMM means): {', '.join([f'{t:.2f}' for t in thresholds])}")

                        else: # Manual thresholding
                            # Use Streamlit's range slider to allow manual threshold setting
                            threshold_tuple = st.slider("Set Thresholds", min_value=min_val, max_value=max_val, value=(min_val, (max_val + min_val) / 2, max_val), step=(max_val - min_val) / 1000)
                            thresholds = sorted(list(threshold_tuple))
                        
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

                            # Pie Chart Generation
                            if thresholds:
                                # Categorize data based on thresholds
                                categories = pd.cut(combined_df['brightness_fit'], bins=[min_val] + thresholds + [max_val], right=False, include_lowest=True, labels=[f"Group {i+1}" for i in range(len(thresholds) + 1)])
                                category_counts = categories.value_counts().reset_index()
                                category_counts.columns = ['Category', 'Count']
                                
                                # Create pie chart using plotly
                                fig_pie = px.pie(category_counts, values='Count', names='Category', title='Percentage of Data Points by Threshold')
                                st.plotly_chart(fig_pie, use_container_width=True)


                        else:
                            st.warning("Min brightness is not less than max brightness.")
                    else:
                        st.info("No brightness data to display.")

            except Exception as e:
                st.error(f"Error processing files: {e}")
                st.session_state.analyze_clicked = False
