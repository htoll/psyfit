import numpy as np
import pandas as pd
import os 
import seaborn as sns
import matplotlib as plt


def plot_movie(sif_file, df_dict, show_fits=False, normalization=None, save_format = 'TIFF', univ_minmax=False, cmap = 'grey', title = None):
    required_cols = ['x_pix', 'y_pix', 'sigx_fit', 'sigy_fit', 'brightness_fit']
    all_matched_pairs = []

    n_files = len(sif_files)
    n_cols = min(4, max(1, n_files))   # between 1 and 4, never more than files
    n_rows = int(np.ceil(n_files / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]    
    all_vals = []
    if univ_minmax and normalization is None:
        all_vals = []a
        for frame in sif_file:
            sif_name = sif_file.name
            if sif_name in df_dict:
                all_vals.append(df_dict[sif_name]["image"])
        if all_vals:
            stacked = np.stack(all_vals)
            global_min = stacked.min()
            global_max = stacked.max()
            normalization = Normalize(vmin=global_min, vmax=global_max)
    for i, sif_file in enumerate(sif_files):
        ax = axes[i]
        sif_name = sif_file.name  
        if sif_name not in df_dict:
            st.warning(f"Warning: Data for {sif_name} not found in df_dict. Skipping.")
            continue

        df = df_dict[sif_name]["df"]
        img = df_dict[sif_name]["image"]
        has_fit = all(col in df.columns for col in required_cols)

        colocalized = np.zeros(len(df), dtype=bool) if has_fit else None

        # Colocalization
        if show_fits and has_fit:
            for idx, row in df.iterrows():
                x, y = row['x_pix'], row['y_pix']
                distances = np.sqrt((df['x_pix'] - x) ** 2 + (df['y_pix'] - y) ** 2)
                distances[idx] = np.inf
                if np.any(distances <= colocalization_radius):
                    colocalized[idx] = True
                    closest_idx = distances.idxmin()
                    all_matched_pairs.append({
                        'x': x, 'y': y, 'brightness': row['brightness_fit'],
                        'closest_x': df.at[closest_idx, 'x_pix'],
                        'closest_y': df.at[closest_idx, 'y_pix'],
                        'closest_brightness': df.at[closest_idx, 'brightness_fit'],
                        'distance': distances[closest_idx]
                    })

        im = ax.imshow(img + 1, cmap=cmap, origin='lower', norm=normalization)
        # Only show colorbar on the last subplot in the first row (column n_cols-1)
        if not univ_minmax:
            plt.colorbar(im, ax=ax, label='pps', fraction=0.046, pad=0.04)


        basename = os.path.basename(sif_name)
        match = re.search(r'(\d+)\.sif$', basename)
        file_number = match.group(1) if match else '?'

        # Overlay fits
        if show_fits and has_fit:
            for is_coloc, (_, row) in zip(colocalized, df.iterrows()):
                color = 'lime' if is_coloc else 'white'
                radius_px = 4 * max(row['sigx_fit'], row['sigy_fit']) / 0.1
                circle = Circle((row['x_pix'], row['y_pix']), radius_px, color=color, fill=False, linewidth=1, alpha=0.7)
                ax.add_patch(circle)
                ax.text(row['x_pix'] + 7.5, row['y_pix'] + 7.5,
                        f"{row['brightness_fit']/1000:.1f} kpps",
                        color=color, fontsize=7, ha='center', va='center')

        wrapped_basename = "\n".join(textwrap.wrap(basename, width=25))
        ax.set_title(f"Sif {file_number}\n{wrapped_basename}", fontsize = 10)
        ax.axis('off')

    # Turn off unused axes
    for ax in axes[n_files:]:
        ax.axis('off')
    if univ_minmax:
        colorbar_ax = axes[min(n_cols - 1, n_files - 1)]
        im = colorbar_ax.images[0]  # Get the image from that subplot
        plt.colorbar(im, ax=colorbar_ax, label='pps', fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Show the figure
    st.pyplot(fig)

    

    # Download button
    buf = io.BytesIO()
    
    # Save the figure to the binary buffer
    # Use bbox_inches='tight' for better layouts
    fig.savefig(buf, format=save_format, bbox_inches='tight')
    
    # Get the value from the binary buffer
    plot_data = buf.getvalue()
    buf.close()

    # Define the mime type based on the format
    if save_format.lower() == 'svg':
        mime_type = "image/svg+xml"
    elif save_format.lower() == 'tiff':
        mime_type = "image/tiff"
    elif save_format.lower() == 'png':
        mime_type = "image/png"
    elif save_format.lower() == 'jpeg' or save_format.lower() == 'jpg':
        mime_type = "image/jpeg"
    else:
        mime_type = "application/octet-stream" # Default for unknown formats

    today = date.today().strftime('%Y%m%d')
    download_name = f"sif_grid_{today}.{save_format}"
    
    st.download_button(
        label=f"Download all plots as {save_format.upper()}",
        data=plot_data,
        file_name=download_name,
        mime=mime_type
    )

    # Return colocalization results
    if all_matched_pairs:
        return pd.DataFrame(all_matched_pairs)
    return None



def run():
    col1, col2 = st.columns([1, 2])
    @st.cache_data
    def df_to_csv_bytes(df):
        return df.to_csv(index=False).encode("utf-8")

    with col1:
        st.header("Analyze SIF Movie")
        uploaded_files = st.file_uploader("Upload .sif", type=["sif"], accept_multiple_files=False)
        threshold = st.number_input("Threshold", min_value=0, value=1, help = '''
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
    
        plot_col1, plot_col2 = st.columns(2)
    
        with plot_col1:
            show_fits = st.checkbox("Show fits")
            plot_brightness_histogram = True
            normalization = st.checkbox("Log Image Scaling")
    
            if st.button("Analyze"):
                st.session_state.analyze_clicked = True
    
        if st.session_state.analyze_clicked and uploaded_files:
            try:
                processed_data, combined_df = process_files(uploaded_files, region, threshold = threshold, signal=signal)
    
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
                        cmap=cmap
                    )
    
                    with plot_col1:
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
                        if combined_df is not None and not combined_df.empty:
                            csv_bytes = df_to_csv_bytes(combined_df)
                            st.download_button(
                                label="Download as CSV",
                                data=csv_bytes,
                                file_name=f"{os.path.splitext(selected_file_name)[0]}_compiled.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No compiled data available to download yet.")
    
                    with plot_col2:
                        if plot_brightness_histogram and not combined_df.empty:
                            brightness_vals = combined_df['brightness_fit'].values
                            default_min_val = float(np.min(brightness_vals))
                            default_max_val = float(np.max(brightness_vals))
    
                            user_min_val_str = st.text_input("Min Brightness (pps)", value=f"{default_min_val:.2e}")
                            user_max_val_str = st.text_input("Max Brightness (pps)", value=f"{default_max_val:.2e}")
    
                            try:
                                user_min = float(user_min_val_str)
                                user_max = float(user_max_val_str)
                            except ValueError:
                                st.warning("Please enter valid numbers (you can use scientific notation like 1e6).")
                                return
    
                            num_bins = st.number_input("# Bins:", value=50)
    
                            if user_min < user_max:
                                fig_hist, _, _ = plot_histogram(
                                    combined_df,
                                    min_val=user_min,
                                    max_val=user_max,
                                    num_bins=num_bins
                                )
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
                            else:
                                st.warning("Min greater than max.")
                else:
                    st.error(f"Data for file '{selected_file_name}' not found.")
    
            except Exception as e:
                st.error(f"Error processing files: {e}")
                st.session_state.analyze_clicked = False

   
