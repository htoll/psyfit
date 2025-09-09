# tools/monomers.py
import streamlit as st
import os, io, tempfile, hashlib, re
import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px

from utils import plot_brightness, plot_histogram, HWT_aesthetic
from tools.process_files import process_files


def _hash_file(uploaded_file):
    uploaded_file.seek(0)
    h = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return h


def _normalize_saved_values(values_iterable):
    """
    Accept values that might be either:
      - (display_name, temp_path) tuples (new format), OR
      - plain path strings from older runs (legacy)
    Return a list of (display_name, temp_path) tuples.
    """
    normalized = []
    for v in values_iterable:
        if isinstance(v, (tuple, list)) and len(v) == 2:
            name, path = v
            normalized.append((str(name), str(path)))
        elif isinstance(v, str):
            name = os.path.basename(v)
            normalized.append((name, v))
    return normalized

from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

def plot_monomer_brightness(
    image_data_cps,
    df,
    show_fits=True,
    plot_brightness_histogram=False,
    normalization=False,
    pix_size_um=0.1,
    cmap='magma',
    single_ucnp_brightness=None
):
    """
    Plot brightness map and overlay Gaussian-fit circles colored by brightness category.

    Categories:
      - Monomers: brightness_fit < 2 * single_ucnp_brightness
      - Dimers:   2 * single_ucnp_brightness <= brightness_fit < 3 * single_ucnp_brightness
      - Multimers:brightness_fit >= 3 * single_ucnp_brightness

    Notes:
      - `single_ucnp_brightness` defaults to np.mean(image_data_cps).
      - Thresholds are compared in the same units as `row['brightness_fit']`.
    """

    # --- figure sizing and normalization ---
    fig_width, fig_height = 5, 5
    scale = fig_width / 5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    norm = LogNorm() if normalization else None
    im = ax.imshow(image_data_cps + 1, cmap=cmap, norm=norm, origin='lower')
    ax.tick_params(axis='both', length=0,
                   labelleft=False, labelright=False,
                   labeltop=False, labelbottom=False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10 * scale)
    cbar.set_label('pps', fontsize=10 * scale)

    # --- thresholds & colors ---
    if single_ucnp_brightness is None:
        single_ucnp_brightness = float(np.mean(image_data_cps))

    single_np_cutoff = 2.0 * single_ucnp_brightness
    dimer_cutoff     = 3.0 * single_ucnp_brightness
    # multimer_cutoff  = 4.0 * single_ucnp_brightness  # kept for reference, not needed for 3 bins

    category_counts = ["Monomers", "Dimers", "Multimers"]

    # Pull colors from your house palette
    palette = HWT_aesthetic()
    try:
        region_colors = list(palette)[:len(category_counts)]
    except Exception:
        # Fallback colors if HWT_aesthetic() doesn't return a list of colors
        region_colors = ['white', 'cyan', 'magenta']

    # Safe-guard if palette shorter than needed
    while len(region_colors) < len(category_counts):
        region_colors.append('white')

    color_map = {
        "Monomers":  region_colors[0],
        "Dimers":    region_colors[1],
        "Multimers": region_colors[2],
    }

    # --- overlay fits ---
    if show_fits:
        for _, row in df.iterrows():
            x_px = row['x_pix']
            y_px = row['y_pix']

            # NOTE: brightness_fit is assumed in pps to match cutoffs.
            brightness_pps = row['brightness_fit']
            brightness_kpps = brightness_pps / 1000.0

            radius_px = 3 * max(row['sigx_fit'], row['sigy_fit']) / pix_size_um

            # Categorize by brightness
            if brightness_pps < single_np_cutoff:
                cat = "Monomers"
            elif brightness_pps < dimer_cutoff:
                cat = "Dimers"
            else:
                cat = "Multimers"

            circle_color = color_map[cat]
            circle = Circle((x_px, y_px), radius_px,
                            color=circle_color, fill=False,
                            linewidth=1.25 * scale, alpha=0.95)
            ax.add_patch(circle)

            ax.text(x_px + 7.5, y_px + 7.5,
                    f"{brightness_kpps:.1f} kpps",
                    color='white', fontsize=7 * scale,
                    ha='center', va='center')

    plt.tight_layout()
    HWT_aesthetic()  # keep your aesthetic call consistent with the original
    return fig


@st.cache_data(show_spinner=False)
def _process_files_cached(saved_records, region, threshold, signal, pix_size_um=0.1, sig_threshold=0.3):
    class _FakeUpload:
        def __init__(self, name, path):
            self.name = name
            self._path = path
        def getbuffer(self):
            with open(self._path, "rb") as f:
                return memoryview(f.read())

    uploads = [_FakeUpload(name, path) for (name, path) in saved_records]
    pf = getattr(process_files, "__wrapped__", None)
    if pf is None:
        raise RuntimeError("process_files.__wrapped__ not found; cannot bypass Streamlit cache.")

    # FORWARD the parameters to process_files.__wrapped__
    return pf(
        uploads,
        region=region,
        threshold=threshold,
        signal=signal,
        pix_size_um=pix_size_um,
        sig_threshold=sig_threshold,
    )


def run():
    col1, col2 = st.columns([1, 2])

    # Persistent state
    if "saved_files" not in st.session_state:
        # key -> (display_name, temp_path)  (legacy may be plain path str)
        st.session_state.saved_files = {}
    if "processed" not in st.session_state:
        st.session_state.processed = None  # (processed_data, combined_df)
    if "selected_file_name" not in st.session_state:
        st.session_state.selected_file_name = None

    with col1:
        uploaded_files = st.file_uploader(
            "Upload .sif file", type=["sif"], accept_multiple_files=True
        )

        # --- SYNC PHASE: make session match current uploader selection ---
        prev_keys = set(st.session_state.saved_files.keys())
        changed = False

        # 1) Add new uploads
        current_keys = set()
        if uploaded_files:
            for f in uploaded_files:
                key = f"{f.name}:{_hash_file(f)}"
                current_keys.add(key)
                if key not in st.session_state.saved_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".sif") as tmp:
                        tmp.write(f.getbuffer())
                        st.session_state.saved_files[key] = (f.name, tmp.name)
                    changed = True

        # 2) Remove files no longer present in the uploader
        #    (also clean up their temp files)
        stale_keys = [k for k in st.session_state.saved_files.keys() if k not in current_keys]
        for k in stale_keys:
            val = st.session_state.saved_files[k]
            name, path = (val if isinstance(val, (tuple, list)) and len(val) == 2 else (os.path.basename(val), val))
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
            del st.session_state.saved_files[k]
            changed = True

        # 3) If the set of saved files changed (added/removed), invalidate results
        if changed or (set(st.session_state.saved_files.keys()) != prev_keys):
            st.session_state.processed = None
            # If selected file no longer exists, clear selection
            current_names = [v[0] if isinstance(v, (tuple, list)) else os.path.basename(v)
                             for v in st.session_state.saved_files.values()]
            if st.session_state.selected_file_name not in current_names:
                st.session_state.selected_file_name = None


        # --- UI to select file & params (based on synced saved_files) ---
        current_values = list(st.session_state.saved_files.values())
        normalized_records = _normalize_saved_values(current_values)
        file_options = [display for (display, _) in normalized_records]

        if file_options:
            default_index = 0
            if st.session_state.selected_file_name in file_options:
                default_index = file_options.index(st.session_state.selected_file_name)
            selected_file_name = st.selectbox(
                "Select sif to display:", options=file_options, index=default_index
            )
            st.session_state.selected_file_name = selected_file_name

            # Parameters (kept to preserve existing UI)
            threshold = st.number_input(
                                        "Threshold", min_value=1, value=1,
                                        help=("Stringency of fit, higher value is more selective:\n"
                                              "- UCNP signal sets absolute peak cut off\n"
                                              "- Dye signal sets sensitivity of blob detection")
                                        )

            signal = st.selectbox(
                                    "Signal", options=["UCNP", "dye"],
                                    help=("Changes detection method:\n"
                                          "- UCNP for high SNR (sklearn peakfinder)\n"
                                          "- dye for low SNR (sklearn blob detection)")
                                    )
            diagram = """ Splits sif into quadrants (256x256 px):  
                                ┌─┬─┐  
                                │ 1 │ 2 │  
                                ├─┼─┤  
                                │ 3 │ 4 │  
                                └─┴─┘
                                """
            region = st.selectbox("Region", options=["1", "2", "3", "4", "all"], help=diagram)

            cmap = st.selectbox("Colormap", options=["magma", "viridis", "plasma", "hot", "gray", "hsv"])
            st.session_state["monomers_cmap"] = cmap

            # PROCESS
            if st.button("Process uploaded files"):
                with st.spinner("Processing…"):
                    saved_records = tuple(normalized_records)
                    processed_data, combined_df = _process_files_cached(
                        saved_records,
                        region=region,
                        threshold=threshold,   
                        signal=signal,         
                    )
                    st.session_state.processed = (processed_data, combined_df)

    # DISPLAY
    if st.session_state.get("processed"):
        processed_data, combined_df = st.session_state.processed

        plot_col1, plot_col2 = col2.columns(2)

        with plot_col1:
            show_fits = st.checkbox("Show fits", value=True)
            normalization = st.checkbox("Log Image Scaling", value = True)

            selected_file_name = st.session_state.get("selected_file_name")
            if not selected_file_name and processed_data:
                selected_file_name = next(iter(processed_data.keys()))
                st.session_state.selected_file_name = selected_file_name

            if selected_file_name in processed_data:
                data_to_plot = processed_data[selected_file_name]
                df_selected = data_to_plot["df"]
                image_data_cps = data_to_plot["image"]
                normalization_to_use = LogNorm() if normalization else None

                fig_image = plot_monomer_brightness(
                    image_data_cps,
                    df_selected,
                    show_fits=show_fits,
                    normalization=normalization,
                    pix_size_um=0.1,
                    cmap=st.session_state.get("monomers_cmap", "magma"),
                    single_ucnp_brightness = st.session_state.get("single_ucnp_brightness")
                )
                st.pyplot(fig_image)

                svg_buffer = io.StringIO()
                fig_image.savefig(svg_buffer, format='svg')
                st.download_button(
                    label="Download PSFs",
                    data=svg_buffer.getvalue(),
                    file_name=f"{selected_file_name}.svg",
                    mime="image/svg+xml"
                )
            else:
                st.error(f"Data for file '{selected_file_name}' not found.")

        with plot_col2:
            if not combined_df.empty:
                brightness_vals = combined_df['brightness_fit'].values
                default_min_val = float(np.min(brightness_vals))
                default_max_val = float(np.max(brightness_vals))

                user_min_val_str = st.text_input("Min Brightness (pps)", value=f"{default_min_val:.2e}")
                user_max_val_str = st.text_input("Max Brightness (pps)", value=f"{default_max_val:.2e}")

                try:
                    user_min_val = float(user_min_val_str); user_max_val = float(user_max_val_str)
                except ValueError:
                    st.warning("Please enter valid numbers (you can use scientific notation like 1e6).")
                    st.stop()

                if user_min_val >= user_max_val:
                    st.warning("Min brightness must be less than max brightness.")
                else:
                    thresholding_method = st.selectbox(
                        "Choose thresholding method:",
                        options=["Automatic (Mu/Sigma)", "Manual"],
                        help="Automatic sets thresholds at 1.5μ, 2.5μ, 3.5μ"
                    )
                    num_bins = st.number_input("# Bins:", value=50)

                    fig_hist, mu, sigma = plot_histogram(
                        combined_df,
                        min_val=user_min_val,
                        max_val=user_max_val,
                        num_bins=num_bins
                    )

                    thresholds = []
                    if thresholding_method == "Automatic (Mu/Sigma)":
                        if mu is not None and sigma is not None:
                            for k in range(1, 4):
                                t = (2 * k + 1) / 2 * mu
                                if user_min_val < t < user_max_val:
                                    thresholds.append(t)
                        else:
                            st.warning("Gaussian fit failed to converge. Cannot perform automatic thresholding.")
                    else:
                        single_ucnp_brightness = st.number_input("Single Particle Brightness", min_value=user_min_val, max_value=user_max_val, value=(user_max_val + user_min_val) / 2)
                        st.session_state["single_ucnp_brightness"] = single_ucnp_brightness

                        t1 = 2 * single_ucnp_brightness #monomer cutoff
                        t2 = 3 * single_ucnp_brightness #dimer cutoff
                        t3 = 4 * single_ucnp_brightness #multimer cutoff
                        thresholds = sorted([t1, t2, t3])

                    if thresholds:
                        fig_hist_final, _, _ = plot_histogram(
                            combined_df,
                            min_val=user_min_val,
                            max_val=user_max_val,
                            num_bins=num_bins,
                            thresholds=thresholds
                        )
                        st.pyplot(fig_hist_final)

                        with plot_col1:
                            thresholds = sorted(set(thresholds))
                            bins_for_pie = [user_min_val] + thresholds + [user_max_val]
                            num_bins_pie = len(bins_for_pie) - 1
                            base_labels = ["Monomers", "Dimers", "Trimers", "Multimers"]
                            labels_for_pie = (
                                base_labels[:num_bins_pie]
                                if num_bins_pie <= len(base_labels)
                                else base_labels + [f"Group {i+1}" for i in range(len(base_labels), num_bins_pie)]
                            )

                            if len(labels_for_pie) != num_bins_pie:
                                st.warning(f"Label/bin mismatch: {len(labels_for_pie)} labels for {num_bins_pie} bins.")
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

                                fig_pie = px.pie(
                                    category_counts,
                                    values='Count',
                                    names='Category',
                                    color_discrete_sequence=plotly_colors
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.pyplot(fig_hist)

        # Summary PSF count bar plot in left column (col1)
        with col1:
            if st.session_state.get("processed"):
                processed = st.session_state.processed[0]
                psf_counts = {os.path.basename(name): len(processed[name]["df"]) for name in processed.keys()}

                def extract_sif_number(filename):
                    m = re.search(r'_([0-9]+)\.sif$', filename)
                    return m.group(1) if m else filename

                file_names = [extract_sif_number(n) for n in psf_counts.keys()]
                counts = list(psf_counts.values())
                mean_count = np.mean(counts) if counts else 0

                fig_count, ax_count = plt.subplots(figsize=(5, 3))
                ax_count.bar(file_names, counts)
                ax_count.axhline(mean_count, color='black', linestyle='--', label=f'Avg = {mean_count:.1f}', linewidth=0.5)
                ax_count.set_ylabel("# Fit PSFs", fontsize=10)
                ax_count.set_xlabel("SIF #", fontsize=10)
                ax_count.legend(fontsize=10)
                ax_count.tick_params(axis='x', labelsize=8)
                ax_count.tick_params(axis='y', labelsize=8)
                HWT_aesthetic()
                st.pyplot(fig_count)
