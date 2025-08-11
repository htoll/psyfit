import streamlit as st
import numpy as np
import cv2
import tifffile
import io
import contextlib
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# ==============================================================================
#  HELPER FUNCTIONS (UNCHANGED)
# ==============================================================================

def findOptimalPeaks(image, minPeaks=10, maxPeaks=100, minRoundness=0.8):
    # This function is unchanged.
    print(f"Searching for optimal threshold in image of shape {image.shape}...")
    image_max = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else np.max(image)
    start_thresh, end_thresh = int(image_max * 0.98), int(image_max * 0.20)
    step = -max(1, int(image_max / 200))

    for threshold in range(start_thresh, end_thresh, step):
        binaryMask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        contours, _ = cv2.findContours(binaryMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        goodPeaks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5: continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity >= minRoundness:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    goodPeaks.append((M['m10'] / M['m00'], M['m01'] / M['m00']))
        if minPeaks <= len(goodPeaks) <= maxPeaks:
            print(f"  Success! Found optimal threshold: {threshold} with {len(goodPeaks)} peaks.")
            return np.array(goodPeaks), threshold
    print("  Warning: Could not find an optimal threshold. Returning empty set.")
    return np.array([]), -1

def matchFeatures(imageSource, pointsSource, imageTarget, pointsTarget, patchSize=15, searchRadius=75):
    # This function is unchanged.
    halfPatch = patchSize // 2
    matchedSourcePoints, matchedTargetPoints = [], []
    sourcePadded = np.pad(imageSource, halfPatch, mode='constant')
    targetPadded = np.pad(imageTarget, halfPatch, mode='constant')
    for pS in pointsSource:
        pS_padded = (pS + halfPatch).astype(int)
        sourcePatch = sourcePadded[pS_padded[1]-halfPatch : pS_padded[1]+halfPatch+1, pS_padded[0]-halfPatch : pS_padded[0]+halfPatch+1]
        bestMatchPoint, minSsd = None, float('inf')
        x, y = pS.astype(int)
        yMin, yMax = max(0, y - searchRadius), min(imageTarget.shape[0], y + searchRadius)
        xMin, xMax = max(0, x - searchRadius), min(imageTarget.shape[1], x + searchRadius)
        candidatePoints = [pT for pT in pointsTarget if xMin <= pT[0] < xMax and yMin <= pT[1] < yMax]
        for pT in candidatePoints:
            pT_padded = (pT + halfPatch).astype(int)
            targetPatch = targetPadded[pT_padded[1]-halfPatch : pT_padded[1]+halfPatch+1, pT_padded[0]-halfPatch : pT_padded[0]+halfPatch+1]
            if sourcePatch.shape != targetPatch.shape: continue
            ssd = np.sum((sourcePatch.astype(np.float32) - targetPatch.astype(np.float32))**2)
            if ssd < minSsd:
                minSsd, bestMatchPoint = ssd, pT
        if bestMatchPoint is not None:
            matchedSourcePoints.append(pS)
            matchedTargetPoints.append(bestMatchPoint)
    return np.array(matchedSourcePoints), np.array(matchedTargetPoints)

def warpImagePiecewise(sourceImage, sourcePoints, targetPoints):
    # This function is unchanged.
    targetTri = Delaunay(targetPoints)
    warpedImage = np.zeros(sourceImage.shape, dtype=sourceImage.dtype)
    for simplex in targetTri.simplices:
        srcTriangle = sourcePoints[simplex].astype(np.float32)
        tgtTriangle = targetPoints[simplex].astype(np.float32)
        (x, y, w, h) = cv2.boundingRect(tgtTriangle)
        if w == 0 or h == 0: continue
        tgtCropped = tgtTriangle - np.array([x, y])
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, tgtCropped.astype(np.int32), 255)
        transformMatrix = cv2.getAffineTransform(srcTriangle, tgtTriangle)
        warpedRegion = cv2.warpAffine(sourceImage, transformMatrix, (sourceImage.shape[1], sourceImage.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        roi = warpedImage[y:y+h, x:x+w]
        roi[mask > 0] = warpedRegion[y:y+h, x:x+w][mask > 0]
    return warpedImage

def getQuadrant(image, quad_index):
    # This function is unchanged.
    h, w = image.shape[:2]
    h_half, w_half = h // 2, w // 2
    quadrants = {
        1: image[0:h_half, 0:w_half], 2: image[0:h_half, w_half:w],
        3: image[h_half:h, 0:w_half], 4: image[h_half:h, w_half:w]
    }
    return quadrants.get(quad_index)

def getQuadrantOffset(image_shape, quad_index):
    # This function is unchanged.
    h, w = image_shape[:2]
    h_half, w_half = h // 2, w // 2
    offsets = {
        1: (0, 0), 2: (w_half, 0), 3: (0, h_half), 4: (w_half, h_half)
    }
    return offsets.get(quad_index, (0, 0))
def find_correlated_peaks(peak_sets, radius):
    """
    Filters peaks to find 1:1 correspondences across all active quadrants.
    
    Parameters:
    - peak_sets: Dict where key is quad_index and value is np.array of peak coords.
    - radius: The max distance in pixels to consider a match.
    
    Returns:
    - A new dict with the same structure, but containing only correlated peaks.
    """
    if not peak_sets or len(peak_sets) < 2:
        return peak_sets

    quad_indices = list(peak_sets.keys())
    ref_idx = quad_indices[0]
    source_indices = quad_indices[1:]
    
    ref_peaks = peak_sets[ref_idx]
    if len(ref_peaks) == 0:
        return {idx: np.array([]) for idx in quad_indices}

    # Use KD-Trees for efficient nearest neighbor searching
    source_trees = {idx: cKDTree(peak_sets[idx]) for idx in source_indices if len(peak_sets[idx]) > 0}
    
    # Also need a tree for the reference peaks for the mutual check
    ref_tree = cKDTree(ref_peaks)
    
    correlated_indices = {idx: [] for idx in quad_indices}

    for i, p_ref in enumerate(ref_peaks):
        is_fully_correlated = True
        potential_matches = {ref_idx: i}

        for src_idx in source_indices:
            if src_idx not in source_trees:
                is_fully_correlated = False
                break

            # Find nearest source peak to the reference peak
            dist, j = source_trees[src_idx].query(p_ref, k=1)
            
            if dist > radius:
                is_fully_correlated = False
                break
            
            # Mutual check: find nearest reference peak to the source peak
            p_src = peak_sets[src_idx][j]
            _, mutual_i = ref_tree.query(p_src, k=1)

            if i == mutual_i:
                potential_matches[src_idx] = j
            else:
                is_fully_correlated = False
                break
        
        if is_fully_correlated:
            for idx, peak_idx in potential_matches.items():
                correlated_indices[idx].append(peak_idx)
    
    # Build the final filtered peak sets
    filtered_peak_sets = {
        idx: peak_sets[idx][correlated_indices[idx]] for idx in quad_indices
    }
    
    return filtered_peak_sets
def run_registration_workflow(imgRegistration, imgData, activeQuads, peakParams, matchParams, correlationRadius):
    """Encapsulates the entire registration process."""
    print("\nStep 1: Finding initial peaks in all active quadrants...")
    initial_peaks = {}
    for quad_idx in activeQuads:
        quad_img = getQuadrant(imgRegistration, quad_idx)
        peaks, _ = findOptimalPeaks(quad_img, **peakParams)
        if len(peaks) > 0:
            offset = getQuadrantOffset(imgRegistration.shape, quad_idx)
            initial_peaks[quad_idx] = peaks + offset
        else:
            initial_peaks[quad_idx] = np.array([])
        print(f"  Quadrant {quad_idx}: Found {len(initial_peaks[quad_idx])} peaks.")
    
    print(f"\nStep 2: Finding 1:1 correlated peaks within a {correlationRadius} pixel radius...")
    correlated_peaks = find_correlated_peaks(initial_peaks, correlationRadius)
    
    # --- The rest of the workflow now uses the CORRELATED peaks ---
    all_detected_peaks = np.vstack(list(correlated_peaks.values())) if correlated_peaks else np.array([])
    
    refQuadIndex = activeQuads[0]
    refPeaksGlobal = correlated_peaks.get(refQuadIndex, np.array([]))
    if len(refPeaksGlobal) < 3:
        print("Error: Not enough correlated peaks found to perform registration. Try increasing the correlation radius.")
        return None, all_detected_peaks

    sourceQuadIndices = [q for q in activeQuads if q != refQuadIndex]
    
    transformations = {}
    print("\nStep 3: Matching feature patches for final transform calculation...")
    for srcQuadIndex in sourceQuadIndices:
        srcPeaksGlobal = correlated_peaks.get(srcQuadIndex, np.array([]))
        if len(srcPeaksGlobal) == 0:
            continue
            
        print(f"  Matching features between quadrant {srcQuadIndex} and {refQuadIndex}...")
        matchedSourcePoints, matchedTargetPoints = matchFeatures(imgRegistration, srcPeaksGlobal, imgRegistration, refPeaksGlobal, **matchParams)
        print(f"  Found {len(matchedSourcePoints)} final matching points.")

        if len(matchedSourcePoints) >= 3:
            transformations[srcQuadIndex] = (matchedSourcePoints, matchedTargetPoints)
        else:
            print(f"  Not enough patch matches to calculate transform for quadrant {srcQuadIndex}.")

    # --- Warping and Saving ---
    # (This part is unchanged)
    print("\nStep 4: Applying transformations to data image and building output file...")
    quad_h, quad_w = imgData.shape[0] // 2, imgData.shape[1] // 2
    output_stack = np.zeros((4, quad_h, quad_w), dtype=imgData.dtype)
    
    refDataQuad = getQuadrant(imgData, refQuadIndex)
    output_stack[refQuadIndex - 1] = refDataQuad
    print(f"Quadrant {refQuadIndex} (reference) placed in channel {refQuadIndex - 1}.")
    
    for srcQuadIndex, (srcPoints, tgtPoints) in transformations.items():
        print(f"Warping data for quadrant {srcQuadIndex}...")
        warpedFullDataImage = warpImagePiecewise(imgData, srcPoints, tgtPoints)
        ref_x, ref_y = getQuadrantOffset(imgData.shape, refQuadIndex)
        warpedData = warpedFullDataImage[ref_y:ref_y+quad_h, ref_x:ref_x+quad_w]
        output_stack[srcQuadIndex - 1] = warpedData
        print(f"Warped data from quadrant {srcQuadIndex} placed in channel {srcQuadIndex - 1}.")
    
    print("\nProcessing complete.")
    return output_stack, all_detected_peaks

# ==============================================================================
#  STREAMLIT GUI (MODIFIED TO ADD CORRELATION RADIUS SLIDER)
# ==============================================================================
def run():
    st.set_page_config(layout="wide")
    st.title("🔬 Quadrant-Based Image Cross-Registration")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Registration Setup")
        f_reg = st.file_uploader("Upload Registration Image (with fiducials)", type=["tif", "tiff"])
        with st.expander("Set Registration Parameters", expanded=True):
            activeQuads = st.multiselect("Active quadrants (first is reference)", options=[1, 2, 3, 4], default=[1, 2, 3, 4])
            
            st.markdown("###### Peak Detection")
            minPeaks = st.slider("Min desired peaks (per quad)", 5, 500, 10, key="min_peaks")
            maxPeaks = st.slider("Max desired peaks (per quad)", 10, 1000, 200, key="max_peaks")
            minRoundness = st.slider("Min peak roundness", 0.0, 1.0, 0.75, 0.05, key="roundness")
    
            # --- NEW WIDGET ---
            st.markdown("###### Geometric Correlation")
            correlationRadius = st.slider("Peak correlation radius (pixels)", 1, 20, 4, key="correlation_radius")
    
            st.markdown("###### Feature Matching")
            patchSize = st.slider("Match patch size", 5, 51, 21, step=2, key="patch_size")
            searchRadius = st.slider("Match search radius (pixels)", 10, 500, 100, key="search_radius")
    
    with col2:
        st.subheader("2. Data Setup")
        f_data = st.file_uploader("Upload Data Image (to be warped)", type=["tif", "tiff"])
    
    st.markdown("---")
    
    if f_reg and f_data:
        if st.button("🚀 Run Registration", use_container_width=True):
            imgRegistration = tifffile.imread(io.BytesIO(f_reg.read()))
            imgData = tifffile.imread(io.BytesIO(f_data.read()))
            peakParams = {'minPeaks': minPeaks, 'maxPeaks': maxPeaks, 'minRoundness': minRoundness}
            matchParams = {'patchSize': patchSize, 'searchRadius': searchRadius}
    
            log_stream = io.StringIO()
            with st.spinner("Registration in progress... this may take a moment."):
                with contextlib.redirect_stdout(log_stream):
                    output_stack, detected_peaks = run_registration_workflow(
                        imgRegistration, imgData, activeQuads, peakParams, matchParams, correlationRadius
                    )
            
            st.session_state['output_stack'] = output_stack
            st.session_state['log'] = log_stream.getvalue()
            st.session_state['detected_peaks'] = detected_peaks
            st.session_state['registration_image'] = imgRegistration
    
    # --- Results Display (Unchanged) ---
    if 'output_stack' in st.session_state and st.session_state['output_stack'] is not None:
        st.success("✅ Registration complete!")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.subheader("Correlated Peaks")
            fig_peaks, ax_peaks = plt.subplots(figsize=(8, 8))
            ax_peaks.imshow(st.session_state['registration_image'], cmap='gray')
            peaks = st.session_state['detected_peaks']
            if peaks is not None and len(peaks) > 0:
                ax_peaks.plot(peaks[:, 0], peaks[:, 1], 'r+', markersize=8, label=f'{len(peaks)} correlated peaks')
            ax_peaks.set_title("Correlated Fiducials on Registration Image")
            ax_peaks.axis('off')
            ax_peaks.legend()
            st.pyplot(fig_peaks)
            with st.expander("Show Processing Log"):
                st.text(st.session_state['log'])
            buffer = io.BytesIO()
            tifffile.imwrite(buffer, st.session_state['output_stack'], imagej=True)
            buffer.seek(0)
            st.download_button(
                label="💾 Download Registered 4-Channel TIFF", data=buffer, file_name="registered_output.tif", mime="image/tiff", use_container_width=True
            )
        with res_col2:
            st.subheader("Registered Data Channels")
            stack = st.session_state['output_stack']
            fig_channels, axes_channels = plt.subplots(2, 2, figsize=(8, 8))
            for i, ax in enumerate(axes_channels.flat):
                channel_data = stack[i]
                if channel_data.max() > 0:
                    norm_data = cv2.normalize(channel_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    ax.imshow(norm_data, cmap='viridis')
                else:
                    ax.imshow(np.zeros_like(channel_data), cmap='gray', vmin=0, vmax=255)
                ax.set_title(f"Channel {i+1} (From Quad {i+1})")
                ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig_channels)
    elif 'log' in st.session_state:
        st.error("Registration failed. Please check the log for details.")
        with st.expander("Show Processing Log"):
            st.text(st.session_state['log'])
