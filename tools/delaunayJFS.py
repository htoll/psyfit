import streamlit as st
import numpy as np
import cv2
import tifffile
import io
import contextlib
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, cKDTree

# ==============================================================================
#  HELPER FUNCTIONS (UNCHANGED FROM PREVIOUS VERSION)
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
    quadrants = {1: image[0:h_half, 0:w_half], 2: image[0:h_half, w_half:w], 3: image[h_half:h, 0:w_half], 4: image[h_half:h, w_half:w]}
    return quadrants.get(quad_index)

def getQuadrantOffset(image_shape, quad_index):
    # This function is unchanged.
    h, w = image_shape[:2]
    h_half, w_half = h // 2, w // 2
    offsets = {1: (0, 0),
