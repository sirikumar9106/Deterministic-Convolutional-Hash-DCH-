import numpy as np
import pywt
import cv2

def compute_wavelet_decomposition(feature_map: np.ndarray):
    """
    Perform a single-level Discrete Wavelet Transform.
    """
    coefficients = pywt.dwt2(feature_map, 'haar')
    LL, (LH, HL, HH) = coefficients
    return LL, LH, HL, HH

def spectral_feature_map(feature_map: np.ndarray, target_dimension: int = 8) -> np.ndarray:
    """
    FIXED: Instead of slicing LL[0:16], we capture the Magnitude of the 
    feature map's energy and resize it to the target dimension.
    """
    # 1. Decompose the feature map
    LL, (LH, HL, HH) = pywt.dwt2(feature_map, 'haar')

    # 2. FEATURE CONTEXT: In derivative maps (Sobel/Laplacian), 
    # the energy is in the high-frequency bands (LH, HL, HH).
    # We compute the Magnitude Map: M = sqrt(LH^2 + HL^2 + HH^2)
    magnitude_map = np.sqrt(np.power(LH, 2) + np.power(HL, 2) + np.power(HH, 2))

    # 3. RESIZE (Critical Fix): We resize the magnitude map to a small grid
    # to ensure the hash represents the WHOLE image, not just a corner.
    spectral_block = cv2.resize(magnitude_map, (target_dimension, target_dimension), 
                                interpolation=cv2.INTER_AREA)
    
    return spectral_block