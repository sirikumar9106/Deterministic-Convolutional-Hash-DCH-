import numpy as np
import pywt
import cv2

def make_d4_symmetric(matrix: np.ndarray) -> np.ndarray:
    """
    Creates a 'Kaleidoscope' effect by averaging the matrix 
    across all 8 possible 90-degree rotations and reflections.
    """
    s1 = matrix
    s2 = np.rot90(matrix, 1)
    s3 = np.rot90(matrix, 2)
    s4 = np.rot90(matrix, 3)
    s5 = np.fliplr(matrix)
    s6 = np.flipud(matrix)
    s7 = np.fliplr(np.rot90(matrix, 1)) # Transpose
    s8 = np.flipud(np.rot90(matrix, 1)) # Anti-transpose
    
    # Average them all together
    symmetric_matrix = (s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8) / 8.0
    return symmetric_matrix


def spectral_feature_map(feature_map: np.ndarray, target_dimension: int = 8):
    """
    Parallel Logic: Returns two maps.
    1. Spatial Map: Good for crops and slight shifts.
    2. Invariant Map: Symmetrical map for mirrors and rotations.
    """
    # 1. Perform DWT to get high-frequency detail
    coeffs = pywt.dwt2(feature_map, 'haar')
    LL, (LH, HL, HH) = coeffs

    # 2. Magnitude Map: The energy where the actual features live
    magnitude_map = np.sqrt(np.power(LH, 2) + np.power(HL, 2) + np.power(HH, 2))

    # Path A: Spatial Map (The 'Where' - Rigid)
    spatial_map = cv2.resize(magnitude_map, (target_dimension, target_dimension), 
                             interpolation=cv2.INTER_AREA)

    # Path B: Invariant Map (The 'Kaleidoscope' - Symmetric)
    # By making it perfectly symmetrical, a flipped or 90-degree rotated 
    # image will result in the exact same spatial distribution of energy here.
    inv_map = make_d4_symmetric(spatial_map)

    return spatial_map, inv_map