import numpy as np
import pywt


def compute_wavelet_decomposition(feature_map: np.ndarray):
    """
    Perform a single-level Discrete Wavelet Transform.
    """
    coefficients = pywt.dwt2(feature_map, 'haar')
    LL, (LH, HL, HH) = coefficients
    return LL, LH, HL, HH


def extract_low_frequency_block(ll_band: np.ndarray, block_size: int = 16) -> np.ndarray:
    """
    Extract the low-frequency block from the LL band.
    """
    height, width = ll_band.shape

    if block_size > height or block_size > width:
        raise ValueError("Block size larger than LL band dimensions")
    
    low_frequency_block = ll_band[0:block_size, 0:block_size]
    return low_frequency_block


def spectral_feature_map(feature_map: np.ndarray, block_size: int = 16) -> np.ndarray:
    """
    Complete spectral processing pipeline.
    """
    LL, LH, HL, HH = compute_wavelet_decomposition(feature_map)
    low_frequency_block = extract_low_frequency_block(LL, block_size)
    return low_frequency_block