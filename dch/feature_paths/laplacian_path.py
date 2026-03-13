import numpy as np
from ..filters.laplacian_filter import compute_absolute_laplacian
from ..spectral.wavelet_transform import spectral_feature_map


def generate_laplacian_feature_map(image: np.ndarray) -> np.ndarray:
    """
    Generate Laplacian feature map capturing fine structural changes.
    """
    laplacian_map = compute_absolute_laplacian(image)
    return laplacian_map


def compute_laplacian_spectral_features(image: np.ndarray) -> np.ndarray:
    """
    Generate spectral representation of Laplacian feature map.
    """
    laplacian_map = generate_laplacian_feature_map(image)
    spectral_block = spectral_feature_map(laplacian_map)
    return spectral_block