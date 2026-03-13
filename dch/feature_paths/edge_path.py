import numpy as np
from ..filters.sobel_filters import compute_absolute_sobel_sum
from ..spectral.wavelet_transform import spectral_feature_map


def generate_edge_feature_map(image: np.ndarray) -> np.ndarray:
    """
    Generate direction-agnostic edge map using Sobel filters.
    """
    edge_map = compute_absolute_sobel_sum(image)
    return edge_map


def compute_edge_spectral_features(image: np.ndarray) -> np.ndarray:
    """
    Generate spectral representation of the edge feature map.
    """
    edge_map = generate_edge_feature_map(image)
    spectral_block = spectral_feature_map(edge_map)
    return spectral_block