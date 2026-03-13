import numpy as np
from ..filters.texture_filters import compute_absolute_dog
from ..spectral.wavelet_transform import spectral_feature_map


def generate_texture_feature_map(image: np.ndarray) -> np.ndarray:
    """
    Generate micro-texture feature map using Difference of Gaussians.
    """
    texture_map = compute_absolute_dog(image)
    return texture_map


def compute_texture_spectral_features(image: np.ndarray) -> np.ndarray:
    """
    Generate spectral representation of texture feature map.
    """
    texture_map = generate_texture_feature_map(image)
    spectral_block = spectral_feature_map(texture_map)
    return spectral_block