import cv2
import numpy as np


def compute_laplacian(image: np.ndarray) -> np.ndarray:
    """
    Apply Laplacian filter to capture fine structural changes.
    """
    laplacian_map = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian_map


def compute_absolute_laplacian(image: np.ndarray) -> np.ndarray:
    """
    Compute absolute Laplacian to remove sign sensitivity.
    """
    laplacian_map = compute_laplacian(image)
    absolute_map = np.abs(laplacian_map)
    return absolute_map