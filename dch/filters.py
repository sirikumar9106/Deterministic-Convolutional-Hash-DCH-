import cv2
import numpy as np


def sobel_x(image: np.ndarray) -> np.ndarray:
    """
    Compute Sobel gradient in X direction.
    """
    grad = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    return grad


def sobel_y(image: np.ndarray) -> np.ndarray:
    """
    Compute Sobel gradient in Y direction.
    """
    grad = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return grad


def laplacian(image: np.ndarray) -> np.ndarray:
    """
    Compute Laplacian edge map.
    """
    lap = cv2.Laplacian(image, cv2.CV_64F)
    return lap


def gabor_filter(image: np.ndarray,
                 ksize: int = 21,
                 sigma: float = 5.0,
                 theta: float = 0.0,
                 lambd: float = 10.0,
                 gamma: float = 0.5) -> np.ndarray:
    """
    Apply a Gabor filter.
    """
    kernel = cv2.getGaborKernel(
        (ksize, ksize),
        sigma,
        theta,
        lambd,
        gamma,
        0,
        ktype=cv2.CV_32F
    )

    filtered = cv2.filter2D(
        image,
        cv2.CV_64F,
        kernel
    )

    return filtered


def compute_feature_stack(image: np.ndarray):
    """
    Generate feature stack instead of collapsing features.

    Returns list of feature maps.
    """

    image = image.astype(np.float64)

    sx = np.abs(sobel_x(image))
    sy = np.abs(sobel_y(image))
    lap = np.abs(laplacian(image))

    g0 = np.abs(gabor_filter(image, theta=0))
    g90 = np.abs(gabor_filter(image, theta=np.pi / 2))

    feature_stack = [
        sx,
        sy,
        lap,
        g0,
        g90
    ]

    normalized_stack = []

    for fmap in feature_stack:
        max_val = np.max(fmap)
        if max_val > 0:
            fmap = fmap / max_val
        normalized_stack.append(fmap)

    return normalized_stack