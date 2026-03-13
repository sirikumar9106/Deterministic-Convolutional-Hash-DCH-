import cv2
import numpy as np


def compute_sobel_x(image: np.ndarray) -> np.ndarray:
    """
    Compute Sobel gradient along the X direction.
    """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    return sobel_x


def compute_sobel_y(image: np.ndarray) -> np.ndarray:
    """
    Compute Sobel gradient along the Y direction.
    """
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return sobel_y


def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Compute orientation-agnostic edge magnitude.
    """
    sobel_x = compute_sobel_x(image)
    sobel_y = compute_sobel_y(image)
    magnitude = np.sqrt((sobel_x ** 2) + (sobel_y ** 2))
    return magnitude


def compute_absolute_sobel_sum(image: np.ndarray) -> np.ndarray:
    """
    Compute |SobelX| + |SobelY| edge map.
    """

    sobel_x = compute_sobel_x(image)
    sobel_y = compute_sobel_y(image)
    abs_x = np.abs(sobel_x)
    abs_y = np.abs(sobel_y)
    edge_map = abs_x + abs_y
    return edge_map