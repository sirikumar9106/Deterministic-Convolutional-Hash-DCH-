import cv2
import numpy as np


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur to an image.
    """
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred_image


def compute_difference_of_gaussians(image: np.ndarray) -> np.ndarray:
    """
    Compute Difference of Gaussians (DoG) for texture detection.
    """
    gaussian_small = apply_gaussian_blur(image, kernel_size=3, sigma=0.5)
    gaussian_large = apply_gaussian_blur(image, kernel_size=9, sigma=2.0)
    dog_map = gaussian_small - gaussian_large
    return dog_map


def compute_absolute_dog(image: np.ndarray) -> np.ndarray:
    """
    Compute absolute DoG texture map.
    """
    dog_map = compute_difference_of_gaussians(image)
    absolute_map = np.abs(dog_map)
    return absolute_map