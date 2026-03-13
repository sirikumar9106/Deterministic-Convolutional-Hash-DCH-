import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk using OpenCV.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Unable to load image at path: {image_path}")

    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale.
    """

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image


def resize_image(image: np.ndarray, target_size: int = 256) -> np.ndarray:
    """
    Resize the image to a square resolution.
    """

    resized_image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized_image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to the range [0, 1].
    """

    image_float = image.astype(np.float64)
    normalized_image = image_float / 255.0
    return normalized_image


def preprocess_image(image_path: str, target_size: int = 256) -> np.ndarray:
    """
    Complete preprocessing pipeline.
    """
    image = load_image(image_path)
    grayscale_image = convert_to_grayscale(image)
    resized_image = resize_image(grayscale_image, target_size)
    normalized_image = normalize_image(resized_image)
    return normalized_image