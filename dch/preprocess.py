import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """ Load an image from disk.

    Args:
        image_path: Path to image file

    Returns:
        image as numpy array (BGR) """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image from path: {image_path}")
    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """ Convert BGR image to grayscale.

    Args:
        image: BGR image array

    Returns:
        grayscale image
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def resize_image(image: np.ndarray, size: int = 256) -> np.ndarray:
    """
    Resize image to a square resolution.

    Args:
        image: grayscale image
        size: output dimension (default 256)

    Returns: resized image """

    resized = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_AREA
    )
    return resized


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image intensity to range [0,1].
    Args: image: grayscale image

    Returns: normalized float image"""

    image_float = image.astype(np.float32)
    normalized = image_float / 255.0
    return normalized


def preprocess_image(image_path: str, size: int = 256) -> np.ndarray:
    """ Full preprocessing pipeline.

    Steps:
        1. Load image
        2. Convert to grayscale
        3. Resize
        4. Normalize

    Args:
        image_path: input image path
        size: resize dimension

    Returns: normalized grayscale image """

    image = load_image(image_path)
    gray = convert_to_grayscale(image)
    resized = resize_image(gray, size)
    normalized = normalize_image(resized)
    
    return normalized