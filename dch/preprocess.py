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


def convert_to_lh_channel(image: np.ndarray) -> np.ndarray:
    """
    Replaces grayscale conversion with a perceptually richer single-channel map.

    Combines L (lightness) and H (hue) from HLS color space at a 4:1 ratio:

      L contributes 80% — preserves structural gradients that Sobel, Laplacian,
      and DoG filters depend on. This is what keeps geometric invariance working.
      Grayscale was essentially just L, so this preserves all existing behavior.

      H contributes 20% — introduces color discrimination so that two images
      with similar structure but different color content produce different
      feature maps. Not enough to dominate, enough to be visible.

    H is multiplied by S (saturation) before mixing. This suppresses hue values
    from near-grey or desaturated pixels which are numerically unreliable.
    A near-white or near-black pixel can report any hue randomly — multiplying
    by S zeroes out that noise where saturation is weak.

    Why not H alone:
        When false positive pairs share the same dominant hue (orange cat vs
        orange sunset — both orange), H alone adds nothing. The L channel is
        what differentiates their internal structural distributions. Together
        L+H encodes both structure and color mood in one channel without
        increasing hash size or adding pipeline steps.

    Output: single-channel float64 image, range [0, 1].
    Identical shape to what grayscale produced — rest of pipeline unchanged.
    """

    # Convert BGR to HLS. OpenCV channel order: [H, L, S]
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    h_channel = hls_image[:, :, 0].astype(np.float64)  # Hue:        0-180
    l_channel = hls_image[:, :, 1].astype(np.float64)  # Lightness:  0-255
    s_channel = hls_image[:, :, 2].astype(np.float64)  # Saturation: 0-255

    # Normalize each channel independently to [0, 1]
    l_normalized = l_channel / 255.0
    h_normalized = h_channel / 180.0  # OpenCV hue is 0-180 not 0-360
    s_normalized = s_channel / 255.0

    # Suppress unreliable hue in low-saturation regions
    h_weighted = h_normalized * s_normalized

    # 4:1 ratio — L dominates structure, H adds color discriminability
    combined = (0.8 * l_normalized) + (0.2 * h_weighted)

    # Already in [0, 1] — no further normalization needed
    return combined


def resize_image(image: np.ndarray, target_size: int = 256) -> np.ndarray:
    """
    Resize the image to a square resolution.
    """

    resized_image = cv2.resize(
        image,
        (target_size, target_size),
        interpolation=cv2.INTER_AREA
    )
    return resized_image


def preprocess_image(image_path: str, target_size: int = 256) -> np.ndarray:
    """
    Complete preprocessing pipeline.

    normalize_image is intentionally removed from this pipeline.
    convert_to_lh_channel already outputs float64 in [0, 1].
    Calling a /255 normalization after it would crush all values
    to near-zero and destroy the feature signal entirely.
    """

    image = load_image(image_path)
    lh_image = convert_to_lh_channel(image)         # replaces grayscale, already [0, 1]
    resized_image = resize_image(lh_image, target_size)
    return resized_image