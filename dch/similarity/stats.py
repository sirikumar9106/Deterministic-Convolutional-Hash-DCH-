import cv2
import numpy as np
from scipy.stats import skew


def compute_structural_stats(image_path: str) -> np.ndarray:
    """
    Compute statistical moments of Sobel and Laplacian feature maps.

    Returns a 6-element vector:
        [sobel_mean, sobel_variance, sobel_skewness,
         lap_mean,   lap_variance,   lap_skewness]

    These six numbers form a statistical fingerprint of the image's
    structural content. Unlike the hash paths which encode WHERE structures
    are spatially located, statistical moments encode WHAT KIND of structural
    content exists and in what distribution — making them invariant to:

        Mirror    → same edges exist, just reflected   → stats unchanged
        Rotation  → same edges exist, just rotated     → stats unchanged
        Crop      → dominant content still present     → stats similar

    This is why stats are the correct rescue signal for the compounded
    transform case (mirror + rotation + crop simultaneously) where both
    spatial and invariant hash paths lose signal — the stats survive
    all three transforms at once.

    Implementation note:
        Uses grayscale not L+H for stats computation. Statistical moments
        are about energy distribution not color content — grayscale is
        more stable for moment computation and avoids hue noise from
        the L+H channel affecting the distribution shape.
    """
    image = cv2.imread(image_path)
    if image is None:
        return np.zeros(6, dtype=np.float64)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

    # Sobel edge map — captures gradient energy distribution
    sobel_x   = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y   = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
    sobel_map = np.abs(sobel_x) + np.abs(sobel_y)

    # Laplacian map — captures fine structural change distribution
    lap_map = np.abs(cv2.Laplacian(resized, cv2.CV_64F))

    sobel_flat = sobel_map.flatten()
    lap_flat   = lap_map.flatten()

    sobel_mean = np.mean(sobel_flat)
    sobel_var  = np.var(sobel_flat)
    sobel_skew = float(skew(sobel_flat))

    lap_mean   = np.mean(lap_flat)
    lap_var    = np.var(lap_flat)
    lap_skew   = float(skew(lap_flat))

    return np.array([
        sobel_mean, sobel_var, sobel_skew,
        lap_mean,   lap_var,   lap_skew
    ], dtype=np.float64)


def compute_stats_similarity(stats_a: np.ndarray, stats_b: np.ndarray) -> float:
    """
    Compare two statistical fingerprints and return a similarity score [0, 1].

    Uses normalized absolute difference per component then averages.
    Each component is normalized by the larger of the two values so the
    comparison is scale-independent — the ratio of difference matters,
    not the absolute magnitude.

    Returns 1.0 for identical distributions, lower for diverging ones.
    """
    similarity_components = []

    for a, b in zip(stats_a, stats_b):
        max_val = max(abs(a), abs(b))
        if max_val == 0:
            similarity_components.append(1.0)
        else:
            diff = abs(a - b) / max_val
            similarity_components.append(max(0.0, 1.0 - diff))

    return float(np.mean(similarity_components))