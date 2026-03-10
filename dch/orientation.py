import numpy as np
import cv2


def compute_gradients(feature_map: np.ndarray):
    """
    Compute gradients for orientation estimation.
    """

    gx = cv2.Sobel(feature_map, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(feature_map, cv2.CV_64F, 0, 1, ksize=3)

    return gx, gy


def compute_orientation_histogram(gx: np.ndarray,
                                  gy: np.ndarray,
                                  bins: int = 36):
    """
    Compute gradient orientation histogram.
    """

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    orientation = np.arctan2(gy, gx)

    orientation_deg = np.degrees(orientation) % 360

    hist = np.zeros(bins)

    bin_size = 360 / bins

    h, w = orientation_deg.shape

    for i in range(h):
        for j in range(w):

            angle = orientation_deg[i, j]

            mag = magnitude[i, j]

            bin_index = int(angle // bin_size)

            hist[bin_index] += mag

    return hist


def dominant_orientation(histogram: np.ndarray,
                         bins: int = 36):
    """
    Determine dominant orientation.
    """

    max_idx = np.argmax(histogram)

    bin_size = 360 / bins

    angle = max_idx * bin_size

    return angle


def rotate_feature_map(feature_map: np.ndarray,
                       angle: float):
    """
    Rotate feature map to canonical orientation.
    """

    h, w = feature_map.shape

    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        -angle,
        1.0
    )

    rotated = cv2.warpAffine(
        feature_map,
        rotation_matrix,
        (w, h)
    )

    return rotated


def orientation_canonicalization(feature_stack):
    """
    Align each feature map to dominant orientation.
    """

    reference_map = feature_stack[0]

    gx, gy = compute_gradients(reference_map)

    hist = compute_orientation_histogram(gx, gy)

    angle = dominant_orientation(hist)

    aligned_stack = []

    for fmap in feature_stack:

        aligned = rotate_feature_map(fmap, angle)

        aligned_stack.append(aligned)

    return aligned_stack