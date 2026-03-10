import numpy as np
import cv2


def cartesian_to_polar(feature_map: np.ndarray):
    """
    Convert a feature map from Cartesian to polar coordinates.
    """

    h, w = feature_map.shape

    center = (w // 2, h // 2)

    max_radius = min(center[0], center[1])

    polar = cv2.warpPolar(
        feature_map,
        (w, h),
        center,
        max_radius,
        cv2.WARP_POLAR_LINEAR
    )

    return polar


def polar_transform_stack(feature_stack):
    """
    Apply polar transform to each feature map.
    """

    polar_stack = []

    for fmap in feature_stack:

        polar_map = cartesian_to_polar(fmap)

        polar_stack.append(polar_map)

    return polar_stack