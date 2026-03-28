import numpy as np
import pywt
import cv2


def compute_radial_energy_profile(magnitude_map: np.ndarray,
                                   target_dimension: int = 8) -> np.ndarray:
    """
    Compute a Radial Energy Profile (REP) from the magnitude map.

    Replaces make_d4_symmetric as the rotation and flip invariant path.

    When an image is rotated or flipped, the distance of any structural
    feature from the center does not change — only its angle changes.
    By describing the magnitude map purely in terms of how much energy
    exists at each radial distance from center (ignoring angle entirely),
    we get a descriptor that is:

      - Invariant to any rotation angle, not just 90 degrees
      - Invariant to any flip or mirror operation
      - Content discriminative — a cat (texture everywhere, energy spread
        across all rings) vs a sunset (smooth sky, energy only in inner
        rings) produce very different radial profiles

    Output: (target_dimension x target_dimension) float64 array in [0, 1]
    Same shape as spatial_map so medianhash receives identical input format.
    """

    rows, cols = magnitude_map.shape
    center_row = rows / 2.0
    center_col = cols / 2.0
    max_radius = np.sqrt((rows / 2.0) ** 2 + (cols / 2.0) ** 2)

    # Per-pixel distance from center
    row_indices, col_indices = np.indices((rows, cols))
    distance_map = np.sqrt(
        (row_indices - center_row) ** 2 +
        (col_indices - center_col) ** 2
    )

    # Bin pixels into radial rings
    num_rings = target_dimension * target_dimension
    ring_energies = np.zeros(num_rings, dtype=np.float64)

    for ring_index in range(num_rings):
        inner_radius = (ring_index / num_rings) * max_radius
        outer_radius = ((ring_index + 1) / num_rings) * max_radius
        ring_mask = (distance_map >= inner_radius) & (distance_map < outer_radius)

        if np.any(ring_mask):
            ring_energies[ring_index] = np.mean(magnitude_map[ring_mask])
        else:
            ring_energies[ring_index] = 0.0

    # Normalize to [0, 1]
    max_energy = np.max(ring_energies)
    if max_energy > 0:
        ring_energies = ring_energies / max_energy

    # Reshape 1D profile to 2D square — medianhash expects 2D input
    radial_map = ring_energies.reshape(target_dimension, target_dimension)

    return radial_map


def spectral_feature_map(feature_map: np.ndarray, target_dimension: int = 8):
    """
    Returns two maps for the dual-path hashing pipeline.

    Path A — Spatial Map (crop and scale robust):
        Weighted DWT magnitude map resized to target_dimension x target_dimension.
        Captures where structural energy is spatially located.

    Path B — Invariant Map (rotation and flip robust):
        Radial Energy Profile of the same weighted magnitude map.
        Invariant to any rotation angle and any flip/mirror operation.

    Subband weighting rationale:
        LH (horizontal edges) and HL (vertical edges) are given equal weight
        at 0.28 each — horizontal and vertical edges are symmetric in their
        structural importance and neither should be favoured over the other.

        HH (diagonal edges) gets 0.44 — the highest weight. Diagonal complexity
        is the most content discriminative subband. It captures texture, corners,
        and complex local structure that varies most between different content
        types. A smooth sky has near-zero HH while a textured cat or dense
        forest has high HH everywhere. Shifting weight here pushes false positive
        pairs further apart in hash space without affecting true match detection.

        LH + HL + HH weights sum to 1.0: 0.28 + 0.28 + 0.44 = 1.00
    """

    # Discrete Wavelet Transform
    coeffs = pywt.dwt2(feature_map, 'haar')
    LL, (LH, HL, HH) = coeffs

    # Weighted magnitude map — LH and HL equal, HH dominant
    magnitude_map = np.sqrt(
        (0.33 * np.power(LH, 2)) +
        (0.33 * np.power(HL, 2)) +
        (0.33 * np.power(HH, 2))
    )

    # Path A: Spatial Map
    spatial_map = cv2.resize(
        magnitude_map,
        (target_dimension, target_dimension),
        interpolation=cv2.INTER_AREA
    )

    # Path B: Invariant Map
    inv_map = compute_radial_energy_profile(magnitude_map, target_dimension)

    return spatial_map, inv_map