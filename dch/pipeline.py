import numpy as np
from .preprocess import preprocess_image
from .feature_paths.edge_path import generate_edge_feature_map
from .feature_paths.laplacian_path import generate_laplacian_feature_map
from .filters.texture_filters import compute_absolute_dog
from .spectral.wavelet_transform import spectral_feature_map
from .hashing.medianhash import generate_median_hash
from .hashing.hash_concat import bits_to_hex
from .similarity.stats import compute_structural_stats, compute_stats_similarity


def normalize_map(feature_map: np.ndarray) -> np.ndarray:
    """
    Normalize a feature map to [0, 1] range.

    Required before blending DoG with Sobel or Laplacian maps.
    Each filter produces values on different scales — normalizing
    ensures the blend ratio reflects actual character contribution
    not numerical scale dominance.
    """
    map_min = np.min(feature_map)
    map_max = np.max(feature_map)

    if map_max - map_min == 0:
        return np.zeros_like(feature_map, dtype=np.float64)

    return (feature_map - map_min) / (map_max - map_min)


def blend_with_dog(primary_map: np.ndarray, dog_map: np.ndarray,
                   dog_weight: float = 0.15) -> np.ndarray:
    """
    Blend a primary feature map with the DoG texture map at 85/15 ratio.

    DoG captures mid-frequency texture blobs — fur, clouds, surface texture —
    that Sobel (sharp edges) and Laplacian (fine transitions) don't fully
    capture. At 15% contribution DoG adds texture sensitivity without
    overriding the primary filter's structural information.

    Key discrimination this enables:
        High DoG response  → dense texture (fur, feathers, foliage)
        Near-zero DoG      → smooth surfaces (clear sky, calm water)
    Two images with similar edge structure but different surface texture
    now produce different feature maps before hashing.
    """
    primary_normalized = normalize_map(primary_map)
    dog_normalized     = normalize_map(dog_map)
    primary_weight     = 1.0 - dog_weight

    return (primary_weight * primary_normalized) + (dog_weight * dog_normalized)


def generate_dch(image_path: str) -> dict:
    """
    Dual-Path 256-bit DCH Pipeline.

    [Bits   0-127]: Spatial Detection    — crop, scale, watermark robust
    [Bits 128-255]: Geometric Invariance — mirror, rotation robust

    Returns a dict with:
        bits  — 256-element list of binary hash bits
        hex   — 64-character hexadecimal hash string
        stats — 6-element numpy array of structural statistical moments
                [sobel_mean, sobel_var, sobel_skew,
                 lap_mean,   lap_var,   lap_skew]

    The stats vector is returned alongside the hash so the comparison
    layer (compare_images) can use it for the stats rescue signal without
    reloading and reprocessing the image a second time.
    """
    image = preprocess_image(image_path)

    # --- DOG TEXTURE MAP — computed once, shared across both paths ---
    # Blended at 15% into both feature paths before wavelet decomposition.
    # Adds mid-frequency texture sensitivity without changing hash size.
    dog_map = compute_absolute_dog(image)

    # --- PATH 1: EDGE FEATURES (128 bits) ---
    edge_map          = generate_edge_feature_map(image)
    edge_map_enriched = blend_with_dog(edge_map, dog_map, dog_weight=0.15)
    edge_s, edge_i    = spectral_feature_map(edge_map_enriched, target_dimension=8)
    h_edge_spatial    = generate_median_hash(edge_s)    # 64 bits
    h_edge_invariant  = generate_median_hash(edge_i)    # 64 bits

    # --- PATH 2: LAPLACIAN FEATURES (128 bits) ---
    lap_map          = generate_laplacian_feature_map(image)
    lap_map_enriched = blend_with_dog(lap_map, dog_map, dog_weight=0.15)
    lap_s, lap_i     = spectral_feature_map(lap_map_enriched, target_dimension=8)
    h_lap_spatial    = generate_median_hash(lap_s)    # 64 bits
    h_lap_invariant  = generate_median_hash(lap_i)    # 64 bits

    # --- CONCATENATION ---
    spatial_part   = h_edge_spatial   + h_lap_spatial    # 128 bits
    invariant_part = h_edge_invariant + h_lap_invariant  # 128 bits

    combined_bits = spatial_part + invariant_part
    hex_hash      = bits_to_hex(combined_bits)

    # --- STRUCTURAL STATS ---
    # Computed from raw image path — independent of the hash pipeline.
    # Returned here so compare_images receives it without reprocessing.
    stats = compute_structural_stats(image_path)

    return {
        "bits":  combined_bits,
        "hex":   hex_hash,
        "stats": stats
    }