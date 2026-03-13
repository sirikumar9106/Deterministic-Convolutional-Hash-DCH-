from .preprocess import preprocess_image
from .feature_paths.edge_path import generate_edge_feature_map
from .feature_paths.laplacian_path import generate_laplacian_feature_map
from .spectral.wavelet_transform import spectral_feature_map
from .hashing.medianhash import generate_median_hash
from .hashing.hash_concat import bits_to_hex

def generate_dch(image_path: str):
    """
    Dual-Path 256-bit DCH Pipeline.
    [Bits 0-127]: Spatial Detection (Crops/Scaling)
    [Bits 128-255]: Geometric Invariance (Mirror/Rotation)
    """
    image = preprocess_image(image_path)

    # --- PATH 1: EDGE FEATURES (128 bits) ---
    edge_map = generate_edge_feature_map(image)
    edge_s, edge_i = spectral_feature_map(edge_map, target_dimension=8)
    h_edge_spatial = generate_median_hash(edge_s) # 64 bits
    h_edge_invariant = generate_median_hash(edge_i) # 64 bits

    # --- PATH 2: LAPLACIAN FEATURES (128 bits) ---
    lap_map = generate_laplacian_feature_map(image)
    lap_s, lap_i = spectral_feature_map(lap_map, target_dimension=8)
    h_lap_spatial = generate_median_hash(lap_s) # 64 bits
    h_lap_invariant = generate_median_hash(lap_i) # 64 bits

    # --- CONCATENATION ---
    # We group them so they can be compared separately in test_dch.py
    spatial_part = h_edge_spatial + h_lap_spatial   # 128 bits
    invariant_part = h_edge_invariant + h_lap_invariant # 128 bits
    
    combined_bits = spatial_part + invariant_part
    hex_hash = bits_to_hex(combined_bits)

    return {
        "bits": combined_bits,
        "hex": hex_hash
    }