from .preprocess import preprocess_image
from .feature_paths.edge_path import generate_edge_feature_map
from .feature_paths.laplacian_path import generate_laplacian_feature_map
from .feature_paths.texture_path import generate_texture_feature_map
from .spectral.wavelet_transform import spectral_feature_map
from .hashing.medianhash import generate_median_hash
from .hashing.hash_concat import concatenate_hashes, bits_to_hex

def generate_dch(image_path: str):
    """
    Fixed Deterministic Convolutional Hash pipeline.
    Target: 256 bits total for high similarity stability.
    """
    # 1. Preprocess the image (256x256 grayscale)
    image = preprocess_image(image_path)

    # --- PATH 1: EDGES (128 Bits) ---
    # We use a 11x11 grid (121 bits) + 7 padding bits to get close to 128
    edge_map = generate_edge_feature_map(image)
    edge_spectral = spectral_feature_map(edge_map, target_dimension=11) 
    edge_hash = generate_median_hash(edge_spectral) # 121 bits

    # --- PATH 2: LAPLACIAN STRUCTURE (64 Bits) ---
    # 8x8 grid = 64 bits
    laplacian_map = generate_laplacian_feature_map(image)
    lap_spectral = spectral_feature_map(laplacian_map, target_dimension=8)
    laplacian_hash = generate_median_hash(lap_spectral) # 64 bits

    # --- PATH 3: TEXTURE (64 Bits) ---
    # 8x8 grid = 64 bits
    texture_map = generate_texture_feature_map(image)
    tex_spectral = spectral_feature_map(texture_map, target_dimension=8)
    texture_hash = generate_median_hash(tex_spectral) # 64 bits

    # 2. Combine and enforce the 256-bit limit
    # We take all 121 bits of edge, 64 of lap, and 71 bits of the rest to fill 256
    raw_combined = concatenate_hashes([edge_hash, laplacian_hash, texture_hash])
    
    # Slice exactly to 256 bits if it slightly exceeds due to the 11x11 grid
    combined_hash = raw_combined[:256]

    # 3. Convert to Hex
    hex_hash = bits_to_hex(combined_hash)

    return {
        "bits": combined_hash,
        "hex": hex_hash
    }