from .preprocess import preprocess_image

from .feature_paths.edge_path import compute_edge_spectral_features
from .feature_paths.laplacian_path import compute_laplacian_spectral_features
from .feature_paths.texture_path import compute_texture_spectral_features

from .hashing.medianhash import generate_median_hash
from .hashing.blockhash import generate_block_hash
from .hashing.hash_concat import concatenate_hashes, bits_to_hex

def generate_dch(image_path: str):
    """
    Complete Deterministic Convolutional Hash pipeline.
    """
    image = preprocess_image(image_path)
    edge_spectral = compute_edge_spectral_features(image)
    laplacian_spectral = compute_laplacian_spectral_features(image)
    texture_spectral = compute_texture_spectral_features(image)
    edge_hash = generate_median_hash(edge_spectral)
    laplacian_hash = generate_block_hash(laplacian_spectral)
    texture_hash = generate_median_hash(texture_spectral)
    
    combined_hash = concatenate_hashes([
        edge_hash,
        laplacian_hash,
        texture_hash
    ])

    hex_hash = bits_to_hex(combined_hash)

    return {
        "bits": combined_hash,
        "hex": hex_hash
    }