from .preprocess import preprocess_image
from .filters import compute_feature_stack
from .feature_map import generate_feature_stack
from .orientation import orientation_canonicalization
from .polar_transform import polar_transform_stack
from .block_descriptor import generate_descriptor_stack
from .spectral import spectral_features
from .hash_generator import generate_hash


def generate_dch(image_path: str):
    """
    Complete Deterministic Convolutional Hash pipeline.
    """

    image = preprocess_image(image_path)

    feature_stack = compute_feature_stack(image)

    pooled_stack = generate_feature_stack(feature_stack)

    aligned_stack = orientation_canonicalization(pooled_stack)

    polar_stack = polar_transform_stack(aligned_stack)

    descriptor_vector = generate_descriptor_stack(polar_stack)

    spectral_block = spectral_features(descriptor_vector)

    bits, hex_hash = generate_hash(spectral_block)

    return {
        "bits": bits,
        "hex": hex_hash,
        "descriptor": descriptor_vector,
        "spectral_block": spectral_block
    }