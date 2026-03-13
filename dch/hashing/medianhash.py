import numpy as np

def flatten_block(block: np.ndarray) -> np.ndarray:
    """
    Flatten a 2D spectral block into a 1D vector.
    """
    flat_vector = block.flatten()
    return flat_vector


def compute_median_threshold(values: np.ndarray) -> float:
    """
    Compute median value used as hashing threshold.
    """
    median_value = np.median(values)
    return median_value


def generate_binary_hash(values: np.ndarray, threshold: float) -> list:
    """
    Convert values into binary bits using the median threshold.
    """
    binary_bits = []

    for value in values:
        if value >= threshold:
            binary_bits.append(1)
        else:
            binary_bits.append(0)

    return binary_bits


def generate_median_hash(spectral_block: np.ndarray) -> list:
    """
    Complete median hash pipeline.
    """
    flat_values = flatten_block(spectral_block)
    threshold = compute_median_threshold(flat_values)
    hash_bits = generate_binary_hash(flat_values, threshold)
    return hash_bits