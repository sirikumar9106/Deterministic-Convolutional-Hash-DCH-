import numpy as np
from scipy.stats import entropy


def divide_into_blocks(feature_map: np.ndarray, block_size: int = 8):
    """
    Divide feature map into blocks.
    """

    h, w = feature_map.shape

    blocks = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):

            block = feature_map[i:i + block_size, j:j + block_size]

            if block.shape == (block_size, block_size):

                blocks.append(block)

    return blocks


def block_statistics(block: np.ndarray):
    """
    Extract comprehensive statistics from a block.
    """
    flat_block = block.flatten()

    mean_val = np.mean(flat_block)
    std_val = np.std(flat_block)
    energy = np.sum(flat_block ** 2)
    min_val = np.min(flat_block)
    max_val = np.max(flat_block)
    median_val = np.median(flat_block)

    # Entropy (normalized histogram)
    hist, _ = np.histogram(flat_block, bins=10, density=True)
    hist = hist[hist > 0]  # Remove zeros for entropy calculation
    ent_val = entropy(hist) if len(hist) > 0 else 0

    # Skewness and Kurtosis
    if std_val > 0:
        skewness = np.mean(((flat_block - mean_val) / std_val) ** 3)
        kurtosis = np.mean(((flat_block - mean_val) / std_val) ** 4) - 3
    else:
        skewness = 0
        kurtosis = 0

    return np.array([mean_val, std_val, energy, min_val, max_val, median_val, ent_val, skewness, kurtosis])


def descriptor_from_map(feature_map: np.ndarray, block_size: int = 8):
    """
    Generate descriptor vector for a single feature map.
    """

    blocks = divide_into_blocks(feature_map, block_size)

    descriptor_list = []

    for block in blocks:

        stats = block_statistics(block)

        descriptor_list.extend(stats)

    return np.array(descriptor_list)


def generate_descriptor_stack(feature_stack):
    """
    Generate descriptors for all feature maps.
    """

    descriptor_vectors = []

    for fmap in feature_stack:

        descriptor = descriptor_from_map(fmap)

        descriptor_vectors.extend(descriptor)

    descriptor_array = np.array(descriptor_vectors)

    return descriptor_array