import numpy as np

def divide_into_blocks(feature_map: np.ndarray, grid_size: int = 8) -> list:
    """
    Divide feature map into equal blocks.
    """
    height, width = feature_map.shape
    block_height = height // grid_size
    block_width = width // grid_size
    blocks = []

    for row in range(grid_size):
        for col in range(grid_size):

            start_y = row * block_height
            end_y = start_y + block_height
            start_x = col * block_width
            end_x = start_x + block_width
            block = feature_map[start_y:end_y, start_x:end_x]
            blocks.append(block)

    return blocks


def compute_block_means(blocks: list) -> list:
    """
    Compute mean value for each block.
    """
    means = []

    for block in blocks:
        mean_value = np.mean(block)
        means.append(mean_value)
    return means


def generate_block_hash(feature_map: np.ndarray) -> list:
    """
    Generate block mean hash.
    """
    blocks = divide_into_blocks(feature_map)
    block_means = compute_block_means(blocks)
    overall_mean = np.mean(block_means)
    hash_bits = []

    for value in block_means:
        if value >= overall_mean:
            hash_bits.append(1)
        else:
            hash_bits.append(0)

    return hash_bits