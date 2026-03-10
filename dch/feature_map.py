import numpy as np


def average_pooling(feature_map: np.ndarray, pool_size: int = 2) -> np.ndarray:
    """
    Apply average pooling to a single feature map.
    """

    h, w = feature_map.shape

    pooled_h = h // pool_size
    pooled_w = w // pool_size

    pooled = np.zeros((pooled_h, pooled_w), dtype=np.float32)

    for i in range(pooled_h):
        for j in range(pooled_w):

            start_i = i * pool_size
            start_j = j * pool_size

            block = feature_map[
                start_i:start_i + pool_size,
                start_j:start_j + pool_size
            ]

            pooled[i, j] = np.mean(block)

    return pooled


def generate_feature_stack(filtered_stack):
    """
    Apply average pooling to each feature map (no ReLU to preserve information).
    """

    processed_stack = []

    for fmap in filtered_stack:

        pooled_map = average_pooling(fmap)

        processed_stack.append(pooled_map)

    return processed_stack