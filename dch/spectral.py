import numpy as np
import pywt


def descriptor_to_matrix(descriptor: np.ndarray):
    """
    Convert descriptor vector into square matrix.
    """

    length = len(descriptor)

    size = int(np.sqrt(length))

    if size * size != length:

        size = int(np.floor(np.sqrt(length)))

        descriptor = descriptor[:size * size]

    matrix = descriptor.reshape(size, size)

    return matrix


def wavelet_transform(matrix: np.ndarray):
    """
    Perform Haar wavelet transform and combine coefficients.
    """

    coeffs = pywt.dwt2(matrix, 'haar')

    LL, (LH, HL, HH) = coeffs

    # Combine all coefficients into a single matrix for richer representation
    # Stack them to preserve more information
    combined = np.concatenate([
        LL.flatten(),
        LH.flatten(),
        HL.flatten(),
        HH.flatten()
    ])

    # Reshape back to square if possible
    combined_size = int(np.sqrt(len(combined)))
    if combined_size * combined_size == len(combined):
        combined_matrix = combined.reshape(combined_size, combined_size)
    else:
        # Pad or truncate to make square
        target_size = combined_size
        if len(combined) < target_size * target_size:
            combined = np.pad(combined, (0, target_size*target_size - len(combined)), 'constant')
        else:
            combined = combined[:target_size*target_size]
        combined_matrix = combined.reshape(target_size, target_size)

    return combined_matrix


def extract_low_frequency_block(matrix: np.ndarray, block_size: int = 32):
    """
    Extract larger low-frequency coefficients block.
    """

    h, w = matrix.shape

    if block_size > h or block_size > w:

        block_size = min(h, w)

    block = matrix[0:block_size, 0:block_size]

    return block


def spectral_features(descriptor_vector: np.ndarray):
    """
    Full spectral processing pipeline.
    """

    matrix = descriptor_to_matrix(descriptor_vector)

    LL = wavelet_transform(matrix)

    low_block = extract_low_frequency_block(LL)

    return low_block