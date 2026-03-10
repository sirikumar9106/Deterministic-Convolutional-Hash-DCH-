import numpy as np


def flatten_block(block: np.ndarray):
    """
    Flatten coefficient block into 1D vector.
    """
    flat = block.flatten()
    return flat


def normalize_vector(values: np.ndarray):
    """
    Normalize values to prevent extreme coefficient dominance.
    """
    max_val = np.max(np.abs(values))

    if max_val == 0:
        return values

    normalized = values / max_val

    return normalized


def compute_adaptive_threshold(values: np.ndarray):
    """
    Compute adaptive threshold using mean + std for better distribution handling.
    """
    mean_val = np.mean(values)
    std_val = np.std(values)
    threshold = mean_val + 0.5 * std_val  # Adaptive threshold
    return threshold


def generate_binary_hash(values: np.ndarray, threshold: float):
    """
    Convert coefficients into binary hash bits using adaptive threshold.
    """

    bits = []

    for v in values:
        if v >= threshold:
            bits.append(1)
        else:
            bits.append(0)

    return bits


def enforce_hash_length(bits, target_length=256):
    """
    Ensure hash is exactly 256 bits.
    """

    if len(bits) > target_length:
        bits = bits[:target_length]

    elif len(bits) < target_length:
        padding = [0] * (target_length - len(bits))
        bits.extend(padding)

    return bits


def bits_to_hex(bits):
    """
    Convert binary bit list into hexadecimal string.
    """

    bit_string = ''.join(str(b) for b in bits)

    hex_string = ''

    for i in range(0, len(bit_string), 4):
        nibble = bit_string[i:i+4]
        hex_digit = hex(int(nibble, 2))[2:]
        hex_string += hex_digit

    return hex_string


def generate_hash(spectral_block: np.ndarray):
    """
    Full hash generation pipeline with adaptive thresholding.
    """

    flat = flatten_block(spectral_block)

    normalized = normalize_vector(flat)

    threshold = compute_adaptive_threshold(normalized)

    bits = generate_binary_hash(normalized, threshold)

    bits = enforce_hash_length(bits, 256)

    hex_hash = bits_to_hex(bits)

    return bits, hex_hash