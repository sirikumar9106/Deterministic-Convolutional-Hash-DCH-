import numpy as np


def divide_into_quadrants(block: np.ndarray) -> list:
    """
    Divide the 2D spectral block into 4 spatial quadrants.

    This preserves positional meaning that flat flattening destroys.
    Each quadrant corresponds to a specific spatial region of the feature map:
      Q0: top-left     Q1: top-right
      Q2: bottom-left  Q3: bottom-right

    Why quadrants matter:
        When you flatten an 8x8 map to 64 values and compute one global median,
        a strong pattern in the top-left corner shifts the global threshold and
        affects how bottom-right bits get binarized — even though those regions
        are spatially unrelated. This is what caused partial pattern collisions
        where an unrelated image sharing one similar subregion scored falsely high.

        By thresholding each quadrant against its own local median, a strong
        pattern in one region only affects that region's bits. The other three
        quadrants remain independently thresholded against their own context.
    """

    mid_row = block.shape[0] // 2
    mid_col = block.shape[1] // 2

    q0 = block[:mid_row, :mid_col]      # top-left
    q1 = block[:mid_row, mid_col:]      # top-right
    q2 = block[mid_row:, :mid_col]      # bottom-left
    q3 = block[mid_row:, mid_col:]      # bottom-right

    return [q0, q1, q2, q3]


def compute_local_median(quadrant: np.ndarray) -> float:
    """
    Compute median value within a single quadrant.

    Using a local median per quadrant instead of a global median across
    the entire map means each spatial region is compared only against
    its own distribution — not contaminated by other regions.
    """

    return float(np.median(quadrant.flatten()))


def binarize_quadrant(quadrant: np.ndarray, threshold: float) -> list:
    """
    Binarize a single quadrant against its own local median threshold.

    Flattening happens here, after the spatial division — so the flatten
    only collapses within one quadrant, not across the whole map.
    Each bit still carries implicit positional meaning: bits 0-15 always
    come from top-left, bits 16-31 from top-right, and so on.
    """

    flat_values = quadrant.flatten()
    bits = []

    for value in flat_values:
        if value >= threshold:
            bits.append(1)
        else:
            bits.append(0)

    return bits


def generate_median_hash(spectral_block: np.ndarray) -> list:
    """
    Complete local-median hash pipeline.

    Replaces the old single global-median approach with per-quadrant
    local thresholding. Output bit count is identical to before —
    an 8x8 block still produces 64 bits — so pipeline.py and the
    Hamming distance comparisons are completely unaffected.

    Bit layout of the output (for an 8x8 input):
      bits  0-15:  top-left quadrant     (4x4 region)
      bits 16-31:  top-right quadrant    (4x4 region)
      bits 32-47:  bottom-left quadrant  (4x4 region)
      bits 48-63:  bottom-right quadrant (4x4 region)

    This layout means spatially close regions produce adjacent bits,
    so Hamming distance now has mild spatial sensitivity — two images
    that differ only in one corner will show bit differences clustered
    in that corner's bit range, not scattered randomly across all 64 bits.
    """

    quadrants = divide_into_quadrants(spectral_block)
    hash_bits = []

    for quadrant in quadrants:
        local_threshold = compute_local_median(quadrant)
        quadrant_bits = binarize_quadrant(quadrant, local_threshold)
        hash_bits.extend(quadrant_bits)

    return hash_bits