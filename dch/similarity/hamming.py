def compute_hamming_distance(hash_a: list, hash_b: list) -> int:
    """
    Compute Hamming distance between two hashes.
    """
    if len(hash_a) != len(hash_b):
        raise ValueError("Hashes must be of equal length")

    distance = 0

    for index in range(len(hash_a)):
        if hash_a[index] != hash_b[index]:
            distance += 1

    return distance


def compute_similarity(hash_a: list, hash_b: list) -> float:
    """
    Compute similarity score between two hashes.
    """
    distance = compute_hamming_distance(hash_a, hash_b)
    hash_length = len(hash_a)
    similarity_score = 1 - (distance / hash_length)
    return similarity_score