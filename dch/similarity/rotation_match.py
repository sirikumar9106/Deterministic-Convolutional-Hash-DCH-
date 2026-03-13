def circular_shift(hash_bits: list, shift: int) -> list:
    """
    Perform circular shift of a hash bit vector.
    """
    shift = shift % len(hash_bits)
    shifted_hash = hash_bits[-shift:] + hash_bits[:-shift]
    return shifted_hash


def check_rotation_match(hash_a: list, hash_b: list, max_shifts: int = 32) -> float:
    """
    Check similarity under circular bit shifts.
    """
    best_similarity = 0.0

    for shift in range(max_shifts):
        shifted_hash = circular_shift(hash_b, shift)
        matches = 0

        for index in range(len(hash_a)):

            if hash_a[index] == shifted_hash[index]:
                matches += 1

        similarity = matches / len(hash_a)

        if similarity > best_similarity:
            best_similarity = similarity

    return best_similarity