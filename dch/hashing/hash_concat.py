def concatenate_hashes(hash_list: list) -> list:
    """
    Concatenate multiple hash bit lists into a single vector.
    """

    combined_hash = []
    for hash_bits in hash_list:
        for bit in hash_bits:
            combined_hash.append(bit)

    return combined_hash


def bits_to_hex(hash_bits: list) -> str:
    """
    Convert binary hash bits into hexadecimal string.
    """
    bit_string = ''.join(str(bit) for bit in hash_bits)
    hex_string = ''

    for index in range(0, len(bit_string), 4):
        nibble = bit_string[index:index + 4]
        hex_digit = hex(int(nibble, 2))[2:]
        hex_string += hex_digit

    return hex_string