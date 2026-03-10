import matplotlib.pyplot as plt
from dch.pipeline import generate_dch


def hamming_distance(hash1, hash2):
    distance = 0
    for a, b in zip(hash1, hash2):
        if a != b:
            distance += 1
    return distance


def hash_images(image_paths):
    results = {}

    for path in image_paths:
        result = generate_dch(path)
        results[path] = result
        print("\n===== HASH RESULT =====")
        print("Image:", path)
        print("Binary length:", len(result["bits"]))
        print("Hex Hash:", result["hex"])

    return results


def compare_hashes(results):
    keys = list(results.keys())

    print("\n===== HASH SIMILARITY =====")

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):

            img1 = keys[i]
            img2 = keys[j]

            hash1 = results[img1]["bits"]
            hash2 = results[img2]["bits"]

            distance = hamming_distance(hash1, hash2)

            similarity = 1 - (distance / len(hash1))

            print(f"{img1} vs {img2}")
            print("Hamming distance:", distance)
            print("Similarity:", round(similarity * 100, 2), "%\n")


def main():

    images = [
        "test.png",
        "test1.png",
        "test2.png",
        "test3.png",
        "test4.png"
    ]

    results = hash_images(images)

    compare_hashes(results)


if __name__ == "__main__":
    main()