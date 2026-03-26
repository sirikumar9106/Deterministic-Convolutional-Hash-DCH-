from PIL import Image
import imagehash


def compute_ahash(image_path):

    image = Image.open(image_path)

    ahash = imagehash.average_hash(image)

    return ahash


def compare_ahash(base_image, target_image):

    hash1 = compute_ahash(base_image)

    hash2 = compute_ahash(target_image)

    distance = hash1 - hash2

    similarity = 1 - (distance / 64)

    print("--------------------------------")
    print("Base:", base_image)
    print("Target:", target_image)
    print("aHash distance:", distance)
    print("aHash similarity:", round(similarity * 100, 2), "%")
    print("--------------------------------")


def main():

    base = "test.png"

    images = [
        "test1.png",
        "test2.png",
        "test3.png",
        "test4.png"
    ]

    for img in images:

        compare_ahash(base, img)


if __name__ == "__main__":

    main()