from PIL import Image
import imagehash


def compute_phash(image_path):

    image = Image.open(image_path)

    phash = imagehash.phash(image)

    return phash


def compare_phash(base_image, target_image):

    hash1 = compute_phash(base_image)

    hash2 = compute_phash(target_image)

    distance = hash1 - hash2

    similarity = 1 - (distance / 64)

    print("--------------------------------")
    print("Base:", base_image)
    print("Target:", target_image)
    print("pHash distance:", distance)
    print("pHash similarity:", round(similarity * 100, 2), "%")
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

        compare_phash(base, img)


if __name__ == "__main__":

    main()