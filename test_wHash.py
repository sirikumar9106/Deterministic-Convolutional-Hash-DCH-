from PIL import Image
import imagehash


def compute_whash(image_path):

    image = Image.open(image_path)

    whash = imagehash.whash(image)

    return whash


def compare_whash(base_image, target_image):

    hash1 = compute_whash(base_image)

    hash2 = compute_whash(target_image)

    distance = hash1 - hash2

    similarity = 1 - (distance / 64)

    print("--------------------------------")
    print("Base:", base_image)
    print("Target:", target_image)
    print("wHash distance:", distance)
    print("wHash similarity:", round(similarity * 100, 2), "%")
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

        compare_whash(base, img)


if __name__ == "__main__":

    main()