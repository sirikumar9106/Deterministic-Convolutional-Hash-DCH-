from dch.pipeline import generate_dch
from dch.similarity.hamming import compute_similarity


def compare_images(base_image: str, target_image: str):

    base_hash = generate_dch(base_image)

    target_hash = generate_dch(target_image)

    similarity = compute_similarity(base_hash["bits"], target_hash["bits"])

    print("------------------------------------")
    print("Base Image :", base_image)
    print("Target Image :", target_image)
    print("Base Hash :", base_hash["hex"])
    print("Target Hash :", target_hash["hex"])
    print("Similarity :", round(similarity * 100, 2), "%")
    print("------------------------------------\n")


def main():

    base_image = "test.png"

    test_images = [
        "test1.png",
        "test2.png",
        "test3.png",
        "test4.png"
    ]

    for image in test_images:

        compare_images(base_image, image)


if __name__ == "__main__":

    main()