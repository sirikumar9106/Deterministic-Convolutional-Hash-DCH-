from dch.pipeline import generate_dch
from dch.similarity.hamming import compute_similarity


def compare_images(base_image: str, target_image: str):
    # 1. Generate the Parallel Hashes
    base_res = generate_dch(base_image)
    target_res = generate_dch(target_image)
    
    base_bits = base_res["bits"]
    target_bits = target_res["bits"]

    # 2. Split the hash into its two specialized detectors (128 bits each)
    # Segment 1: The Spatial "Crop" Path
    spatial_base = base_bits[0:128]
    spatial_target = target_bits[0:128]
    sim_spatial = compute_similarity(spatial_base, spatial_target)

    # Segment 2: The Geometric "Mirror/Rotation" Path
    inv_base = base_bits[128:256]
    inv_target = target_bits[128:256]
    sim_invariant = compute_similarity(inv_base, inv_target)

    # 3. FINAL LOGIC: Pick the highest similarity from the two paths
    # This ensures that if the content is the same (regardless of orientation), we find it.
    final_similarity = max(sim_spatial, sim_invariant)

    print("------------------------------------")
    print("Base Image   :", base_image)
    print("Target Image :", target_image)
    print("------------------------------------")
    print(f"Spatial Path (Crop Robust)    : {round(sim_spatial * 100, 2)}%")
    print(f"Invariant Path (Mirror Robust): {round(sim_invariant * 100, 2)}%")
    print(f">>> FINAL DCH MATCH SCORE     : {round(final_similarity * 100, 2)}%")
    print("------------------------------------\n")


def main():
    base_image = "test.png"
    test_images = ["test1.png", "test2.png", "test3.png", "test4.png"]

    for image in test_images:
        compare_images(base_image, image)


if __name__ == "__main__":
    main()