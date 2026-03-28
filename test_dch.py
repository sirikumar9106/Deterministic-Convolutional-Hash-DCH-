from dch.pipeline import generate_dch
from dch.similarity.hamming import compute_similarity
from dch.similarity.stats import compute_stats_similarity


# ─────────────────────────────────────────────────────────────────────────────
# Coherence score
# Measures whether matching bits are clustered (true match) or
# scattered (false positive) — independent of similarity percentage.
# ─────────────────────────────────────────────────────────────────────────────

def compute_coherence_score(bits_a: list, bits_b: list) -> float:
    """
    High coherence = matching bits form consecutive runs → true match behaviour.
    Low coherence  = matching bits scattered randomly    → false positive behaviour.
    """
    total_bits   = len(bits_a)
    match_vector = [1 if bits_a[i] == bits_b[i] else 0 for i in range(total_bits)]

    runs        = []
    current_run = 0
    for bit in match_vector:
        if bit == 1:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)

    if not runs:
        return 0.0

    total_matching = sum(match_vector)
    if total_matching == 0:
        return 0.0

    longest_run       = max(runs)
    num_runs          = len(runs)
    clustered_bits    = sum(r for r in runs if r >= 3)
    longest_run_score = longest_run / total_bits
    cluster_ratio     = clustered_bits / total_matching if total_matching > 0 else 0.0
    avg_run_length    = total_matching / num_runs if num_runs > 0 else 0
    frag_penalty      = min(avg_run_length / 8.0, 1.0)

    coherence = (
        (0.35 * longest_run_score) +
        (0.40 * cluster_ratio)     +
        (0.25 * frag_penalty)
    )
    return min(coherence, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Final score with coherence penalty
# Coherence only fires in the ambiguous 60-70% zone AND only when the
# other path is also weak — high confidence matches are untouched.
# ─────────────────────────────────────────────────────────────────────────────

def compute_final_score(sim_spatial, sim_invariant,
                        coherence_spatial, coherence_invariant) -> float:
    spatial_confidence   = sim_spatial
    invariant_confidence = sim_invariant

    if 0.60 <= sim_spatial <= 0.70 and sim_invariant < 0.55:
        spatial_confidence = sim_spatial * (0.7 + 0.3 * coherence_spatial)

    if 0.60 <= sim_invariant <= 0.70 and sim_spatial < 0.55:
        invariant_confidence = sim_invariant * (0.7 + 0.3 * coherence_invariant)

    return max(spatial_confidence, invariant_confidence)


# ─────────────────────────────────────────────────────────────────────────────
# Image comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_images(base_image: str, target_image: str,
                   accept_threshold: float = 0.60) -> dict:

    # Generate hashes and stats for both images
    base_res   = generate_dch(base_image)
    target_res = generate_dch(target_image)

    base_bits   = base_res["bits"]
    target_bits = target_res["bits"]

    # Spatial path — bits 0:128
    spatial_base   = base_bits[0:128]
    spatial_target = target_bits[0:128]
    sim_spatial    = compute_similarity(spatial_base, spatial_target)

    # Invariant path — bits 128:256
    inv_base      = base_bits[128:256]
    inv_target    = target_bits[128:256]
    sim_invariant = compute_similarity(inv_base, inv_target)

    # Coherence per path
    coh_spatial   = compute_coherence_score(spatial_base, spatial_target)
    coh_invariant = compute_coherence_score(inv_base, inv_target)

    # Final score with coherence penalty in ambiguous zone
    final = compute_final_score(sim_spatial, sim_invariant,
                                coh_spatial, coh_invariant)

    # Stats similarity — retrieved from pipeline output, no reprocessing
    sim_stats = compute_stats_similarity(base_res["stats"], target_res["stats"])

    # Stats rescue — only activates when both hash paths are weak
    # AND stats similarity strongly indicates same structural distribution.
    # Targets the compounded transform edge case (mirror + rotation + crop).
    # Threshold 0.82 — above this the same image's distribution is recognized,
    # below this different-but-similar images (different city, same aesthetic)
    # stay correctly rejected.
    final_with_stats = final
    stats_rescued    = False

    if final < accept_threshold and sim_stats >= 0.82:
        final_with_stats = (0.70 * final) + (0.30 * sim_stats)
        stats_rescued    = (final_with_stats >= accept_threshold)

    verdict    = "SELECT" if final_with_stats >= accept_threshold else "REJECT"
    rescue_tag = " [STATS RESCUE]" if stats_rescued else ""

    return {
        "base":        base_image,
        "target":      target_image,
        "spa":         round(sim_spatial * 100, 2),
        "inv":         round(sim_invariant * 100, 2),
        "spa_ch":      round(coh_spatial * 100, 2),
        "inv_ch":      round(coh_invariant * 100, 2),
        "stats":       round(sim_stats * 100, 2),
        "final":       round(final_with_stats * 100, 2),
        "verdict":     verdict,
        "rescue_tag":  rescue_tag,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output and batch runner
# ─────────────────────────────────────────────────────────────────────────────

def print_result(r: dict):
    print(f"  {r['base']} vs {r['target']:<12} "
          f"spa={r['spa']:>6}%  inv={r['inv']:>6}%  "
          f"spa_ch={r['spa_ch']:>6}%  inv_ch={r['inv_ch']:>6}%  "
          f"stats={r['stats']:>6}%  "
          f"final={r['final']:>6}%  "
          f"[{r['verdict']}]{r['rescue_tag']}")


def run_batch(base_image: str, test_images: list,
              accept_threshold: float = 0.60):
    print(f"\n{'='*120}")
    print(f"  BASE: {base_image}")
    print(f"{'='*120}")
    print(f"  {'Comparison':<26} {'spa':>8}  {'inv':>8}  {'spa_ch':>9}  "
          f"{'inv_ch':>9}  {'stats':>8}  {'final':>8}  verdict")
    print(f"  {'-'*110}")

    results = []
    for target in test_images:
        r = compare_images(base_image, target, accept_threshold)
        print_result(r)
        results.append(r)

    selected = [r for r in results if r["verdict"] == "SELECT"]
    rejected = [r for r in results if r["verdict"] == "REJECT"]
    rescued  = [r for r in results if r["rescue_tag"]]

    print(f"\n  SUMMARY — {base_image}: "
          f"{len(selected)} selected, {len(rejected)} rejected "
          f"({len(rescued)} stats-rescued) out of {len(results)}")
    print(f"{'='*120}\n")
    return results


def main():
    test_images = [f"test{i}.png" for i in range(1, 21)]

    run_batch("testA.png", test_images, accept_threshold=0.60)
    run_batch("testB.png", test_images, accept_threshold=0.60)


if __name__ == "__main__":
    main()