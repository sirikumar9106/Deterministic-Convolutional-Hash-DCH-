# Deterministic Convolutional Hashing (DCH) — Dual-Path Perceptual Hashing

A zero-model, deterministic perceptual image hashing system designed to detect duplicate, cropped, scaled, mirrored, and rotated versions of images — without any trained neural network.

---

## Why This Exists

Most image deduplication pipelines rely on AI models to compare images directly. This works but carries a cost — model inference at scale consumes significant compute, energy, and produces carbon footprint proportional to dataset size.

DCH is built as a **pre-filter**. The idea is:

> Run a fast classical algorithm first to sort a large dataset into confident matches and confident non-matches. Let the AI model spend its compute only on the ambiguous remainder.

A dataset that needed 57,000 AI comparisons now needs far fewer. Same accuracy, fraction of the energy.

**The zero-model constraint is intentional and non-negotiable.** No weights, no training data, no GPU required. DCH runs deterministically on any hardware.

---

## What Makes DCH Different from pHash / aHash / dHash

Traditional perceptual hashing does this:

```
image → scale to 8x8 → compute DCT or pixel differences → hash
```

The problem: when you scale to 8x8 first, you lose structural information. A mirrored image produces completely different frequency patterns at that scale. aHash fails entirely on flips. pHash gives ~60% similarity on mirrored images at best.

DCH does this instead:

```
image → extract structural feature maps → wavelet decomposition → hash
```

The feature extraction step is inspired by the early convolutional layers of a CNN — it finds **what structures exist** (edges, gradients, texture) before any spatial compression happens. When an image is flipped, the same edges still exist — just repositioned. The wavelet step then compresses this structural information in a way that is stable across geometric transforms.

This is what gets DCH to **80-90% similarity on mirrored and rotated images** where traditional methods plateau at 60-70%.

---

## Architecture

### Full Pipeline

```
Image
  │
  ▼
preprocess.py
  • Load (BGR)
  • Convert to L+H channel  ← replaces grayscale
  • Resize to 256×256
  │
  ├─────────────────────────────────────┐
  ▼                                     ▼
Edge Path                         Laplacian Path
sobel_filters.py                  laplacian_filter.py
|SobelX| + |SobelY|               |∇²f|
  │                                     │
  ▼                                     ▼
wavelet_transform.py              wavelet_transform.py
  DWT (Haar) → weighted              DWT (Haar) → weighted
  magnitude map                      magnitude map
  │                                     │
  ├──────────────┐         ├──────────────┐
  ▼              ▼         ▼              ▼
Spatial Map   Invariant  Spatial Map   Invariant
(resize 8×8)  Map (REP)  (resize 8×8)  Map (REP)
  │              │         │              │
  ▼              ▼         ▼              ▼
medianhash   medianhash  medianhash   medianhash
64 bits      64 bits     64 bits      64 bits
  │              │         │              │
  └──────────────┴─────────┴──────────────┘
                           │
                           ▼
                    256-bit final hash
               [0:128]  = spatial part
               [128:256] = invariant part
```

### Hash Layout

| Bits | Source | Purpose |
|------|--------|---------|
| 0 – 63 | Edge spatial map | Crop / scale detection |
| 64 – 127 | Laplacian spatial map | Crop / scale detection |
| 128 – 191 | Edge invariant map (REP) | Rotation / flip detection |
| 192 – 255 | Laplacian invariant map (REP) | Rotation / flip detection |

---

## Key Design Decisions

### L+H Channel (preprocess.py)
Grayscale conversion discards all color information — two images that differ only in color become structurally identical to the feature extractors. DCH replaces grayscale with a weighted combination of **L (lightness)** and **H (hue)** from HLS color space:

```python
combined = (0.8 × L_normalized) + (0.2 × H_weighted)
H_weighted = H_normalized × S_normalized
```

- L at 80% preserves all structural gradient information (same as grayscale)
- H at 20% introduces color discriminability — enough to be visible, not enough to dominate
- H is multiplied by S (saturation) to suppress unreliable hue values in near-grey pixels

### Weighted DWT Subbands (wavelet_transform.py)
The Haar DWT produces three detail subbands. Equal weighting (0.33 each) was replaced with content-aware weighting:

```python
# Spatial path magnitude — HH boosted
magnitude = sqrt(0.28×LH² + 0.28×HL² + 0.44×HH²)

# Invariant path magnitude — equal weights
magnitude = sqrt(0.33×LH² + 0.33×HL² + 0.34×HH²)
```

- **LH** (horizontal edges) — fires similarly on many unrelated images. Lowest discriminative weight.
- **HL** (vertical edges) — more content-specific. Equal to LH.
- **HH** (diagonal edges) — texture, corners, complex local structure. Most content discriminative. Given highest weight on the spatial path.
- The invariant path uses equal weights because boosting HH inflates radial energy uniformly, making unrelated images look more similar in the REP.

### Radial Energy Profile — REP (wavelet_transform.py)
The invariant path originally used D4 symmetry (averaging all 8 rotations and reflections). This was too aggressive — it destroyed content information along with orientation information, causing unrelated images with similar brightness to score falsely high.

REP replaces it. The insight: **when an image is rotated or flipped, the distance of any structural feature from the center does not change — only its angle changes.**

REP computes how much energy exists at each radial distance from center, ignoring angle entirely:

```
1. Compute per-pixel distance from image center
2. Bin pixels into radial rings
3. Compute mean energy per ring
4. Normalize to [0, 1]
5. Reshape to 8×8 for medianhash
```

A cat (texture everywhere) has energy spread across all rings. A smooth sky has energy concentrated only in inner rings. These profiles are very different — and they stay identical regardless of rotation or flip.

### Local Quadrant Thresholding (medianhash.py)
Original median hash flattened the entire 8×8 map to 64 values and computed one global median. This had two problems:

1. A strong subpattern in one corner shifted the global threshold, affecting how unrelated corners were binarized
2. Flattening destroyed all spatial position information

The fix: divide the 8×8 map into 4 quadrants, compute a local median per quadrant, binarize each quadrant independently.

```
Q0 (top-left)     → local median → 16 bits
Q1 (top-right)    → local median → 16 bits
Q2 (bottom-left)  → local median → 16 bits
Q3 (bottom-right) → local median → 16 bits
Total: 64 bits — identical output size
```

---

## File Structure

```
dch/
├── preprocess.py                  # Load, L+H channel, resize
│
├── filters/
│   ├── sobel_filters.py           # Sobel X/Y gradients, edge magnitude
│   ├── laplacian_filter.py        # Laplacian for fine structural changes
│   └── texture_filters.py         # Difference of Gaussians (built, not yet wired)
│
├── feature_paths/
│   ├── edge_path.py               # Sobel feature map → wavelet
│   ├── laplacian_path.py          # Laplacian feature map → wavelet
│   └── texture_path.py            # DoG feature map → wavelet (not in pipeline yet)
│
├── spectral/
│   └── wavelet_transform.py       # DWT, weighted magnitude, REP invariant map
│
├── hashing/
│   ├── medianhash.py              # Quadrant-local median hash (64 bits per map)
│   ├── blockhash.py               # Block mean hash (built, not yet wired)
│   └── hash_concat.py             # Concatenation and hex conversion
│
├── similarity/
│   ├── hamming.py                 # Hamming distance and similarity score
│   └── rotation_match.py          # Circular bit shift matcher
│
├── pipeline.py                    # Main orchestration → 256-bit hash
└── test_dch.py                    # Test harness and comparison logic
```

---

## Usage

### Generate a hash
```python
from dch.pipeline import generate_dch

result = generate_dch("image.png")
print(result["hex"])   # 64-character hex string
print(result["bits"])  # 256-element bit list
```

### Compare two images
```python
from dch.pipeline import generate_dch
from dch.similarity.hamming import compute_similarity

base = generate_dch("original.png")
target = generate_dch("candidate.png")

# Split into specialist paths
spatial_sim  = compute_similarity(base["bits"][0:128],   target["bits"][0:128])
invariant_sim = compute_similarity(base["bits"][128:256], target["bits"][128:256])

print(f"Spatial (crop robust):    {spatial_sim * 100:.2f}%")
print(f"Invariant (flip robust):  {invariant_sim * 100:.2f}%")
```

### Run the test harness
```python
python test_dch.py
```

Place your base image as `test.png` and comparison images as `test1.png` through `test4.png` in the working directory.

---

## Dependencies

```
opencv-python
numpy
pywavelets
```

Install:
```bash
pip install opencv-python numpy PyWavelets
```

---

## Performance (57,000 image dataset)

| Metric | Result |
|--------|--------|
| True match similarity (rotation/mirror) | 80 – 90% |
| True match similarity (crop/scale) | 65 – 85% |
| False positive rate | ~15.8% (active area of improvement) |
| False negative rate | ~0.6% |
| Traditional pHash on same transforms | 60 – 70% |

The false positive rate is an active development problem. DCH is designed as a pre-filter — the remaining ambiguous cases are passed to an AI model which handles a fraction of the original dataset size.

---

## Known Limitations and Open Problems

### 1. The ambiguous zone
Extreme crops (1:10 ratio or smaller) and certain false positive pairs produce numerically identical score profiles on both paths. No combining formula has been found that resolves this ambiguity without introducing unacceptable false negatives.

### 2. Same-hue false positives
Two images sharing the same dominant color (e.g. orange cat and orange sunset) can still score moderately high because the L+H combination does not fully discriminate when hue is identical. The structural similarity dominates.

### 3. texture_path.py and blockhash.py are built but not wired
The Difference of Gaussians texture path and block mean hash implementation are complete but not integrated into `pipeline.py`. These may offer additional discriminative signal — their impact on the pipeline has not been evaluated.

### 4. Decision logic is experimental
The `max(spatial, invariant)` combining logic in `test_dch.py` is a testing construct, not a production decision. The correct production decision function — one that handles high-spatial/low-invariant, high-invariant/low-spatial, and both-moderate cases without sacrificing either false positive or false negative rate — is an open problem.

### 5. rotation_match.py is partially superseded
The circular bit shift matcher was the original rotation handling strategy before REP was introduced. Whether it still adds value in combination with REP has not been evaluated.

---

## What's Not Used and Why

| File | Status | Reason |
|------|--------|--------|
| `texture_path.py` | Built, not wired | DoG adds a third feature type but requires pipeline restructuring to integrate without increasing hash size |
| `blockhash.py` | Built, not wired | Block mean hash alternative to median hash — not evaluated |
| `rotation_match.py` | Partially superseded | REP handles rotation invariance more correctly in the wavelet domain; circular shifting was a 1D approximation of a 2D transform |

---

## Contributing / Collaboration Notes

If you are reading this as a collaborator:

- The core architecture is stable — `preprocess.py`, `wavelet_transform.py`, `medianhash.py` have been through multiple iterations and are in a good state
- The open problem is in **decision logic** — how to combine the two 128-bit path scores into a reliable final verdict
- The invariant path (REP-based) is a strong discriminator (~97% accurate on unrelated image rejection) but fails on extreme crops
- The spatial path is a strong crop detector but generates false positives
- Any solution that significantly reduces false positives without touching false negatives is a win — the target is reducing ~9,000 false positives to ~5,000–6,000 on the 57,000 image benchmark dataset

---

*DCH — built from scratch, no models, no training data, no GPU required.*
