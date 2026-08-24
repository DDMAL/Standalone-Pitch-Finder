"""Augmentation aimed at one specific failure mode: bad staff-finding.

The CNN exists because the heuristic pitch finder isn't robust to
inaccurate staff-line geometry, not because the heuristic's own math is
wrong (see ../README.md). So the augmentation that matters is whatever
reproduces what a bad staff-finding pass actually does to a crop --
`expanded_boxes()`'s vertical extent comes from the staff's own estimated
line spacing, so a wrong line-gap estimate makes that crop too tall/short
or shifted up/down; a misestimated line curvature tilts it slightly.
Generic photometric noise is secondary and included mainly for scanner/ink
variation, not because it targets the actual bottleneck.

No augmentation touches the horizontal axis or flips anything: a
notehead's identity depends on its vertical position, and flipping would
invert that.

v2 tried calibrating the vertical jitter ranges against real measurements
(measure_staff_error.py: height ratio p10/p90 of 0.37/3.37, shift std
0.475) instead of guessing. That made things worse, not better: exact-match
on the clean test set dropped from 82.6% to 68.8%, because `shift` was
sampled as a fraction of the *original* height independent of the sampled
height ratio -- at the extreme combination (small new_height, large shift)
the jittered crop can end up not overlapping the real notehead region at
all, training the model on (image, label) pairs where the image doesn't
contain the evidence for the label. Reverted to v1's untied, milder
constants below until that overlap bug is actually fixed (constrain shift
relative to the *resulting* crop, not the original one).
"""
import cv2
import numpy as np

VERTICAL_SCALE_JITTER = 0.20   # crop height scaled by up to +/-20%
VERTICAL_SHIFT_JITTER = 0.15   # crop center shifted by up to +/-15% of its own height
ROTATION_DEG = 4.0             # +/- degrees, mimics misestimated line curvature
BRIGHTNESS_JITTER = 0.15       # multiplicative
NOISE_STD = 0.03               # additive gaussian, in [0,1] pixel-value units


def jitter_bounds(top, bottom, rng: np.random.Generator):
    """Perturb a crop's (top, bottom) pixel bounds the way a wrong
    line-gap/curvature estimate from staff-finding would -- not a generic
    random crop."""
    height = bottom - top
    scale = 1.0 + rng.uniform(-VERTICAL_SCALE_JITTER, VERTICAL_SCALE_JITTER)
    shift = rng.uniform(-VERTICAL_SHIFT_JITTER, VERTICAL_SHIFT_JITTER) * height
    center = (top + bottom) / 2 + shift
    new_height = max(4.0, height * scale)
    return center - new_height / 2, center + new_height / 2


def augment_image(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small rotation + brightness jitter + light noise on an already
    cropped+resized [0,1] float image (rotating post-resize is simpler than
    rotating the source region and re-cropping, and fine at this scale)."""
    h, w = crop.shape
    angle = rng.uniform(-ROTATION_DEG, ROTATION_DEG)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    crop = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    brightness = 1.0 + rng.uniform(-BRIGHTNESS_JITTER, BRIGHTNESS_JITTER)
    crop = np.clip(crop * brightness, 0, 1)
    noise = rng.normal(0, NOISE_STD, size=crop.shape).astype(np.float32)
    return np.clip(crop + noise, 0, 1).astype(np.float32)


def build_augmented_rows(rows, n_aug, seed):
    """n_aug augmented copies of each row, re-cropped from jittered bounds
    (needs image access, so this groups rows by page and loads each page's
    image once rather than per-row). Import kept local to avoid a cycle
    with data.py at module load time."""
    from data import ROOT, crop_from_bounds
    from page_inputs import resolve_page_inputs
    import cv2 as _cv2

    rng = np.random.default_rng(seed)
    by_page = {}
    for r in rows:
        by_page.setdefault(r["page"], []).append(r)

    out = []
    for page, page_rows in by_page.items():
        inputs = resolve_page_inputs(ROOT / page)
        image = _cv2.imread(str(inputs.image), _cv2.IMREAD_GRAYSCALE)
        for r in page_rows:
            for _ in range(n_aug):
                top, bottom = jitter_bounds(r["box_top"], r["box_bottom"], rng)
                crop = crop_from_bounds(image, r["ulx"], r["ncols"], top, bottom)
                if crop is None:
                    continue
                crop = augment_image(crop, rng)
                out.append({"page": page, "class_name": r["class_name"],
                            "truth": r["truth"], "crop": crop})
    return out
