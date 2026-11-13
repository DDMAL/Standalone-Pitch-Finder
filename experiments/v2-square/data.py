"""Load the cached crops (dataset/build_crops.py) and turn them into model
input, with v1's staff-error augmentation done on the GPU.

Each cached crop is a context crop CONTEXT (1.5) times taller than the model
window. The un-augmented input is its central window. The augmented one is
v1's recipe (../v1-hfngl/augment.py), same constants: the window's height
scaled by up to +-20% and its centre shifted by up to +-15% of its height
(what a wrong line-gap estimate does to a staff-derived crop), then a
rotation of up to +-4 degrees, brightness +-15% and gaussian noise. v1
pre-generated 3 augmented copies per crop; here every epoch draws fresh ones,
and AUG_PROB keeps v1's 1-in-4 share of un-augmented crops.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import IMG_H, IMG_W, STEP_MIN

ROOT = Path(__file__).resolve().parents[2]
CROPS = ROOT / "data" / "v2_crops"
CONTEXT = 1.5                  # dataset/build_crops.py
VERTICAL_SCALE_JITTER = 0.20   # v1 augment.py
VERTICAL_SHIFT_JITTER = 0.15
ROTATION_DEG = 4.0
BRIGHTNESS_JITTER = 0.15
NOISE_STD = 0.03
AUG_PROB = 0.75                # v1: real + 3 augmented copies


def load(manuscripts, split, single_note_only=True, crops_dir=CROPS):
    """dict of numpy arrays (crops, step, meta columns, manuscript) over the given manuscripts."""
    parts = []
    for ms in manuscripts:
        d = dict(np.load(Path(crops_dir) / f"{ms}_{split}.npz"))
        keep = d["neume_size"].astype(int) == 1 if single_note_only else np.ones(len(d["step"]), bool)
        d = {k: v[keep] for k, v in d.items()}
        d["manuscript"] = np.full(keep.sum(), ms)
        parts.append(d)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def affine(n, rng=None, device="cpu", out_h=IMG_H, shift_jitter=VERTICAL_SHIFT_JITTER, out_w=IMG_W):
    """(n, 2, 3) affine_grid thetas mapping the model window into the context crop.
    shift_jitter: max centre shift as a fraction of the window. v1's 0.15 suits staff
    crops (a misplaced staff); a staff-free crop is centred on its note, so a large
    shift would swap the target note for a neighbour -- those use a smaller one."""
    if rng is None:
        s, d, a = np.ones(n), np.zeros(n), np.zeros(n)
    else:
        aug = rng.random(n) < AUG_PROB
        s = np.where(aug, 1 + rng.uniform(-VERTICAL_SCALE_JITTER, VERTICAL_SCALE_JITTER, n), 1.0)
        d = np.where(aug, rng.uniform(-shift_jitter, shift_jitter, n), 0.0)
        a = np.where(aug, np.radians(rng.uniform(-ROTATION_DEG, ROTATION_DEG, n)), 0.0)
    # rotation is in output *pixels* (as v1's post-resize warpAffine), so the
    # normalized axes are rescaled by the window's aspect ratio around it
    c, si, ar = np.cos(a), np.sin(a), out_h / out_w
    theta = np.zeros((n, 2, 3), np.float32)
    theta[:, 0, 0], theta[:, 0, 1] = c, -si * ar
    theta[:, 1, 0], theta[:, 1, 1] = s / CONTEXT * si / ar, s / CONTEXT * c
    theta[:, 1, 2] = d / (CONTEXT / 2)  # shift as a fraction of the window, in context units
    return torch.tensor(theta, device=device)


def batch_input(crops_u8, rng=None, device="cpu", out_h=IMG_H, shift_jitter=VERTICAL_SHIFT_JITTER, out_w=IMG_W):
    """uint8 (B, ctx_h, out_w) or (B, C, ctx_h, out_w) tensor on device -> float
    (B, C, out_h, out_w) model input. Channel 0 is the image; any further channel
    (the box mask) gets the same geometry but no brightness or noise. out_w is the
    crops' width: IMG_W for a note, wider for a whole neume (../v3-multinote)."""
    x = crops_u8.to(device).float() / 255.0
    if x.dim() == 3:
        x = x.unsqueeze(1)
    theta = affine(len(x), rng, device, out_h, shift_jitter, out_w)
    grid = F.affine_grid(theta, (len(x), x.shape[1], out_h, out_w), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    if rng is not None:
        aug = torch.tensor(rng.random(len(x)) < AUG_PROB, device=device).view(-1, 1, 1, 1)
        bright = torch.tensor(1 + rng.uniform(-BRIGHTNESS_JITTER, BRIGHTNESS_JITTER, len(x)),
                              device=device, dtype=torch.float32).view(-1, 1, 1, 1)
        img = x[:, :1]
        noisy = (img * bright + torch.randn_like(img) * NOISE_STD).clamp(0, 1)
        x = torch.cat([torch.where(aug, noisy, img), x[:, 1:]], 1)
    return x


def model_input(d, mask=False):
    """The cached array a config trains on: crops, or crops + box masks as 2 channels."""
    return np.stack([d["crops"], d["masks"]], 1) if mask else d["crops"]


def class_of(step):
    return (np.asarray(step, int) - STEP_MIN).astype(np.int64)
