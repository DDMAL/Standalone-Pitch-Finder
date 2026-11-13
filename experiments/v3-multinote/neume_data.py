"""Neume crops (dataset/build_neume_crops.py) -> model input, and back from the
model's slots to steps.

Two crop caches: fixed crops (the neume's box plus MARGIN_UNITS of paper each
side, left-aligned on a white 64-px canvas) and --context crops (paper around
the box, plus the box's columns), from which neume_batch_input cuts the box at
train time with each side's margin drawn at random, and at test time with the
fixed margin. Vertical augmentation as v2's data.py (height scale, shift,
rotation, brightness, noise; its constants).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from paths import DATA
from data import (AUG_PROB, BRIGHTNESS_JITTER, NOISE_STD, ROTATION_DEG, VERTICAL_SCALE_JITTER,  # v2's data.py
                  VERTICAL_SHIFT_JITTER, batch_input)
from model import IMG_H, STEP_MIN  # v2's model.py
from neume_model import MAX_NOTES, NEUME_W, NONE

CROPS = DATA / "v3_neumes128"
CROPS_CTX = DATA / "v3_neumes128_ctx"
CROPS_PAIRS = DATA / "v3_neumes128_ctx_pairs"  # training pseudo-neumes (build_neume_crops.py --pairs)
MANUSCRIPTS = ("einsiedeln", "ms73")
GEOM = ("box_l", "box_r", "unit_px")  # per-neume box columns of the --context crops
PAD = -100                            # steps of the empty slots


def load_neumes(split, manuscripts=MANUSCRIPTS, crops_dir=CROPS):
    parts = []
    for ms in manuscripts:
        d = dict(np.load(Path(crops_dir) / f"{ms}_{split}.npz"))
        d["manuscript"] = np.full(len(d["n_notes"]), ms)
        parts.append(d)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def slot_classes(steps):
    """(N, MAX_NOTES) steps padded with PAD -> class per slot, NONE for the padding."""
    steps = np.asarray(steps, int)
    return np.where(steps > PAD, steps - STEP_MIN, NONE).astype(np.int64)


def neume_batch_input(ctx_u8, box_l, box_r, unit_px, rng=None, device="cpu", out_h=IMG_H, out_w=64,
                      shift_jitter=VERTICAL_SHIFT_JITTER, margin_jitter=None, margin=0.25, widen=0.0,
                      hscale_jitter=None, hscale=1.0):
    """Whole-neume context crops (B, ctx_h, ctx_w uint8; dataset/build_neume_crops.py
    --context) -> (B, 1, out_h, out_w) model input: the neume box's columns (box_l..box_r,
    px in the context crop) plus a margin on each side, left-aligned, paper beyond.
    The margin is `margin` + `widen` units (unit_px px each) without rng; with rng and
    margin_jitter=(lo, hi), each side's is drawn from U(lo, hi) units -- a box too tight
    for its ink, or one taking in a neighbour's -- and the vertical augmentation is
    batch_input's (height scale, shift, rotation, brightness, noise).
    The neume is stretched horizontally by hscale (a fixed factor, for tests), or with
    rng and hscale_jitter=(lo, hi) by a factor drawn log-uniformly from it -- another
    scribe's spacing; never so far that the neume no longer fits the window."""
    ink = 1.0 - ctx_u8.to(device).float() / 255.0  # zero padding = paper
    if ink.dim() == 3:
        ink = ink.unsqueeze(1)
    n, _, h_in, w_in = ink.shape
    box_l, box_r, unit_px = (torch.as_tensor(v, dtype=torch.float32, device=device) for v in (box_l, box_r, unit_px))
    if rng is not None and margin_jitter is not None:
        ml, mr = (torch.tensor(rng.uniform(*margin_jitter, n), dtype=torch.float32, device=device) for _ in "lr")
    else:
        ml = mr = torch.full((n,), margin + widen, dtype=torch.float32, device=device)
    left = box_l - ml * unit_px
    width = box_r + mr * unit_px - left
    if rng is not None and hscale_jitter is not None:
        k = torch.tensor(np.exp(rng.uniform(*np.log(hscale_jitter), n)), dtype=torch.float32, device=device)
    else:
        k = torch.full((n,), float(hscale), dtype=torch.float32, device=device)
    k = torch.where(k > 1, torch.minimum(k, torch.clamp(out_w / width, min=1.0)), k)  # a stretch keeps it in the window
    if rng is None:
        s, d, a = np.ones(n), np.zeros(n), np.zeros(n)
    else:
        aug = rng.random(n) < AUG_PROB
        s = np.where(aug, 1 + rng.uniform(-VERTICAL_SCALE_JITTER, VERTICAL_SCALE_JITTER, n), 1.0)
        d = np.where(aug, rng.uniform(-shift_jitter, shift_jitter, n), 0.0)
        a = np.where(aug, np.radians(rng.uniform(-ROTATION_DEG, ROTATION_DEG, n)), 0.0)
    # pixel-space map, output centre -> input: undo the stretch k, rotate (s scales the
    # window's height), then translate so the window starts at `left`; normalized by each side's size
    c, si = (torch.tensor(v, dtype=torch.float32, device=device) for v in (np.cos(a), np.sin(a)))
    s, d = (torch.tensor(v, dtype=torch.float32, device=device) for v in (s, d))
    theta = torch.zeros((n, 2, 3), dtype=torch.float32, device=device)
    theta[:, 0, 0], theta[:, 0, 1] = c * out_w / (w_in * k), -si * s * out_h / w_in
    theta[:, 0, 2] = (2 * left + out_w / k) / w_in - 1
    theta[:, 1, 0], theta[:, 1, 1] = si * out_w / (h_in * k), c * s * out_h / h_in
    theta[:, 1, 2] = 2 * d * out_h / h_in
    grid = F.affine_grid(theta, (n, 1, out_h, out_w), align_corners=False)
    ink = F.grid_sample(ink, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    ink = ink * (torch.arange(out_w, device=device)[None, :] < (width * k)[:, None]).view(n, 1, 1, out_w)
    x = 1.0 - ink
    if rng is not None:
        aug = torch.tensor(rng.random(n) < AUG_PROB, device=device).view(-1, 1, 1, 1)
        bright = torch.tensor(1 + rng.uniform(-BRIGHTNESS_JITTER, BRIGHTNESS_JITTER, n),
                              device=device, dtype=torch.float32).view(-1, 1, 1, 1)
        x = torch.where(aug, (x * bright + torch.randn_like(x) * NOISE_STD).clamp(0, 1), x)
    return x


def model_input(cfg, crops, geom, rng=None, dev="cpu", widen=0.0, hscale=1.0):
    """Model input for a batch: fixed crops as they are, --context crops cut to their box
    (margins and spacing jittered when training; the fixed margin + widen units and the
    hscale stretch when not)."""
    if geom is None:
        assert hscale == 1.0, "fixed crops can't be stretched"
        return batch_input(crops, rng=rng, device=dev, shift_jitter=cfg["shift"], out_w=NEUME_W)
    return neume_batch_input(crops, *geom, rng=rng, device=dev, out_w=NEUME_W, shift_jitter=cfg["shift"],
                             margin_jitter=cfg["margin_jitter"], widen=widen,
                             hscale_jitter=cfg.get("hscale_jitter"), hscale=hscale)


def geom_of(d, dev):
    """The --context crops' box columns as device tensors, or None for fixed crops."""
    return tuple(torch.tensor(d[k], dtype=torch.float32, device=dev) for k in GEOM) if "box_l" in d else None


def predict_neumes(model, crops, dev, cfg=None, geom=None, widen=0.0, hscale=1.0, batch=1024):
    """(N, MAX_NOTES) predicted steps (PAD from the first NONE slot on) and note counts.
    geom (box_l, box_r, unit_px) for --context crops, with widen units added per side
    and the neume stretched horizontally by hscale."""
    cfg = cfg or dict(shift=0.0, margin_jitter=None)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            g = None if geom is None else tuple(v[i:i + batch] for v in geom)
            out.append(model(model_input(cfg, crops[i:i + batch], g, dev=dev, widen=widen, hscale=hscale))
                       .argmax(2).cpu().numpy())
    cls = np.concatenate(out)
    is_none = cls == NONE
    n = np.where(is_none.any(1), is_none.argmax(1), MAX_NOTES)
    steps = np.where(np.arange(MAX_NOTES)[None] < n[:, None], cls + STEP_MIN, PAD)
    return steps, n


def neume_exact(pred_steps, pred_n, d):
    """Per neume: the right number of notes, each on the right step."""
    return (pred_n == d["n_notes"]) & np.all(pred_steps == np.where(d["steps"] > PAD, d["steps"], PAD), 1)
