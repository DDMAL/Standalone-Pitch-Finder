"""One crop per neume (step 2): the union of its notes' MEI boxes stands in for
the one box IC gives a neume, and the label is every note's stave step, in MEI
order. Notes and their boxes come from v2's dataset/notes.csv.

  vertical    centred on the neume's box, BBOX_HALF_UNITS (10) units above and below,
              a unit being the page's median note-box height -- as v2's
              build_crops.py --mode bbox, no staff used
  horizontal  the neume's box plus MARGIN_UNITS on each side, at the vertical's scale
              (6.4 px per unit in a 128-px window), left-aligned on a WIDTH-px white
              canvas (64 px = 10 units); a wider neume is squeezed to fit (squeezed = 1)

Stored as context crops CONTEXT (1.5) times taller, as build_crops.py, for the
augmentation in data.py. Neumes of more than MAX_NOTES notes are left out (8 of
187,535 usable neumes).

--context keeps paper around the box instead, so the box's left and right edges
can be jittered in training (data.neume_batch_input): CTX_W px from LEFT_UNITS
left of the box, at the same 6.4 px per unit, with the box's columns (box_l,
box_r) and px per unit (unit_px) stored per neume. A neume too wide for the
64-px window with MAX_MARGIN units on each side is squeezed horizontally
(unit_px < 6.4, squeezed = 1).

    python build_neume_crops.py [--workers 6]   # -> data/v3_neumes128/<manuscript>_<split>.npz
    python build_neume_crops.py --context       # -> data/v3_neumes128_ctx/...
    python build_neume_crops.py --pairs         # -> data/v3_neumes128_ctx_pairs/<manuscript>_train.npz

--pairs makes training-only pseudo-neumes: two neighbouring neumes on one staff,
at most PAIR_GAP units apart and MAX_NOTES notes together, as one neume (their
boxes' union, their notes in order), cut as --context. Rare contours, long
neumes and repeated pitches (one neume ending on the pitch the next starts on)
become common, from real ink; IC merging two neumes into one glyph looks the same.

Per neume: crops (192 x 64 uint8), steps (MAX_NOTES, padded with PAD), n_notes,
page, neume_id, and flags over its notes: ligated, inclinatum (tilt se), stacked
(two notes share a column), shared (two notes share one box), repaired (MS73
zero-height box), squeezed.
"""
import argparse
import collections
import csv
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, NOTES_CSV  # noqa: E402  (also puts v2 on the import path)
from build_crops import BBOX_HALF_UNITS, CONTEXT, IMG_H  # noqa: E402  (v2's crop rules)
from neume_model import MAX_NOTES  # noqa: E402

OUT = DATA / "v3_neumes128"
OUT_CTX = DATA / "v3_neumes128_ctx"
OUT_PAIRS = DATA / "v3_neumes128_ctx_pairs"
PAIR_GAP = (-0.5, 1.0)  # --pairs: horizontal gap between the two boxes, in units (overlap allowed)
WIDTH = 64            # crop width in px: 10 units at 6.4 px per unit
MARGIN_UNITS = 0.25   # paper kept left and right of the neume's box
CTX_W = 80            # --context: crop width in px
LEFT_UNITS = 1.0      # --context: paper kept left of the box
MAX_MARGIN = 0.6      # --context: widest margin the training jitter draws, per side (train_neume.py)
PAD = -100            # steps of the empty slots
STACK_OVERLAP = 0.5   # as eval_multinote.py: x-overlap over half the narrower box = one column
CTX_H = round(IMG_H * CONTEXT)


def box(r):
    return tuple(float(r[k]) for k in ("box_ulx", "box_uly", "box_lrx", "box_lry"))


def neume_flags(notes):
    b = [box(r) for r in notes]
    stacked = shared = False
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            ov = min(b[i][2], b[j][2]) - max(b[i][0], b[j][0])
            stacked |= ov > STACK_OVERLAP * min(b[i][2] - b[i][0], b[j][2] - b[j][0])
            shared |= np.allclose(b[i], b[j])
    return dict(ligated=int(any(r["ligated"] == "1" for r in notes)),
                inclinatum=int(any(r["tilt"] == "se" for r in notes)),
                stacked=int(stacked), shared=int(shared),
                repaired=int(any(r["box_repaired"] == "1" for r in notes)))


def cut_neume(image, x0, y0, x1, y1, unit):
    """(CTX_H x WIDTH uint8 context crop, squeezed) for a neume box; outside the page is white."""
    cy = (y0 + y1) / 2
    h = 2 * BBOX_HALF_UNITS * unit * CONTEXT
    Y0, Y1 = int(round(cy - h / 2)), int(round(cy + h / 2))
    X0, X1 = int(round(x0 - MARGIN_UNITS * unit)), int(round(x1 + MARGIN_UNITS * unit))
    X1 = max(X1, X0 + 1)
    H, W = image.shape
    pad = cv2.copyMakeBorder(image[max(0, Y0):min(H, Y1), max(0, X0):min(W, X1)],
                             max(0, -Y0), max(0, Y1 - H), max(0, -X0), max(0, X1 - W),
                             cv2.BORDER_CONSTANT, value=255)
    s = CTX_H / pad.shape[0]
    w = int(round(pad.shape[1] * s))
    squeezed = w > WIDTH
    w = min(max(w, 1), WIDTH)
    out = np.full((CTX_H, WIDTH), 255, np.uint8)
    out[:, :w] = cv2.resize(pad, (w, CTX_H), interpolation=cv2.INTER_AREA)
    return out, int(squeezed)


def cut_neume_ctx(image, x0, y0, x1, y1, unit):
    """(CTX_H x CTX_W uint8 context crop, box_l, box_r, unit_px, squeezed) for a neume box:
    LEFT_UNITS of paper left of the box at unit_px px per unit (6.4, less for a neume too
    wide for the window); rows as cut_neume. Outside the page is white."""
    cy = (y0 + y1) / 2
    h = 2 * BBOX_HALF_UNITS * unit * CONTEXT
    full = CTX_H / (2 * BBOX_HALF_UNITS * CONTEXT)  # 6.4 px per unit
    unit_px = min(full, WIDTH / ((x1 - x0) / unit + 2 * MAX_MARGIN))
    X0 = x0 - LEFT_UNITS * unit
    X1 = X0 + CTX_W / unit_px * unit
    Y0, Y1 = int(round(cy - h / 2)), int(round(cy + h / 2))
    Xa, Xb = int(np.floor(X0)), int(np.ceil(X1))
    H, W = image.shape
    pad = cv2.copyMakeBorder(image[max(0, Y0):min(H, Y1), max(0, Xa):min(W, Xb)],
                             max(0, -Y0), max(0, Y1 - H), max(0, -Xa), max(0, Xb - W),
                             cv2.BORDER_CONSTANT, value=255)
    out = cv2.resize(pad, (CTX_W, CTX_H), interpolation=cv2.INTER_AREA)
    px = CTX_W / (Xb - Xa)  # output px per page px, horizontally (Xa, Xb rounded: ~unit_px / unit)
    return out, (x0 - Xa) * px, (x1 - Xa) * px, px * unit, int(unit_px < full)


def neighbour_pairs(by, unit):
    """(neume_id a + b, notes of a then b) for neighbouring neumes on one staff, close enough."""
    seq = collections.defaultdict(list)
    for nid, notes in by.items():  # dict order = document order
        seq[notes[0]["staff"]].append((nid, sorted(notes, key=lambda r: int(r["nc_index"]))))
    out = []
    for neumes in seq.values():
        for (ia, a), (ib, b) in zip(neumes, neumes[1:]):
            gap = (min(box(r)[0] for r in b) - max(box(r)[2] for r in a)) / unit
            if PAIR_GAP[0] <= gap <= PAIR_GAP[1] and len(a) + len(b) <= MAX_NOTES:
                out.append((f"{ia}+{ib}", a + b))
    return out


def page_job(args):
    manuscript, image_name, rows, context, pairs = args
    image = cv2.imread(str(DATA / "images" / manuscript / image_name), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_name)
    unit = float(np.median([box(r)[3] - box(r)[1] for r in rows]))  # as build_crops.bbox_bounds
    by = collections.defaultdict(list)
    for r in rows:
        by[r["neume_id"]].append(r)
    for nid, notes in by.items():
        if len(notes) != int(notes[0]["neume_size"]):
            raise ValueError(f"{manuscript} {image_name} neume {nid}: {len(notes)} usable of {notes[0]['neume_size']}")
    items = neighbour_pairs(by, unit) if pairs else list(by.items())
    out, skipped = [], 0
    for nid, notes in items:
        if not pairs:
            notes = sorted(notes, key=lambda r: int(r["nc_index"]))
        if len(notes) > MAX_NOTES:
            skipped += 1
            continue
        b = np.array([box(r) for r in notes])
        nb = (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())
        if context or pairs:
            crop, box_l, box_r, unit_px, squeezed = cut_neume_ctx(image, *nb, unit)
            geom = dict(box_l=box_l, box_r=box_r, unit_px=unit_px)
        else:
            (crop, squeezed), geom = cut_neume(image, *nb, unit), {}
        steps = np.full(MAX_NOTES, PAD, np.int8)
        steps[:len(notes)] = [int(r["step"]) for r in notes]
        out.append(dict(crop=crop, steps=steps, n_notes=len(notes), page=notes[0]["page"], neume_id=nid,
                        squeezed=squeezed, **geom, **neume_flags(notes)))
    return manuscript, rows[0]["split"], out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--context", action="store_true", help="wider crops + box columns, for margin jitter")
    ap.add_argument("--pairs", action="store_true", help="training pseudo-neumes from neighbouring neumes")
    args = ap.parse_args()
    out_dir = OUT_PAIRS if args.pairs else OUT_CTX if args.context else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    by_page = collections.defaultdict(list)
    for r in csv.DictReader(open(NOTES_CSV)):
        if r["kind"] == "nc" and not r["exclude"]:
            by_page[(r["manuscript"], r["image"])].append(r)
    # a neume with an excluded note would lose that note's step: drop the whole neume
    partial = collections.Counter()
    for r in csv.DictReader(open(NOTES_CSV)):
        if r["kind"] == "nc" and r["exclude"] and r["split"]:
            partial[(r["manuscript"], r["page"], r["neume_id"])] += 1
    for key, rows in by_page.items():
        by_page[key] = [r for r in rows if (r["manuscript"], r["page"], r["neume_id"]) not in partial]
    jobs = [(ms, img, rows, args.context, args.pairs) for (ms, img), rows in sorted(by_page.items())
            if rows and (not args.pairs or rows[0]["split"] == "train")]
    print(f"{sum(len(j[2]) for j in jobs)} notes on {len(jobs)} pages; "
          f"{len(partial)} neumes dropped for an excluded note", flush=True)

    acc, skipped = collections.defaultdict(list), 0
    with mp.Pool(args.workers) as pool:
        for ms, split, items, sk in pool.imap(page_job, jobs):  # imap: a fixed cache order
            acc[(ms, split)] += items
            skipped += sk
    print(f"left out {skipped} {'pairs' if args.pairs else 'neumes'} of more than {MAX_NOTES} notes")
    for (ms, split), items in sorted(acc.items()):
        path = out_dir / f"{ms}_{split}.npz"
        np.savez(path, crops=np.stack([i.pop("crop") for i in items]),
                 **{k: np.array([i[k] for i in items]) for k in items[0]})
        n = np.array([i["n_notes"] for i in items])
        print(f"{path.name}: {len(items)} neumes ({np.sum(n > 1)} multi-note, {n.sum()} notes), "
              f"squeezed {100 * np.mean([i['squeezed'] for i in items]):.2f}%, {path.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
