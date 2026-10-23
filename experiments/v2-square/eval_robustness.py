"""How much does each model lose when what it relies on is wrong?

Test split, single-note neumes. Crops are re-cut from the page images under
controlled errors, so every condition sees exactly the same notes:

  staff model (cls_aug, crops from the staff), the MEI staff distorted per staff:
    shift   the whole staff moved up or down by 0.5 / 1 / 2 steps (sign random per staff)
    gap     line spacing misestimated x0.8 / x1.2 / x1.5 (around the staff's centre)
    missing the top or the bottom line not detected (the crop is built from the other three)
  staff-free models (crops centred on the note's box), the box distorted per note:
    shift   box off by 0.5 / 1 step (sign random per note)
    unit    page scale (median note-box height) misestimated x0.8 / x1.2
    on MEI boxes (head only) for the bbox models, on IC-style boxes (a virga's box
    runs down its stem, dataset/build_crops.ic_like_box) for the ic models; a shifted
    box moves the crop and, for the *_mask models, the box channel with it.

The label is always the true step on the true staff. Staff-free models never
see the staff, so the staff conditions don't apply to them, and vice versa.
Models without a checkpoint are skipped with a warning.

    python eval_robustness.py
"""
import collections
import csv
import multiprocessing as mp
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dataset"))
from build_crops import BBOX_HALF_UNITS, CONTEXT, IMG_W, MARGIN_STEPS, cut, ic_like_box, staff_span  # noqa: E402

from data import model_input  # noqa: E402
from train import CKPT, CONFIGS, MANUSCRIPTS, build_model, device, predict_steps  # noqa: E402

DATA = HERE.parents[1] / "data"
SEED = 42
STAFF_CONDITIONS = ["clean", "shift 0.5", "shift 1", "shift 2", "gap x0.8", "gap x1.2", "gap x1.5",
                    "missing top line", "missing bottom line"]
BOX_CONDITIONS = ["clean", "shift 0.5", "shift 1", "unit x0.8", "unit x1.2"]
GROUPS = {  # crop family -> which box, which conditions, window height, models
    "staff": dict(box="mei", conditions=STAFF_CONDITIONS, out_h=128, models=["cls_aug"]),
    "mei128": dict(box="mei", conditions=BOX_CONDITIONS, out_h=128,
                   models=["cls_aug_bbox128", "cls_aug_bbox128_shift15"]),
    "mei256": dict(box="mei", conditions=BOX_CONDITIONS, out_h=256, models=["cls_aug_bbox256"]),
    "ic128": dict(box="ic", conditions=BOX_CONDITIONS, out_h=128,
                  models=["cls_aug_ic128", "cls_aug_ic128_mask", "cls_aug_ic128_mask_shift15"]),
}


def staff_bounds(r, cond, sign):
    x = (float(r["box_ulx"]) + float(r["box_lrx"])) / 2
    top, bottom, hg = staff_span(r, x)
    kind, _, val = cond.partition(" ")
    if kind == "shift":
        d = sign * float(val) * hg
        top, bottom = top + d, bottom + d
    elif kind == "gap":
        s, c = float(val[1:]), (top + bottom) / 2
        top, bottom, hg = c - s * (c - top), c + s * (bottom - c), s * hg
    elif cond == "missing top line":
        top = top + 2 * hg
    elif cond == "missing bottom line":
        bottom = bottom - 2 * hg
    t, b = top - MARGIN_STEPS * hg, bottom + MARGIN_STEPS * hg
    y0, y1 = float(r["box_uly"]), float(r["box_lry"])
    return min(t, y0), max(b, y1)  # as build_crops: widened to the note's box if it sticks out


def box_crop(gray, box, cond, sign, unit, hg, ctx_h):
    """(crop, mask) for a staff-free model; the box itself is what the condition moves."""
    x0, y0, x1, y1 = box
    kind, _, val = cond.partition(" ")
    if kind == "shift":
        d = sign * float(val) * hg
        y0, y1 = y0 + d, y1 + d
    elif kind == "unit":
        unit *= float(val[1:])
    cy = (y0 + y1) / 2
    t, b = cy - BBOX_HALF_UNITS * unit, cy + BBOX_HALF_UNITS * unit
    c0 = cy - CONTEXT * (b - t) / 2
    to_row = lambda y: int(round((y - c0) / (CONTEXT * (b - t)) * ctx_h))
    mask = np.zeros((ctx_h, IMG_W), np.uint8)
    mask[max(0, to_row(y0)):max(0, to_row(y1)) + 1] = 255
    return cut(gray, x0, x1, t, b, ctx_h), mask


def page_job(args):
    ms, image_name, page_rows, singles, groups = args
    gray = cv2.imread(str(DATA / "images" / ms / image_name), cv2.IMREAD_GRAYSCALE)
    rng = np.random.default_rng([SEED, zlib.crc32(image_name.encode())])  # hash() differs per process
    staff_sign = {s: rng.choice((-1, 1)) for s in sorted({r["staff"] for r in singles})}
    note_sign = rng.choice((-1, 1), len(singles))
    unit = float(np.median([float(r["box_lry"]) - float(r["box_uly"]) for r in page_rows]))
    mei_boxes = [tuple(float(r[k]) for k in ("box_ulx", "box_uly", "box_lrx", "box_lry")) for r in singles]
    ic_boxes = [ic_like_box(gray, r, unit)[:4] for r in singles] if "ic128" in groups else None
    out = {}
    for g in groups:
        spec = GROUPS[g]
        ctx_h = round(spec["out_h"] * CONTEXT)
        for cond in spec["conditions"]:
            if g == "staff":
                crops = [cut(gray, b[0], b[2], *staff_bounds(r, cond, staff_sign[r["staff"]]), ctx_h)
                         for r, b in zip(singles, mei_boxes)]
                out[(g, cond)] = {"crops": np.stack(crops)}
            else:
                boxes = ic_boxes if spec["box"] == "ic" else mei_boxes
                pairs = [box_crop(gray, b, cond, note_sign[k], unit, staff_span(r, 0)[2], ctx_h)
                         for k, (r, b) in enumerate(zip(singles, boxes))]
                out[(g, cond)] = {"crops": np.stack([p[0] for p in pairs]), "masks": np.stack([p[1] for p in pairs])}
    return ms, [int(r["step"]) for r in singles], out


def main():
    dev = device()
    models, groups = {}, []
    for g, spec in GROUPS.items():
        for name in spec["models"]:
            if not (CKPT / f"{name}.pt").exists():
                print(f"WARNING: no checkpoint for {name}, skipped", flush=True)
                continue
            m = build_model(name).to(dev)
            m.load_state_dict(torch.load(CKPT / f"{name}.pt", map_location=dev))
            models[name] = m
            if g not in groups:
                groups.append(g)

    by_page = collections.defaultdict(list)
    for r in csv.DictReader(open(HERE / "dataset" / "notes.csv")):
        if r["split"] == "test" and r["kind"] == "nc" and not r["exclude"]:
            by_page[(r["manuscript"], r["image"])].append(r)
    jobs = [(ms, img, rows, [r for r in rows if r["neume_size"] == "1"], groups)
            for (ms, img), rows in sorted(by_page.items())]
    jobs = [j for j in jobs if j[3]]

    truth, manuscript = [], []
    preds = collections.defaultdict(list)
    with mp.Pool(6) as pool:
        for n, (ms, steps, crops) in enumerate(pool.imap(page_job, jobs), 1):
            truth += steps
            manuscript += [ms] * len(steps)
            for g in groups:
                spec = GROUPS[g]
                for name in [m for m in spec["models"] if m in models]:
                    for cond in spec["conditions"]:
                        X = torch.tensor(model_input(crops[(g, cond)], CONFIGS[name]["mask"]), device=dev)
                        preds[(name, cond)].append(predict_steps(models[name], X, True, dev, out_h=spec["out_h"]))
            if n % 25 == 0:
                print(f"  {n}/{len(jobs)} pages", flush=True)

    truth, manuscript = np.array(truth), np.array(manuscript)
    rows = []
    for (name, cond), p in preds.items():
        p = np.concatenate(p)
        for ms in ("all",) + MANUSCRIPTS:
            mask = np.ones(len(truth), bool) if ms == "all" else manuscript == ms
            err = np.abs(p[mask] - truth[mask])
            rows.append(dict(model=name, condition=cond, manuscript=ms, n=int(mask.sum()),
                             exact=float(np.mean(err == 0)), within1=float(np.mean(err <= 1))))
    out = HERE / "results" / "robustness.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'model':28s} {'condition':20s} | {'all':>7s} {'Einsied.':>8s} {'MS73':>7s} | within 1 (all)")
    for g in groups:
        for name in [m for m in GROUPS[g]["models"] if m in models]:
            for cond in GROUPS[g]["conditions"]:
                r = {x["manuscript"]: x for x in rows if x["model"] == name and x["condition"] == cond}
                print(f"{name:28s} {cond:20s} | {100 * r['all']['exact']:6.2f}% {100 * r['einsiedeln']['exact']:7.2f}% "
                      f"{100 * r['ms73']['exact']:6.2f}% | {100 * r['all']['within1']:6.2f}%")
            print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
