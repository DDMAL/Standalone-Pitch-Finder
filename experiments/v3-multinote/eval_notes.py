"""Step 1: read every note of a neume from its own MEI box, exactly as v2 reads
a single-note neume, and see what goes wrong.

Test split, every usable note. Scored per note, and per neume (all its notes
right). Note groups, within the multi-note neumes:
  size      2 / 3 / 4+ notes
  stacked   shares its column with another note of the neume (x-overlap over half
            the narrower box: a podatus's two heads); else side by side
  ligated   MEI @ligated (Einsiedeln's ligature boxes are the least reliable:
            ../mei_groundtruth_check/README.md)
  inclinatum  tilt=se (the diamonds of a climacus)
  MS73 box  original / repaired (a zero-height zone given the page's median note
            size at its top-left corner) / shared (identical to another note's box
            in the same neume -- two notes, one crop)
Models: v2's staff-free model with +-0.5 or +-1.5-step box jitter (the final
one), trained on single-note neumes (v2) or on every note (train_notes.py); and
v2's staff-based model for reference. Its crop comes from the MEI staff and only
the box's x-range, so two notes stacked in one column get the same crop.

--robustness: the staff-free models on v2's single-note test crops with the box
moved up or down 0.5 / 1 step, re-cut exactly as v2's eval_robustness.py (its
page_job, same random signs) -- what trusting the box's height costs.

    python eval_notes.py [--robustness]
        # -> results/notes_scores.csv, notes_predictions.npz (eval_neume.py reads it) / notes_robustness.csv
"""
import argparse
import collections
import csv
import multiprocessing as mp

import numpy as np
import torch

from paths import NOTES_CSV, RESULTS
from data import load, model_input  # v2's data.py
from train import MANUSCRIPTS, device, predict_steps  # v2's train.py
from train_notes import load_model

MODELS = ["cls_aug", "cls_aug_bbox128", "cls_aug_bbox128_all", "cls_aug_bbox128_shift15", "cls_aug_bbox128_shift15_all"]
BOX_MODELS = ["cls_aug_bbox128", "cls_aug_bbox128_all", "cls_aug_bbox128_shift15", "cls_aug_bbox128_shift15_all"]
STACK_OVERLAP = 0.5  # x-overlap, as a fraction of the narrower box, that counts as one column


def note_boxes():
    """{(manuscript, id): (ulx, uly, lrx, lry)} for the usable test notes."""
    out = {}
    for r in csv.DictReader(open(NOTES_CSV)):
        if r["split"] == "test" and r["kind"] == "nc" and not r["exclude"]:
            out[(r["manuscript"], r["id"])] = tuple(float(r[k]) for k in ("box_ulx", "box_uly", "box_lrx", "box_lry"))
    return out


def neume_flags(d):
    """Per note: stacked (shares a column with another note of its neume), shared (same box as one)."""
    boxes = note_boxes()
    b = np.array([boxes[(ms, i)] for ms, i in zip(d["manuscript"], d["id"])])
    stacked, shared = np.zeros(len(b), bool), np.zeros(len(b), bool)
    for idx in neumes(d).values():
        for i in idx:
            for j in idx:
                if i == j:
                    continue
                ov = min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0])
                if ov > STACK_OVERLAP * min(b[i, 2] - b[i, 0], b[j, 2] - b[j, 0]):
                    stacked[i] = True
                if np.allclose(b[i], b[j]):
                    shared[i] = True
    return stacked, shared


def neumes(d):
    """{(manuscript, page, neume_id): [row indices]}."""
    out = collections.defaultdict(list)
    for i, key in enumerate(zip(d["manuscript"], d["page"], d["neume_id"])):
        out[key].append(i)
    return out


def robustness(dev):
    import eval_robustness as R  # v2's eval_robustness.py
    models = {name: load_model(name, dev) for name in BOX_MODELS}
    by_page = collections.defaultdict(list)
    for r in csv.DictReader(open(NOTES_CSV)):
        if r["split"] == "test" and r["kind"] == "nc" and not r["exclude"]:
            by_page[(r["manuscript"], r["image"])].append(r)
    jobs = [(ms, img, rows, [r for r in rows if r["neume_size"] == "1"], ["mei128"])
            for (ms, img), rows in sorted(by_page.items())]
    truth, manuscript, preds = [], [], collections.defaultdict(list)
    with mp.Pool(6) as pool:
        for ms, steps, crops in pool.imap(R.page_job, [j for j in jobs if j[3]]):
            truth += steps
            manuscript += [ms] * len(steps)
            for name, (m, c) in models.items():
                for cond in R.BOX_CONDITIONS[:3]:  # clean, shift 0.5, shift 1
                    X = torch.tensor(crops[("mei128", cond)]["crops"], device=dev)
                    preds[(name, cond)].append(predict_steps(m, X, True, dev))
    truth, manuscript, rows = np.array(truth), np.array(manuscript), []
    print("single-note test neumes, box moved: exact (Einsiedeln, MS73)\n")
    for (name, cond), p in preds.items():
        p = np.concatenate(p) == truth
        r = dict(model=name, condition=cond, all=float(p.mean()),
                 **{ms: float(p[manuscript == ms].mean()) for ms in MANUSCRIPTS})
        rows.append(r)
        print(f"{name:28s} {cond:10s} {100 * r['all']:6.2f}%  ({100 * r['einsiedeln']:6.2f}, {100 * r['ms73']:6.2f})")
    with open(RESULTS / "notes_robustness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robustness", action="store_true")
    args = ap.parse_args()
    dev = device()
    if args.robustness:
        robustness(dev)
        return
    d, preds = None, {}
    for name in MODELS:
        m, c = load_model(name, dev)
        dc = load(MANUSCRIPTS, "test", single_note_only=False, crops_dir=c["crops"])
        if d is None:
            d = dc
        assert np.array_equal(dc["id"], d["id"]), f"{name}: crops out of order"
        preds[name] = predict_steps(m, torch.tensor(model_input(dc, c["mask"]), device=dev), c["classifier"], dev,
                                    out_h=c["out_h"])

    size = d["neume_size"].astype(int)
    multi = size > 1
    stacked, shared = neume_flags(d)
    repaired = d["box_repaired"].astype(int) == 1
    groups = []
    for ms in ("all",) + MANUSCRIPTS:
        m = np.ones(len(size), bool) if ms == "all" else d["manuscript"] == ms
        groups += [(ms, "single-note neumes", m & ~multi), (ms, "multi-note neumes", m & multi),
                   (ms, "  2 notes", m & (size == 2)), (ms, "  3 notes", m & (size == 3)),
                   (ms, "  4+ notes", m & (size >= 4)),
                   (ms, "  stacked", m & multi & stacked), (ms, "  side by side", m & multi & ~stacked),
                   (ms, "  ligated", m & multi & (d["ligated"].astype(int) == 1)),
                   (ms, "  inclinatum", m & multi & (d["tilt"] == "se"))]
        if ms == "ms73":
            groups += [(ms, "  box original", m & multi & ~repaired), (ms, "  box repaired", m & multi & repaired),
                       (ms, "  box shared", m & multi & shared)]

    rows = []
    names = list(preds)
    print("test split, every note on its own MEI box: exact (within one step)\n")
    print(f"{'':12s} {'group':22s} {'n':>6s} | " + " | ".join(f"{n:^27s}" for n in names))
    for ms, g, mask in groups:
        if mask.sum() == 0:
            continue
        cells = []
        for name in names:
            err = np.abs(preds[name][mask] - d["step"][mask])
            ex, w1 = float(np.mean(err == 0)), float(np.mean(err <= 1))
            rows.append(dict(level="note", manuscript=ms, group=g.strip(), model=name, n=int(mask.sum()), exact=ex, within1=w1))
            cells.append(f"{100 * ex:6.2f}% ({100 * w1:5.1f}%)".center(27))
        print(f"{ms:12s} {g:22s} {int(mask.sum()):6d} | " + " | ".join(cells))

    print("\nwhole neumes, every note right:\n")
    by = neumes(d)
    for ms in ("all",) + MANUSCRIPTS:
        for label, keep in (("single-note", lambda k: len(k) == 1), ("multi-note", lambda k: len(k) > 1),
                            ("  2 notes", lambda k: len(k) == 2), ("  3 notes", lambda k: len(k) == 3),
                            ("  4+ notes", lambda k: len(k) >= 4)):
            idx = [v for k, v in by.items() if (ms == "all" or k[0] == ms) and keep(v)]
            if not idx:
                continue
            cells = []
            for name in names:
                ok = float(np.mean([np.all(preds[name][v] == d["step"][v]) for v in idx]))
                rows.append(dict(level="neume", manuscript=ms, group=label.strip(), model=name, n=len(idx), exact=ok, within1=""))
                cells.append(f"{100 * ok:6.2f}%".center(27))
            print(f"{ms:12s} {label:22s} {len(idx):6d} | " + " | ".join(cells))

    out = RESULTS / "notes_scores.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(RESULTS / "notes_predictions.npz", step=d["step"], manuscript=d["manuscript"], page=d["page"],
             id=d["id"], neume_id=d["neume_id"], neume_size=d["neume_size"], stacked=stacked, shared=shared,
             **{f"pred_{k}": v for k, v in preds.items()})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
