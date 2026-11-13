"""Step 2: score NeumeCNN on the test neumes, next to
step 1 on exactly the same neumes.

  step 1  every note read from its own MEI box (eval_notes.py, models trained
          on every note: cls_aug_bbox128_all / cls_aug_bbox128_shift15_all) --
          the upper bound of one box per note
  step 2  one crop per neume, every note's step out of one model (train_neume.py)

Scores: notes right (a note counts only if its slot holds the right step, so a
note the model left out is wrong), note count right (step 2 only), neumes
entirely right (every step right and no note missing or extra). Groups as eval_notes.py, per neume: stacked / ligated /
inclinatum mean at least one such note; MS73 repaired / shared, likewise.

--robustness re-cuts the test neumes from the page images with the neume's box
moved up or down by 0.5 / 1 step (sign random per neume, as eval_robustness.py)
or widened / narrowed by 0.1 - 0.5 unit on each side, and scores the step-2 models on
them; and stretches each neume horizontally by x0.8 / x1.25, another scribe's spacing
(--context models only: a fixed crop has no room to stretch into).

    python eval_neume.py [--robustness]   # needs results/notes_predictions.npz (eval_notes.py)
                                          # -> results/neume_scores.csv, neume_robustness.csv
"""
import argparse
import collections
import csv
import sys
import zlib

import cv2
import numpy as np
import torch

from paths import CKPT, HERE, NOTES_CSV, RESULTS
from neume_data import CROPS_CTX, MANUSCRIPTS, geom_of, load_neumes, neume_exact, predict_neumes
from neume_model import MAX_NOTES
from train import device  # v2's train.py
from train_neume import CONFIGS, load_model

sys.path.insert(0, str(HERE / "dataset"))
from build_neume_crops import CTX_W, DATA, box, cut_neume, cut_neume_ctx  # noqa: E402

CONDITIONS = {"clean": (0.0, 0.0, 1.0), "shift 0.5": (0.5, 0.0, 1.0), "shift 1": (1.0, 0.0, 1.0),
              "narrower 0.25": (0.0, -0.25, 1.0), "narrower 0.1": (0.0, -0.1, 1.0),
              "wider 0.1": (0.0, 0.1, 1.0), "wider 0.25": (0.0, 0.25, 1.0), "wider 0.5": (0.0, 0.5, 1.0),
              "spacing x0.8": (0.0, 0.0, 0.8), "spacing x1.25": (0.0, 0.0, 1.25)}

STEP1 = ["cls_aug_bbox128_all", "cls_aug_bbox128_shift15_all"]


def step1_notes_right(d):
    """{model: (N, MAX_NOTES) bool} from eval_notes.py's predictions, aligned to d's neumes."""
    z = np.load(RESULTS / "notes_predictions.npz", allow_pickle=True)
    idx = collections.defaultdict(list)
    for i, key in enumerate(zip(z["manuscript"], z["page"].astype(str), z["neume_id"])):
        idx[key].append(i)  # npz rows of a neume are in note order (notes.csv order)
    out = {}
    for name in STEP1:
        if f"pred_{name}" not in z:
            print(f"WARNING: no step-1 predictions for {name}, left out")
            continue
        right = z[f"pred_{name}"] == z["step"]
        ok = np.zeros((len(d["n_notes"]), MAX_NOTES), bool)
        for n, key in enumerate(zip(d["manuscript"], d["page"].astype(str), d["neume_id"])):
            rows = idx[key]
            assert len(rows) == d["n_notes"][n] and np.array_equal(z["step"][rows], d["steps"][n, :len(rows)]), key
            ok[n, :len(rows)] = right[rows]
        out[name] = ok
    return out


_TEST_ROWS = {}


def test_rows():
    if not _TEST_ROWS:
        for r in csv.DictReader(open(NOTES_CSV)):
            if r["split"] == "test" and r["kind"] == "nc" and not r["exclude"]:
                _TEST_ROWS.setdefault((r["manuscript"], r["image"]), []).append(r)
    return _TEST_ROWS


def recut(d, shift, widen, context=False):
    """d's neumes cut again from the page images, box moved by shift steps (sign random
    per neume) and widened by widen units each side; a unit is one step (build_crops.py).
    context: --context crops (widen is then left to predict_neumes) -> (crops, geom arrays)."""
    want = {key: i for i, key in enumerate(zip(d["manuscript"], d["page"].astype(str), d["neume_id"]))}
    out = np.zeros((len(d["crops"]), d["crops"].shape[1], CTX_W) if context else d["crops"].shape, np.uint8)
    geom = np.zeros((3, len(out)), np.float32)
    done = 0
    for (ms, image), page_rows in test_rows().items():
        gray = cv2.imread(str(DATA / "images" / ms / image), cv2.IMREAD_GRAYSCALE)
        unit = float(np.median([box(r)[3] - box(r)[1] for r in page_rows]))
        by = collections.defaultdict(list)
        for r in page_rows:
            by[r["neume_id"]].append(r)
        for nid, notes in by.items():
            i = want.get((ms, notes[0]["page"], nid))
            if i is None:  # dropped by build_neume_crops.py
                continue
            b = np.array([box(r) for r in notes])
            sign = 1 if zlib.crc32(f"{ms}:{nid}".encode()) % 2 else -1
            dy = sign * shift * unit
            if context:
                out[i], *g, _ = cut_neume_ctx(gray, b[:, 0].min(), b[:, 1].min() + dy, b[:, 2].max(), b[:, 3].max() + dy,
                                              unit)
                geom[:, i] = g
            else:
                out[i], _ = cut_neume(gray, b[:, 0].min() - widen * unit, b[:, 1].min() + dy,
                                      b[:, 2].max() + widen * unit, b[:, 3].max() + dy, unit)
            done += 1
    assert done == len(out), f"re-cut {done} of {len(out)} neumes"
    return (out, dict(zip(("box_l", "box_r", "unit_px"), geom))) if context else out


def test_data(name, d, cache={}):
    """The test neumes in the crops a config was trained on (same neumes, same order as d)."""
    crops = CONFIGS[name]["crops"]
    if crops not in cache:
        cache[crops] = d if crops != CROPS_CTX else load_neumes("test", crops_dir=crops)
        assert np.array_equal(cache[crops]["neume_id"], d["neume_id"]), f"{name}: neumes out of order"
    return cache[crops]


def run(name, m, d, dev, shift=0.0, widen=0.0, hscale=1.0):
    """(steps, n) of a config's model on d's neumes, the box moved by shift steps / widened by
    widen units, the neume stretched by hscale; None for a fixed-crop model asked to stretch."""
    c = CONFIGS[name]
    if c["crops"] != CROPS_CTX and hscale != 1.0:
        return None
    if c["crops"] == CROPS_CTX:
        dc = test_data(name, d)
        if shift:
            crops, g = recut(d, shift, 0.0, context=True)
            dc = dict(dc, crops=crops, **g)
        return predict_neumes(m, torch.tensor(dc["crops"], device=dev), dev, c, geom_of(dc, dev), widen=widen,
                              hscale=hscale)
    crops = d["crops"] if not (shift or widen) else recut(d, shift, widen)
    return predict_neumes(m, torch.tensor(crops, device=dev), dev, c)


def robustness(d, dev):
    models = {name: load_model(name, dev) for name in CONFIGS if (CKPT / f"{name}.pt").exists()}
    multi, rows = d["n_notes"] > 1, []
    print("\nbox errors, neumes entirely right: all / multi-note (Einsiedeln, MS73)\n")
    for cond, (shift, widen, hscale) in CONDITIONS.items():
        cells = []
        for name, m in models.items():
            out = run(name, m, d, dev, shift, widen, hscale)
            if out is None:
                cells.append(f"{name}: -")
                continue
            ok = neume_exact(*out, d)
            r = dict(condition=cond, model=name, all=float(ok.mean()), multi=float(ok[multi].mean()),
                     **{f"multi_{ms}": float(ok[multi & (d["manuscript"] == ms)].mean()) for ms in MANUSCRIPTS})
            rows.append(r)
            cells.append(f"{name}: {100 * r['all']:6.2f} / {100 * r['multi']:6.2f} "
                         f"({100 * r['multi_einsiedeln']:5.2f}, {100 * r['multi_ms73']:5.2f})")
        print(f"{cond:13s} | " + " | ".join(cells), flush=True)
    with open(RESULTS / "neume_robustness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robustness", action="store_true")
    args = ap.parse_args()
    dev = device()
    d = load_neumes("test")
    if args.robustness:
        robustness(d, dev)
        return
    n_true = d["n_notes"]
    has = np.arange(MAX_NOTES)[None] < n_true[:, None]
    notes_right, count_right = step1_notes_right(d), {}
    for name in CONFIGS:
        if not (CKPT / f"{name}.pt").exists():
            print(f"WARNING: no checkpoint for {name}, left out")
            continue
        steps, n = run(name, load_model(name, dev), d, dev)
        notes_right[name] = has & (steps == d["steps"])
        count_right[name] = n == n_true

    groups = []
    for ms in ("all",) + MANUSCRIPTS:
        m = np.ones(len(n_true), bool) if ms == "all" else d["manuscript"] == ms
        groups += [(ms, "single-note", m & (n_true == 1)), (ms, "multi-note", m & (n_true > 1)),
                   (ms, "  2 notes", m & (n_true == 2)), (ms, "  3 notes", m & (n_true == 3)),
                   (ms, "  4+ notes", m & (n_true >= 4)),
                   (ms, "  with stacked notes", m & (n_true > 1) & (d["stacked"] == 1)),
                   (ms, "  ligated", m & (n_true > 1) & (d["ligated"] == 1)),
                   (ms, "  with inclinata", m & (n_true > 1) & (d["inclinatum"] == 1))]
        if ms == "ms73":
            groups += [(ms, "  box repaired", m & (n_true > 1) & (d["repaired"] == 1)),
                       (ms, "  box shared", m & (n_true > 1) & (d["shared"] == 1))]

    names = list(notes_right)
    rows = []
    print("test neumes: notes right / note count right / neumes entirely right\n")
    print(f"{'':11s} {'group':22s} {'neumes':>6s} | " + " | ".join(f"{n:^25s}" for n in names))
    for ms, g, mask in groups:
        if not mask.any():
            continue
        cells = []
        for name in names:
            ok = notes_right[name][mask]
            note_acc = float(ok.sum() / has[mask].sum())
            whole = np.all(ok == has[mask], 1)
            if name in count_right:  # an extra predicted note is no right note's slot: check the count too
                whole &= count_right[name][mask]
            neume_acc = float(whole.mean())
            cnt = float(count_right[name][mask].mean()) if name in count_right else None
            rows.append(dict(manuscript=ms, group=g.strip(), model=name, neumes=int(mask.sum()), notes=int(has[mask].sum()),
                             notes_right=note_acc, count_right="" if cnt is None else cnt, neumes_right=neume_acc))
            cells.append(f"{100 * note_acc:6.2f} / {'  -  ' if cnt is None else f'{100 * cnt:5.1f}'} / {100 * neume_acc:6.2f}")
        print(f"{ms:11s} {g:22s} {int(mask.sum()):6d} | " + " | ".join(cells))
    out = RESULTS / "neume_scores.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
