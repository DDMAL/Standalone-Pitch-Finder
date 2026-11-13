"""Unseen manuscript: Salzinnes (CDN-Hsmu M2149.L4), every neume, multi-note
ones included. Same 6 pages and the same MEI reading as v2's eval_ood.py (its
page_rows: images 1/4 of the MEI coordinates, C clefs in the C3 convention
raised an octave); none of the models has seen this manuscript.

  step 1  every note on its own MEI box, cut as v2's build_crops.py --mode bbox:
          v2's final single-note model and the two all-note models (train_notes.py)
  step 2  one crop per neume (dataset/build_neume_crops.py), every NeumeCNN
          config (train_neume.py)

A neume with a note v2's page_rows drops (no box, no clef, ...) is left out
whole, as build_neume_crops.py does on the training manuscripts.

    python eval_salzinnes.py   # -> results/salzinnes_scores.csv
"""
import collections
import csv
import re
import sys

import numpy as np
import torch

from paths import HERE, RESULTS
from build_crops import CONTEXT, bbox_bounds, cut  # v2's crop rules
from eval_ood import IMG_DIR, MEI_DIR, page_rows  # v2's eval_ood.py
from neume_data import PAD, geom_of, neume_exact, predict_neumes
from neume_model import MAX_NOTES
from train import device, predict_steps  # v2's train.py
from train_neume import CONFIGS as NEUME_CONFIGS, CROPS_CTX, load_model as load_neume_model
from train_notes import load_model as load_note_model

sys.path.insert(0, str(HERE / "dataset"))
from build_neume_crops import cut_neume, cut_neume_ctx  # noqa: E402

NOTE_MODELS = ["cls_aug_bbox128_shift15", "cls_aug_bbox128_all", "cls_aug_bbox128_shift15_all"]


def load_pages():
    """Per neume: steps, note crops (one per note), fixed and --context neume crops."""
    neumes, dropped = [], collections.Counter()
    for img in sorted(IMG_DIR.glob("*.png")):
        folio = re.search(r"_(\d{3}[rv])_resized", img.name).group(1)
        gray, rows, drop = page_rows(MEI_DIR / f"CDN-Hsmu_M2149.L4_{folio}.mei", img)
        dropped += drop
        by = collections.defaultdict(list)
        for r in rows:
            by[r["neume_id"]].append(r)
        bounds = dict(zip((r["id"] for r in rows), bbox_bounds(rows, rows)))  # unit from every note on the page
        unit = float(np.median([r["box_lry"] - r["box_uly"] for r in rows]))
        for nid, notes in by.items():
            notes = sorted(notes, key=lambda r: r["nc_index"])
            if len(notes) != int(notes[0]["neume_size"]) or len(notes) > MAX_NOTES:
                dropped["neume_incomplete_or_long"] += 1
                continue
            b = np.array([(r["box_ulx"], r["box_uly"], r["box_lrx"], r["box_lry"]) for r in notes])
            nb = (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())
            ctx, *g, _ = cut_neume_ctx(gray, *nb, unit)
            steps = np.full(MAX_NOTES, PAD, np.int8)
            steps[:len(notes)] = [r["step"] for r in notes]
            neumes.append(dict(
                folio=folio, steps=steps, n_notes=len(notes),
                note_crops=[cut(gray, r["box_ulx"], r["box_lrx"], *bounds[r["id"]][:2], round(128 * CONTEXT))
                            for r in notes],
                crop=cut_neume(gray, *nb, unit)[0], ctx=ctx, box_l=g[0], box_r=g[1], unit_px=g[2]))
    return neumes, dropped


def main():
    dev = device()
    neumes, dropped = load_pages()
    d = {k: np.array([n[k] for n in neumes]) for k in ("steps", "n_notes", "box_l", "box_r", "unit_px")}
    n_true = d["n_notes"]
    multi = n_true > 1
    print(f"Salzinnes: {len(neumes)} neumes ({multi.sum()} multi-note, {n_true.sum()} notes); dropped {dict(dropped)}")
    has = np.arange(MAX_NOTES)[None] < n_true[:, None]

    results = {}  # name -> (notes right (N, MAX_NOTES), count right or None)
    note_crops = torch.tensor(np.concatenate([np.stack(n["note_crops"]) for n in neumes]), device=dev)
    owner = np.repeat(np.arange(len(neumes)), n_true)
    slot = np.concatenate([np.arange(k) for k in n_true])
    for name in NOTE_MODELS:
        m, c = load_note_model(name, dev)
        pred = predict_steps(m, note_crops, True, dev)
        ok = np.zeros((len(neumes), MAX_NOTES), bool)
        ok[owner, slot] = pred == d["steps"][owner, slot]
        results[f"step 1: {name}"] = (ok, None)
    for name, c in NEUME_CONFIGS.items():
        m = load_neume_model(name, dev)
        if c["crops"] == CROPS_CTX:
            steps, n = predict_neumes(m, torch.tensor(np.stack([x["ctx"] for x in neumes]), device=dev), dev, c,
                                      geom_of(d, dev))
        else:
            steps, n = predict_neumes(m, torch.tensor(np.stack([x["crop"] for x in neumes]), device=dev), dev, c)
        results[f"step 2: {name}"] = (has & (steps == d["steps"]), n == n_true)
        assert np.array_equal(neume_exact(steps, n, d), np.all((has & (steps == d["steps"])) == has, 1) & (n == n_true))

    rows = []
    print(f"\n{'model':36s} | single-note neumes right | multi-note: notes right / count right / neumes right")
    for name, (ok, cnt) in results.items():
        whole = np.all(ok == has, 1) & (cnt if cnt is not None else True)
        r = dict(model=name, neumes=len(neumes), single_right=float(whole[~multi].mean()),
                 multi_notes_right=float(ok[multi].sum() / has[multi].sum()),
                 multi_count_right="" if cnt is None else float(cnt[multi].mean()),
                 multi_neumes_right=float(whole[multi].mean()))
        rows.append(r)
        c = "  -  " if cnt is None else f"{100 * r['multi_count_right']:5.1f}"
        print(f"{name:36s} | {100 * r['single_right']:22.2f}% | {100 * r['multi_notes_right']:6.2f} / {c} / "
              f"{100 * r['multi_neumes_right']:6.2f}")
    with open(RESULTS / "salzinnes_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {RESULTS / 'salzinnes_scores.csv'}")


if __name__ == "__main__":
    main()
