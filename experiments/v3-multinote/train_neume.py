"""Step 2: train NeumeCNN (neume_model.py) on one crop per neume
(dataset/build_neume_crops.py): every note's step out of one model, in MEI
order, plus where the neume ends (the first NONE slot).

Same recipe as v2's train.py (Adam, lr 1e-3, weight decay 1e-4, batch 128, 30
epochs, GPU augmentation, seed 42); the loss is the cross-entropy summed over
the MAX_NOTES slots. The epoch kept is the one where the most val neumes come out
entirely right (note count and every step). Both box jitters of step 1 are
configs: in one crop per neume the notes' order, not their boxes, tells them apart.
*_hjit configs also jitter the box's left and right edges (each side's margin drawn
from MARGIN_JITTER units; build_neume_crops.py --context crops): with fixed margins
a box 0.25 unit wider on each side took in a neighbour's ink and dropped whole-neume
accuracy from 96% to 77%. *_hscale also stretches each neume horizontally by a
factor drawn log-uniformly from HSCALE_JITTER: on Salzinnes, whose notes sit further
apart, the model miscounted notes. The range is generic (+-40%, symmetric in log),
fixed before training and not tuned on Salzinnes, which stays an unseen test.

Rare shapes (contours seen < 100 times in training: 38% of their test neumes
right) and repeated pitches (0.3% of training neumes) get two fixes, chosen on
the validation and test sets only:
  *_pairs_*  adds pseudo-neumes made of two neighbouring neumes (build_neume_crops.py
             --pairs) and draws each epoch's neumes with weight (contour count)^-SHAPE_WEIGHT,
             so rare contours make up a third of the samples (single-note neumes a quarter)
  *_c16      keeps 16 horizontal bins instead of 8 (two notes side by side on one step
             may share a bin, and repeated pitches are mostly undercounted)

    python -u train_neume.py [--only neume_shift15] 2>&1 | tee checkpoints/train_neume.log
"""
import argparse
import collections
import json
import time

import numpy as np
import torch
import torch.nn as nn

from paths import CKPT
from neume_data import CROPS, CROPS_CTX, CROPS_PAIRS, MANUSCRIPTS, geom_of, load_neumes, model_input, neume_exact, \
    predict_neumes, slot_classes
from neume_model import MAX_NOTES, NONE, NeumeCNN
from train import BATCH, BBOX_SHIFT, EPOCHS, LR, SEED, WD, device, set_seed  # v2's train.py

MARGIN_JITTER = (-0.25, 0.6)  # units of margin per side: a box cutting into its ink .. taking in a neighbour's
HSCALE_JITTER = (0.7, 1.4)    # horizontal stretch of the whole neume: another scribe's spacing
CONFIGS = {
    "neume_shift15": dict(shift=1.5 / 20, crops=CROPS, margin_jitter=None),
    "neume_shift05": dict(shift=BBOX_SHIFT, crops=CROPS, margin_jitter=None),
    "neume_shift15_hjit": dict(shift=1.5 / 20, crops=CROPS_CTX, margin_jitter=MARGIN_JITTER),
    "neume_shift15_hjit_hscale": dict(shift=1.5 / 20, crops=CROPS_CTX, margin_jitter=MARGIN_JITTER,
                                      hscale_jitter=HSCALE_JITTER),
    "neume_pairs_w03": dict(shift=1.5 / 20, crops=CROPS_CTX, margin_jitter=MARGIN_JITTER, hscale_jitter=HSCALE_JITTER,
                            pairs=CROPS_PAIRS, shape_weight=0.3),
    "neume_c16": dict(shift=1.5 / 20, crops=CROPS_CTX, margin_jitter=MARGIN_JITTER, hscale_jitter=HSCALE_JITTER,
                      pool_cols=16),
    # both together; the lighter weight trades some of the rare-shape gain for fewer extra notes
    "neume_pairs_w03_c16": dict(shift=1.5 / 20, crops=CROPS_CTX, margin_jitter=MARGIN_JITTER,
                                hscale_jitter=HSCALE_JITTER, pairs=CROPS_PAIRS, shape_weight=0.3, pool_cols=16),
    "neume_pairs_w02_c16": dict(shift=1.5 / 20, crops=CROPS_CTX, margin_jitter=MARGIN_JITTER,
                                hscale_jitter=HSCALE_JITTER, pairs=CROPS_PAIRS, shape_weight=0.2, pool_cols=16),
}
FINAL = "neume_pairs_w02_c16"  # chosen on the validation set; Salzinnes was not looked at for any choice
FINAL_SPACING = "neume_shift15_hjit_hscale"  # the previous final model, without the rare-shape fixes


def contour(steps, n):
    s = [int(v) for v in steps[:n]]
    return "".join("u" if b > a else "d" if b < a else "s" for a, b in zip(s, s[1:]))


def load_model(name, dev):
    m = NeumeCNN(CONFIGS[name].get("pool_cols", 8)).to(dev)
    m.load_state_dict(torch.load(CKPT / f"{name}.pt", map_location=dev))
    return m.eval()


def train_one(name, tr, va, dev):
    c = CONFIGS[name]
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = NeumeCNN(c.get("pool_cols", 8)).to(dev)
    n_epoch = len(tr["n_notes"])  # samples per epoch: the real training neumes, pairs or not
    if c.get("pairs"):
        pr = load_neumes("train", crops_dir=c["pairs"])
        tr = {k: np.concatenate([tr[k], pr[k]]) for k in tr}
    weights = None
    if c.get("shape_weight"):
        keys = [contour(s, n) for s, n in zip(tr["steps"], tr["n_notes"])]
        count = collections.Counter(keys)
        weights = torch.tensor([count[k] ** -c["shape_weight"] for k in keys], dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss()
    X = torch.tensor(tr["crops"], device=dev)
    y = torch.tensor(slot_classes(tr["steps"]), device=dev)
    Xva = torch.tensor(va["crops"], device=dev)
    g, gva = geom_of(tr, dev), geom_of(va, dev)
    best, history = (-1.0, None), []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, total = time.time(), 0.0
        if weights is not None:
            perm = torch.multinomial(weights, n_epoch, replacement=True)
        else:
            perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            xb = model_input(c, X[b], None if g is None else tuple(v[b] for v in g), rng=rng, dev=dev)
            opt.zero_grad()
            loss = loss_fn(model(xb).reshape(-1, NONE + 1), y[b].reshape(-1)) * MAX_NOTES
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        total *= len(X) / len(perm)  # report the loss per sample as before
        steps, n = predict_neumes(model, Xva, dev, c, gva)
        ok = neume_exact(steps, n, va)
        acc = float(ok.mean())
        per = {ms: float(ok[va["manuscript"] == ms].mean()) for ms in MANUSCRIPTS}
        multi = float(ok[va["n_notes"] > 1].mean())
        history.append(dict(epoch=epoch, loss=total / len(X), val_neume_exact=acc, val_multi_exact=multi,
                            **{f"val_{k}": v for k, v in per.items()}))
        flag = ""
        if acc > best[0]:
            best = (acc, epoch)
            torch.save(model.state_dict(), CKPT / f"{name}.pt")
            flag = " *"
        print(f"  {name} epoch {epoch:2d}  loss {total / len(X):.4f}  val neumes right {100 * acc:5.2f}% "
              f"(multi-note {100 * multi:5.2f}; E {100 * per['einsiedeln']:5.2f}, M {100 * per['ms73']:5.2f})  "
              f"{time.time() - t0:4.1f}s{flag}", flush=True)
    return dict(best_epoch=best[1], best_val_neume_exact=best[0], history=history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(CONFIGS), default=list(CONFIGS))
    args = ap.parse_args()
    CKPT.mkdir(exist_ok=True)
    dev = device()
    meta_path = CKPT / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for name in args.only:
        crops = CONFIGS[name]["crops"]
        tr, va = load_neumes("train", crops_dir=crops), load_neumes("val", crops_dir=crops)
        counts = {s: {ms: int(np.sum(d["manuscript"] == ms)) for ms in MANUSCRIPTS} for s, d in (("train", tr), ("val", va))}
        print(f"training {name} on {crops.name} (device {dev}; neumes {counts})", flush=True)
        meta[name] = train_one(name, tr, va, dev)
        meta[name].update(seed=SEED, epochs=EPOCHS, batch=BATCH, lr=LR, weight_decay=WD, n_train=counts["train"],
                          crops=crops.name, shift_jitter=CONFIGS[name]["shift"],
                          margin_jitter=CONFIGS[name]["margin_jitter"], hscale_jitter=CONFIGS[name].get("hscale_jitter"),
                          pairs=str(CONFIGS[name].get("pairs") or ""), shape_weight=CONFIGS[name].get("shape_weight"),
                          pool_cols=CONFIGS[name].get("pool_cols", 8), model="NeumeCNN")
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"  {name}: best val neumes right {100 * meta[name]['best_val_neume_exact']:.2f}% "
              f"at epoch {meta[name]['best_epoch']}", flush=True)


if __name__ == "__main__":
    main()
