"""Train v1's four models on v2's square-notation data (single-note neumes,
Einsiedeln + MS73 together) and save the best-on-val checkpoint of each.

Same as v1: architecture, optimizer (Adam, lr 1e-3, weight decay 1e-4),
loss (MSE / cross-entropy), augmentation recipe, seed. Different, because
there is ~80x more data (76k training crops vs 958): mini-batches of 128
instead of 16, EPOCHS passes instead of 300, fresh augmentation every epoch
instead of 3 fixed copies, and the epoch kept is the one with the best val
exact-match (v1 had no val split).

The *_bbox* configs are the staff-free comparison: crops centred on the note's
box, 20 note-box heights tall, no staff used anywhere (dataset/build_crops.py
--mode bbox). Their augmentation shifts by +-0.5 step instead of +-15% of the
window (see data.affine); bbox256 keeps the staff crops' 12.8 px per step, with
16 pooled rows so each bin still covers 16 px.

    python -u train.py [--only cls_aug] 2>&1 | tee checkpoints/train.log
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import CROPS, ROOT, batch_input, class_of, load, model_input
from model import STEP_MAX, STEP_MIN, StepCNN, StepCNNClassifier

HERE = Path(__file__).resolve().parent
CKPT = HERE / "checkpoints"
MANUSCRIPTS = ("einsiedeln", "ms73")
SEED = 42
EPOCHS = 30
BATCH = 128
LR, WD = 1e-3, 1e-4
BBOX_SHIFT = 0.025  # +-0.5 step of a 20-step staff-free window


def _cfg(classifier, augment, crops=CROPS, out_h=128, pool=8, shift=0.15, mask=False):
    return dict(classifier=classifier, augment=augment, crops=crops, out_h=out_h, pool=pool, shift=shift, mask=mask)


CONFIGS = {
    "reg": _cfg(False, False), "reg_aug": _cfg(False, True),
    "cls": _cfg(True, False), "cls_aug": _cfg(True, True),
    "cls_aug_bbox128": _cfg(True, True, ROOT / "data" / "v2_crops_bbox128", 128, 8, BBOX_SHIFT),
    "cls_aug_bbox256": _cfg(True, True, ROOT / "data" / "v2_crops_bbox256", 256, 16, BBOX_SHIFT),
    # same as bbox128 but trained with +-1.5 step centre shifts, so the target isn't always dead centre
    "cls_aug_bbox128_shift15": _cfg(True, True, ROOT / "data" / "v2_crops_bbox128", 128, 8, 1.5 / 20),
    # IC-style boxes (a virga's box runs down its stem; dataset/build_crops.py --mode icbox):
    # centre only / + the box as a 2nd input channel / + that and +-1.5 step shifts
    "cls_aug_ic128": _cfg(True, True, ROOT / "data" / "v2_crops_icbox128", 128, 8, BBOX_SHIFT),
    "cls_aug_ic128_mask": _cfg(True, True, ROOT / "data" / "v2_crops_icbox128", 128, 8, BBOX_SHIFT, mask=True),
    "cls_aug_ic128_mask_shift15": _cfg(True, True, ROOT / "data" / "v2_crops_icbox128", 128, 8, 1.5 / 20, mask=True),
}


def build_model(name):
    c = CONFIGS[name]
    return StepCNNClassifier(c["pool"], 2 if c["mask"] else 1) if c["classifier"] else StepCNN()


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict_steps(model, crops, classifier, dev, batch=1024, out_h=128):
    """Integer step predictions for uint8 crops (no augmentation)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            y = model(batch_input(crops[i:i + batch], device=dev, out_h=out_h))
            if classifier:
                out.append(y.argmax(1).cpu().numpy() + STEP_MIN)
            else:
                out.append(np.clip(np.rint(y.cpu().numpy()), STEP_MIN, STEP_MAX).astype(int))
    return np.concatenate(out)


def exact_by_manuscript(pred, d):
    return {ms: float(np.mean(pred[d["manuscript"] == ms] == d["step"][d["manuscript"] == ms]))
            for ms in MANUSCRIPTS}


def train_one(name, tr, va, dev):
    c = CONFIGS[name]
    classifier, augment = c["classifier"], c["augment"]
    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = build_model(name).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss() if classifier else nn.MSELoss()
    X = torch.tensor(model_input(tr, c["mask"]), device=dev)
    y = torch.tensor(class_of(tr["step"]) if classifier else tr["step"].astype(np.float32), device=dev)
    Xva = torch.tensor(model_input(va, c["mask"]), device=dev)
    best, history = (-1.0, None), []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, total = time.time(), 0.0
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), BATCH):
            b = perm[i:i + BATCH]
            xb = batch_input(X[b], rng=rng if augment else None, device=dev, out_h=c["out_h"], shift_jitter=c["shift"])
            opt.zero_grad()
            loss = loss_fn(model(xb), y[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        pred = predict_steps(model, Xva, classifier, dev, out_h=c["out_h"])
        acc = float(np.mean(pred == va["step"]))
        per = exact_by_manuscript(pred, va)
        history.append(dict(epoch=epoch, loss=total / len(X), val_exact=acc, **{f"val_{k}": v for k, v in per.items()}))
        flag = ""
        if acc > best[0]:
            best = (acc, epoch)
            torch.save(model.state_dict(), CKPT / f"{name}.pt")
            flag = " *"
        print(f"  {name} epoch {epoch:2d}  loss {total / len(X):.4f}  val exact {100 * acc:5.2f}% "
              f"(E {100 * per['einsiedeln']:5.2f}, M {100 * per['ms73']:5.2f})  {time.time() - t0:4.1f}s{flag}", flush=True)
    return dict(best_epoch=best[1], best_val_exact=best[0], history=history)


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
        tr, va = load(MANUSCRIPTS, "train", crops_dir=crops), load(MANUSCRIPTS, "val", crops_dir=crops)
        counts = {s: {ms: int(np.sum(d["manuscript"] == ms)) for ms in MANUSCRIPTS} for s, d in (("train", tr), ("val", va))}
        print(f"training {name} on {Path(crops).name} (device {dev}; single-note crops {counts})", flush=True)
        meta[name] = train_one(name, tr, va, dev)
        meta[name].update(seed=SEED, epochs=EPOCHS, batch=BATCH, lr=LR, weight_decay=WD, n_train=counts["train"],
                          crops=Path(crops).name, out_h=CONFIGS[name]["out_h"], pool=CONFIGS[name]["pool"],
                          shift_jitter=CONFIGS[name]["shift"], mask=CONFIGS[name]["mask"])
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"  {name}: best val exact {100 * meta[name]['best_val_exact']:.2f}% at epoch {meta[name]['best_epoch']}", flush=True)


if __name__ == "__main__":
    main()
