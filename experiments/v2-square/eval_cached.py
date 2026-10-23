"""Score the trained checkpoints on the cached crops of a split (MEI staff, no
augmentation): exact / within-one / mean steps off, per manuscript and per
single-note class. No heuristic and no detected staff -- see evaluate.py for those.

    python eval_cached.py [--split test]
"""
import argparse
import csv

import numpy as np
import torch

from data import load, model_input
from train import CKPT, CONFIGS, HERE, MANUSCRIPTS, build_model, device, predict_steps

CLASS_OF_TILT = {"": "punctum", "s": "virga", "n": "reverse virga", "se": "inclinatum"}


def score(pred, truth):
    err = np.abs(pred - truth)
    return dict(n=len(truth), exact=float(np.mean(err == 0)), within1=float(np.mean(err <= 1)), mae=float(np.mean(err)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    dev = device()
    d, preds = None, {}
    for name, c in CONFIGS.items():
        if not (CKPT / f"{name}.pt").exists():
            print(f"WARNING: no checkpoint for {name}, left out")
            continue
        dc = load(MANUSCRIPTS, args.split, crops_dir=c["crops"])
        if d is None:
            d = dc
        assert np.array_equal(dc["id"], d["id"]) and np.array_equal(dc["page"], d["page"]), f"{name}: crops out of order"
        m = build_model(name).to(dev)
        m.load_state_dict(torch.load(CKPT / f"{name}.pt", map_location=dev))
        preds[name] = predict_steps(m, torch.tensor(model_input(dc, c["mask"]), device=dev), c["classifier"], dev,
                                    out_h=c["out_h"])

    groups = [("all", np.ones(len(d["step"]), bool))]
    groups += [(ms, d["manuscript"] == ms) for ms in MANUSCRIPTS]
    groups += [(f"{ms} / {CLASS_OF_TILT[t]}", (d["manuscript"] == ms) & (d["tilt"] == t))
               for ms in MANUSCRIPTS for t in CLASS_OF_TILT]
    rows = []
    print(f"{args.split} split, single-note neumes, MEI staff\n")
    print(f"{'group':28s} {'n':>6s} | " + " | ".join(f"{n:^22s}" for n in preds))
    print(f"{'':28s} {'':>6s} | " + " | ".join(f"{'exact':>7s} {'±1':>6s} {'MAE':>6s}" for _ in preds))
    for g, mask in groups:
        if mask.sum() < 20:
            continue
        cells = []
        for name in preds:
            s = score(preds[name][mask], d["step"][mask])
            rows.append(dict(split=args.split, group=g, model=name, **s))
            cells.append(f"{100 * s['exact']:6.2f}% {100 * s['within1']:5.1f}% {s['mae']:6.3f}")
        print(f"{g:28s} {int(mask.sum()):6d} | " + " | ".join(cells))
    out = HERE / "results" / f"{args.split}_cached_scores.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez(HERE / "results" / f"{args.split}_cached_predictions.npz", step=d["step"], manuscript=d["manuscript"],
             page=d["page"], id=d["id"], tilt=d["tilt"], **{f"pred_{k}": v for k, v in preds.items()})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
