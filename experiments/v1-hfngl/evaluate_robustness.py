"""Does staff-error-augmented training actually make the CNN more robust to
bad staff-finding? Compares stepcnn_real.pt vs stepcnn_real_aug.pt on:

  (a) the normal held-out test set (crops from corrected/best-available staff)
  (b) the same test glyphs, but re-cropped using each of the 5 hand-corrected
      pages' TRUE pre-correction staff geometry (data.uncorrected_crop_for_row)
      -- real bad-staff-finding crops, not synthetic jitter, so this measures
      generalization to an actual failure mode rather than to the augmentation
      recipe itself.

Heuristic's own corrected/uncorrected numbers are printed alongside for
context (same as evaluate.py's experiments 1/2, restricted to subset (b)'s
rows).

    python evaluate_robustness.py
"""
import numpy as np
import torch

from model import StepCNN
from data import REAL_LABELED_PAGES, MANUAL_STAFF_PAGES, load_real_labeled_page, load_shapes, uncorrected_crop_for_row
from split import build_split
from train import CKPT_DIR


def summarize(errs):
    errs = np.asarray(errs, dtype=np.float64)
    if len(errs) == 0:
        return {"n": 0, "mae": float("nan"), "exact": float("nan"), "within1": float("nan")}
    return {"n": len(errs), "mae": float(errs.mean()),
            "exact": float((errs == 0).mean()), "within1": float((errs <= 1).mean())}


def load_model(name):
    path = CKPT_DIR / name
    model = StepCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def report(title, errs, n):
    s = summarize(errs)
    print(f"  {title:38s} {n:5d} {100*s['exact']:6.1f}% {100*s['within1']:8.1f}% {s['mae']:6.3f}")


def main():
    shapes = load_shapes()
    print("loading real-labeled pages...")
    all_rows = []
    for page in REAL_LABELED_PAGES:
        all_rows += load_real_labeled_page(page, shapes)

    print("building split (must match train.py's)...")
    _, test_rows = build_split(all_rows, log=lambda *a: None)
    robust_rows = [r for r in test_rows if r["page"] in MANUAL_STAFF_PAGES]
    print(f"test={len(test_rows)}  robustness subset (5 corrected pages)={len(robust_rows)}")

    model_real = load_model("stepcnn_real.pt")
    model_aug = load_model("stepcnn_real_aug.pt")

    # (a) normal test set, both models
    y_test = np.array([r["truth"] for r in test_rows], dtype=np.float32)
    X_test = np.stack([r["crop"] for r in test_rows])[:, None, :, :]
    with torch.no_grad():
        pred_real_a = model_real(torch.tensor(X_test)).numpy()
        pred_aug_a = model_aug(torch.tensor(X_test)).numpy()
    errs_real_a = np.abs(np.round(pred_real_a) - y_test)
    errs_aug_a = np.abs(np.round(pred_aug_a) - y_test)

    # (b) robustness subset, re-cropped under TRUE uncorrected staff geometry
    image_cache = {}
    y_rob, X_rob_list, heur_c_rob, heur_u_rob = [], [], [], []
    for r in robust_rows:
        crop = uncorrected_crop_for_row(r, image_cache)
        if crop is None:
            continue
        X_rob_list.append(crop)
        y_rob.append(r["truth"])
        heur_c_rob.append(r["heur_corrected"])
        heur_u_rob.append(r["heur_uncorrected"])
    y_rob = np.array(y_rob, dtype=np.float32)
    X_rob = np.stack(X_rob_list)[:, None, :, :]
    heur_c_rob = np.array(heur_c_rob, dtype=np.float32)
    heur_u_rob = np.array(heur_u_rob, dtype=np.float32)
    print(f"robustness subset with valid uncorrected-geometry crops: {len(y_rob)}/{len(robust_rows)}")

    with torch.no_grad():
        pred_real_b = model_real(torch.tensor(X_rob)).numpy()
        pred_aug_b = model_aug(torch.tensor(X_rob)).numpy()
    errs_real_b = np.abs(np.round(pred_real_b) - y_rob)
    errs_aug_b = np.abs(np.round(pred_aug_b) - y_rob)

    valid_c = ~np.isnan(heur_c_rob)
    valid_u = ~np.isnan(heur_u_rob)
    errs_heur_c = np.abs(heur_c_rob[valid_c] - y_rob[valid_c])
    errs_heur_u = np.abs(heur_u_rob[valid_u] - y_rob[valid_u])

    print(f"\n{'':38s} {'n':>5s} {'exact%':>7s} {'within1%':>9s} {'MAE':>6s}")
    print("-- (a) full test set, corrected/best-available staff --")
    report("CNN, no augmentation", errs_real_a, len(errs_real_a))
    report("CNN, staff-error augmentation", errs_aug_a, len(errs_aug_a))
    print("-- (b) robustness subset, TRUE uncorrected staff geometry --")
    report("heuristic, corrected staff (reference)", errs_heur_c, valid_c.sum())
    report("heuristic, uncorrected staff (reference)", errs_heur_u, valid_u.sum())
    report("CNN, no augmentation", errs_real_b, len(errs_real_b))
    report("CNN, staff-error augmentation", errs_aug_b, len(errs_aug_b))


if __name__ == "__main__":
    main()
