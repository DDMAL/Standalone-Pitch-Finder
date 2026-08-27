"""10-way comparison on one shared 247-glyph test split: {heuristic,
CNN-regression no-aug, CNN-regression aug, CNN-classifier no-aug,
CNN-classifier aug} x {corrected staff, uncorrected staff}. See
../README.md for what each means. For the 8 pages that never needed staff
correction, "uncorrected" is identical to "corrected" (there was nothing to
fix), so those rows are the same in both columns -- only the 5
hand-corrected pages' rows actually differ, same convention the heuristic
columns already used.

For the CNN columns specifically, "uncorrected staff" means the finalized
HYBRID crop from data.uncorrected_crop_for_row(): the true pre-correction
crop by default, overridden with a bbox-height-based crop only when its
height looks anomalous (see data.py's HYBRID_* constants for why a pure
"always trust the bad staff" or "never trust it" rule both measured worse).
The heuristic's own "uncorrected staff" column is unaffected by this --
it never used expanded-box crops in the first place.

Loads checkpoints from train.py if present (training reproduces the same
split from the same seed, so a checkpoint trained separately is still
evaluated on the right held-out set); trains fresh otherwise.

get_test_predictions() is the reusable part -- plot_by_class.py imports it
too, so both work from the exact same test rows/predictions.

    python evaluate.py
"""
import numpy as np
import torch

from model import StepCNN, StepCNNClassifier, STEP_MIN
from data import REAL_LABELED_PAGES, MANUAL_STAFF_PAGES, load_real_labeled_page, load_shapes, uncorrected_crop_for_row
from split import build_split
from augment import build_augmented_rows
from train import train_cnn, train_classifier, CKPT_DIR, N_AUG, SEED

METHODS = [
    "heuristic (corrected staff)", "heuristic (uncorrected staff)",
    "CNN regression no-aug (corrected staff)", "CNN regression no-aug (uncorrected staff)",
    "CNN regression aug (corrected staff)", "CNN regression aug (uncorrected staff)",
    "CNN classifier no-aug (corrected staff)", "CNN classifier no-aug (uncorrected staff)",
    "CNN classifier aug (corrected staff)", "CNN classifier aug (uncorrected staff)",
]


def summarize(errs):
    errs = np.asarray(errs, dtype=np.float64)
    if len(errs) == 0:
        return {"n": 0, "mae": float("nan"), "exact": float("nan"), "within1": float("nan")}
    return {"n": len(errs), "mae": float(errs.mean()),
            "exact": float((errs == 0).mean()), "within1": float((errs <= 1).mean())}


def load_model(ckpt_name, cls=StepCNN):
    model = cls()
    model.load_state_dict(torch.load(CKPT_DIR / ckpt_name, map_location="cpu"))
    model.eval()
    return model


def _augmented_train_set(X_train, y_train, aug_rows):
    X_aug = np.stack([r["crop"] for r in aug_rows])[:, None, :, :]
    y_aug = np.array([r["truth"] for r in aug_rows], dtype=np.float32)
    return np.concatenate([X_train, X_aug], axis=0), np.concatenate([y_train, y_aug], axis=0)


def load_or_train_cnn(ckpt_name, X_train, y_train, aug_rows=None):
    if (CKPT_DIR / ckpt_name).exists():
        print(f"  loaded {ckpt_name}")
        return load_model(ckpt_name)
    print(f"  no {ckpt_name}, training fresh...")
    if aug_rows is not None:
        X_train, y_train = _augmented_train_set(X_train, y_train, aug_rows)
    model = train_cnn(X_train, y_train)
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CKPT_DIR / ckpt_name)
    return model


def load_or_train_classifier(ckpt_name, X_train, y_train, aug_rows=None):
    if (CKPT_DIR / ckpt_name).exists():
        print(f"  loaded {ckpt_name}")
        return load_model(ckpt_name, cls=StepCNNClassifier)
    print(f"  no {ckpt_name}, training fresh...")
    if aug_rows is not None:
        X_train, y_train = _augmented_train_set(X_train, y_train, aug_rows)
    model = train_classifier(X_train, y_train)
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CKPT_DIR / ckpt_name)
    return model


def uncorrected_crops(test_rows):
    """(crops, valid, tops, bottoms, overridden) -- per-row crop under the
    TRUE uncorrected staff geometry, plus the page-pixel (top, bottom)
    bounds that crop was cut from (needed to overlay staff lines on it
    later), plus whether the hybrid rule replaced that crop with a
    bbox-centered one (see data.uncorrected_crop_for_row). Falls back to
    the row's own (corrected-geometry) crop/bounds for the 8 pages that
    never needed correction, since that IS what "uncorrected" means there
    (valid=True, overridden=False). For one of the 5 corrected pages,
    valid=False marks a glyph with no box at all under the uncorrected
    geometry -- a placeholder crop is still emitted (never used once
    masked to NaN) so the array stays rectangular."""
    image_cache = {}
    crops, valid = [], np.ones(len(test_rows), dtype=bool)
    tops, bottoms, overridden = [], [], np.zeros(len(test_rows), dtype=bool)
    for i, r in enumerate(test_rows):
        if r["page"] not in MANUAL_STAFF_PAGES:
            crops.append(r["crop"])
            tops.append(r["box_top"]); bottoms.append(r["box_bottom"])
            continue
        crop, top, bottom, over = uncorrected_crop_for_row(r, image_cache)
        if crop is None:
            crops.append(r["crop"])  # placeholder; masked out via `valid`
            tops.append(r["box_top"]); bottoms.append(r["box_bottom"])
            valid[i] = False
        else:
            crops.append(crop)
            tops.append(top); bottoms.append(bottom)
            overridden[i] = over
    return np.stack(crops)[:, None, :, :], valid, np.array(tops), np.array(bottoms), overridden


def get_test_predictions():
    """(test_rows, y_test, preds) -- preds is {method_name: array aligned
    1:1 with test_rows}, NaN where that method has no guess (heuristic with
    no pixel-anchor coverage, or a glyph with no coverage under the
    uncorrected geometry). CNN predictions are already rounded."""
    shapes = load_shapes()
    print("loading real-labeled pages...")
    all_rows = []
    for page in REAL_LABELED_PAGES:
        all_rows += load_real_labeled_page(page, shapes)

    print("building split (must match train.py's)...")
    train_rows, test_rows = build_split(all_rows)
    print(f"train={len(train_rows)} test={len(test_rows)} "
          f"pages={len({r['page'] for r in test_rows})} classes={len({r['class_name'] for r in test_rows})}")

    y_test = np.array([r["truth"] for r in test_rows], dtype=np.float32)
    X_test = np.stack([r["crop"] for r in test_rows])[:, None, :, :]
    X_train = np.stack([r["crop"] for r in train_rows])[:, None, :, :]
    y_train = np.array([r["truth"] for r in train_rows], dtype=np.float32)

    heur_c = np.array([r["heur_corrected"] for r in test_rows], dtype=np.float32)
    heur_u = np.array([r["heur_uncorrected"] for r in test_rows], dtype=np.float32)

    print("building uncorrected-staff crops for the CNN columns...")
    X_test_u, valid_u_crop, _, _, _ = uncorrected_crops(test_rows)

    print("loading/training CNN checkpoints...")
    aug_rows = build_augmented_rows(train_rows, N_AUG, seed=SEED)
    model_real = load_or_train_cnn("stepcnn_real.pt", X_train, y_train)
    model_aug = load_or_train_cnn("stepcnn_real_aug.pt", X_train, y_train, aug_rows=aug_rows)
    model_cls = load_or_train_classifier("stepcnn_cls.pt", X_train, y_train)
    model_cls_aug = load_or_train_classifier("stepcnn_cls_aug.pt", X_train, y_train, aug_rows=aug_rows)

    def classify(model, X):
        logits = model(torch.tensor(X))
        return (torch.argmax(logits, dim=1).numpy() + STEP_MIN).astype(np.float32)

    with torch.no_grad():
        pred_real_c = np.round(model_real(torch.tensor(X_test)).numpy())
        pred_real_u = np.round(model_real(torch.tensor(X_test_u)).numpy())
        pred_aug_c = np.round(model_aug(torch.tensor(X_test)).numpy())
        pred_aug_u = np.round(model_aug(torch.tensor(X_test_u)).numpy())
        pred_cls_c = classify(model_cls, X_test)
        pred_cls_u = classify(model_cls, X_test_u)
        pred_cls_aug_c = classify(model_cls_aug, X_test)
        pred_cls_aug_u = classify(model_cls_aug, X_test_u)
    for pred_u in [pred_real_u, pred_aug_u, pred_cls_u, pred_cls_aug_u]:
        pred_u[~valid_u_crop] = np.nan

    preds = dict(zip(METHODS, [
        heur_c, heur_u, pred_real_c, pred_real_u, pred_aug_c, pred_aug_u,
        pred_cls_c, pred_cls_u, pred_cls_aug_c, pred_cls_aug_u,
    ]))
    return test_rows, y_test, preds


def main():
    test_rows, y_test, preds = get_test_predictions()
    print(f"\n{'experiment':32s} {'n':>5s} {'exact%':>7s} {'within1%':>9s} {'MAE':>6s}")
    for name, pred in preds.items():
        valid = ~np.isnan(pred)
        errs = np.abs(pred[valid] - y_test[valid])
        s = summarize(errs)
        print(f"  {name:32s} {valid.sum():5d} {100*s['exact']:6.1f}% {100*s['within1']:8.1f}% {s['mae']:6.3f}")


if __name__ == "__main__":
    main()
