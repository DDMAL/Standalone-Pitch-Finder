"""4-way comparison on one shared test split -- see ../README.md for what
each experiment means. Loads checkpoints from train.py if present
(training reproduces the same split from the same seed, so a checkpoint
trained separately is still evaluated on the right held-out set); trains
fresh otherwise.

    python evaluate.py
"""
from pathlib import Path

import numpy as np
import torch

from model import StepCNN
from data import REAL_LABELED_PAGES, load_real_labeled_page, load_shapes
from split import build_split
from train import train_cnn, CKPT_DIR

HERE = Path(__file__).resolve().parent


def summarize(errs):
    errs = np.asarray(errs, dtype=np.float64)
    if len(errs) == 0:
        return {"n": 0, "mae": float("nan"), "exact": float("nan"), "within1": float("nan")}
    return {"n": len(errs), "mae": float(errs.mean()),
            "exact": float((errs == 0).mean()), "within1": float((errs <= 1).mean())}


def load_or_train(ckpt_name, X_train, y_train):
    path = CKPT_DIR / ckpt_name
    model = StepCNN()
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"  loaded {path}")
    else:
        print(f"  no checkpoint at {path}, training fresh...")
        model = train_cnn(X_train, y_train)
        CKPT_DIR.mkdir(exist_ok=True)
        torch.save(model.state_dict(), path)
    model.eval()
    return model


def main():
    shapes = load_shapes()
    print("loading real-labeled pages...")
    all_rows = []
    for page in REAL_LABELED_PAGES:
        rows = load_real_labeled_page(page, shapes)
        all_rows += rows

    print("building split (must match train.py's)...")
    train_rows, test_rows = build_split(all_rows)
    print(f"train={len(train_rows)} test={len(test_rows)} "
          f"pages={len({r['page'] for r in test_rows})} classes={len({r['class_name'] for r in test_rows})}")

    y_test = np.array([r["truth"] for r in test_rows], dtype=np.float32)
    cls_test = np.array([r["class_name"] for r in test_rows])
    X_test = np.stack([r["crop"] for r in test_rows])[:, None, :, :]
    X_train = np.stack([r["crop"] for r in train_rows])[:, None, :, :]
    y_train = np.array([r["truth"] for r in train_rows], dtype=np.float32)

    heur_c = np.array([r["heur_corrected"] for r in test_rows], dtype=np.float32)
    heur_u = np.array([r["heur_uncorrected"] for r in test_rows], dtype=np.float32)
    valid_c, valid_u = ~np.isnan(heur_c), ~np.isnan(heur_u)
    errs_c = np.abs(heur_c[valid_c] - y_test[valid_c])
    errs_u = np.abs(heur_u[valid_u] - y_test[valid_u])

    print("\nloading/training CNN checkpoints...")
    model_real = load_or_train("stepcnn_real.pt", X_train, y_train)
    model_pseudo = load_or_train("stepcnn_real_pseudo.pt", X_train, y_train)  # trains real-only if missing

    with torch.no_grad():
        pred3 = model_real(torch.tensor(X_test)).numpy()
        pred4 = model_pseudo(torch.tensor(X_test)).numpy()
    errs3 = np.abs(np.round(pred3) - y_test)
    errs4 = np.abs(np.round(pred4) - y_test)

    print(f"\n{'experiment':38s} {'n':>5s} {'exact%':>7s} {'within1%':>9s} {'MAE':>6s}")
    for name, errs, n in [
        ("(1) heuristic, corrected staff", errs_c, valid_c.sum()),
        ("(2) heuristic, uncorrected staff", errs_u, valid_u.sum()),
        ("(3) CNN, real labels only", errs3, len(errs3)),
        ("(4) CNN, real + pseudo labels", errs4, len(errs4)),
    ]:
        s = summarize(errs)
        print(f"  {name:38s} {n:5d} {100*s['exact']:6.1f}% {100*s['within1']:8.1f}% {s['mae']:6.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, pred, title in [(axes[0], pred3, "real only"), (axes[1], pred4, "real + pseudo")]:
        ax.scatter(y_test, np.round(pred), alpha=0.6, color="red")
        ax.plot([-1, 8], [-1, 8], "k--", linewidth=1)
        ax.set_xlabel("annotated step (truth)"); ax.set_ylabel("model predicted step (rounded)")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(HERE / "scatter.png", dpi=130)
    print(f"\nsaved scatter to {HERE / 'scatter.png'}")


if __name__ == "__main__":
    main()
