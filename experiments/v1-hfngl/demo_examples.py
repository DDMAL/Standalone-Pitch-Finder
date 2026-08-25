"""Illustrative demo: for a handful of test glyphs, show the actual crop
image fed to the models (corrected-staff version and, for the 5 hand-fixed
pages, the true-uncorrected-staff version of the same glyph) next to every
method's prediction vs. the true label. Meant for a slide/report figure,
not analysis -- picks a spread of examples (robust cases, augmentation
rescues, classifier-beats-regression cases, general failures) rather than
a single repeated story.

    python demo_examples.py
"""
from pathlib import Path

import numpy as np

from evaluate import get_test_predictions, uncorrected_crops
from data import MANUAL_STAFF_PAGES

HERE = Path(__file__).resolve().parent

SHORT_NAMES = {
    "heuristic (corrected staff)": "heuristic",
    "heuristic (uncorrected staff)": "heuristic",
    "CNN regression no-aug (corrected staff)": "reg no-aug",
    "CNN regression no-aug (uncorrected staff)": "reg no-aug",
    "CNN regression aug (corrected staff)": "reg aug",
    "CNN regression aug (uncorrected staff)": "reg aug",
    "CNN classifier no-aug (corrected staff)": "cls no-aug",
    "CNN classifier no-aug (uncorrected staff)": "cls no-aug",
    "CNN classifier aug (corrected staff)": "cls aug",
    "CNN classifier aug (uncorrected staff)": "cls aug",
}
CORRECTED_METHODS = [m for m in SHORT_NAMES if "uncorrected" not in m]
UNCORRECTED_METHODS = [m for m in SHORT_NAMES if "uncorrected" in m]

PER_CATEGORY = 2


def pick_examples(test_rows, y_test, preds):
    """A mixed set of examples, not just total-failure cases: some where
    everything is robust across both conditions, some where augmentation
    specifically rescues a prediction, some where the classifier succeeds
    and regression doesn't, and some general disagreement/failure cases --
    each category sampled separately so the figure shows a spread of
    stories rather than one repeated pattern."""
    rng = np.random.default_rng(0)
    on_manual_page = np.array([r["page"] in MANUAL_STAFF_PAGES for r in test_rows])

    def ok(method, i):
        p = preds[method][i]
        return not np.isnan(p) and p == y_test[i]

    def sample(pool, exclude):
        pool = [i for i in pool if i not in exclude]
        if len(pool) > PER_CATEGORY:
            pool = list(rng.choice(pool, size=PER_CATEGORY, replace=False))
        return pool

    chosen = []

    # A: robust across both conditions (both corrected and uncorrected correct)
    pool = [i for i in np.where(on_manual_page)[0]
            if ok("CNN classifier aug (corrected staff)", i) and ok("CNN classifier aug (uncorrected staff)", i)]
    chosen += sample(pool, set(chosen))

    # B: augmentation rescues it under uncorrected staff, no-aug doesn't
    pool = [i for i in np.where(on_manual_page)[0]
            if ok("CNN regression aug (uncorrected staff)", i) and not ok("CNN regression no-aug (uncorrected staff)", i)]
    chosen += sample(pool, set(chosen))

    # C: classifier succeeds where regression fails (either condition)
    pool = [i for i in np.where(on_manual_page)[0]
            if (ok("CNN classifier aug (corrected staff)", i) and not ok("CNN regression aug (corrected staff)", i))
            or (ok("CNN classifier aug (uncorrected staff)", i) and not ok("CNN regression aug (uncorrected staff)", i))]
    chosen += sample(pool, set(chosen))

    # D: general disagreement between corrected and uncorrected (may include total-failure cases)
    reg_aug_c, reg_aug_u = preds["CNN regression aug (corrected staff)"], preds["CNN regression aug (uncorrected staff)"]
    pool = [i for i in np.where(on_manual_page)[0] if not np.isnan(reg_aug_u[i]) and reg_aug_c[i] != reg_aug_u[i]]
    chosen += sample(pool, set(chosen))

    return chosen


def main():
    test_rows, y_test, preds = get_test_predictions()
    X_test_u, valid_u = uncorrected_crops(test_rows)

    idx = pick_examples(test_rows, y_test, preds)
    print(f"selected {len(idx)} example glyphs: {[test_rows[i]['page'] for i in idx]}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(idx), 2, figsize=(6, 4.2 * len(idx)))
    if len(idx) == 1:
        axes = axes[None, :]

    for row, i in enumerate(idx):
        r = test_rows[i]
        truth = y_test[i]

        for col, (crop, methods, cond_label) in enumerate([
            (r["crop"], CORRECTED_METHODS, "corrected staff"),
            (X_test_u[i, 0], UNCORRECTED_METHODS, "uncorrected staff"),
        ]):
            ax = axes[row, col]
            ax.imshow(crop, cmap="gray", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            lines = [f"truth: {truth:.0f}"]
            for m in methods:
                p = preds[m][i]
                mark = "n/a" if np.isnan(p) else f"{p:.0f}" + (" ok" if p == truth else " X")
                lines.append(f"{SHORT_NAMES[m]}: {mark}")
            ax.set_title(f"{cond_label}\n{r['page']}, {r['class_name']}", fontsize=8)
            ax.text(1.05, 0.5, "\n".join(lines), transform=ax.transAxes,
                    fontsize=8, va="center", family="monospace")

    fig.tight_layout()
    out_path = HERE / "demo_examples.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
