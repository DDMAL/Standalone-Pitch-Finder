"""Grouped bar chart: exact-match % per neume class, one bar group per
method (the same 6 from evaluate.py: {heuristic, CNN no-aug, CNN aug} x
{corrected, uncorrected staff}), on the same held-out test set. Classes are
sorted by test-set count (annotated in the x-axis label) so a class with
1-2 examples doesn't read the same as one with 100+.

    python plot_by_class.py
"""
from pathlib import Path

import numpy as np

from evaluate import get_test_predictions, METHODS

HERE = Path(__file__).resolve().parent
MIN_N = 3  # classes with fewer test examples than this are dropped -- a
           # single example's 0%/100% swing isn't a comparison worth plotting


def main():
    test_rows, y_test, preds = get_test_predictions()
    cls_test = np.array([r["class_name"] for r in test_rows])

    counts = {c: int((cls_test == c).sum()) for c in set(cls_test)}
    classes = [c for c, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n >= MIN_N]
    dropped = [c for c, n in counts.items() if n < MIN_N]
    if dropped:
        print(f"dropping {len(dropped)} classes with < {MIN_N} test examples: {dropped}")

    exact_pct = np.full((len(METHODS), len(classes)), np.nan)
    for i, method in enumerate(METHODS):
        pred = preds[method]
        for j, cls in enumerate(classes):
            mask = (cls_test == cls) & ~np.isnan(pred)
            if mask.sum() == 0:
                continue
            exact_pct[i, j] = 100 * (pred[mask] == y_test[mask]).mean()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(11, len(classes) * 1.1), 6))
    x = np.arange(len(classes))
    width = 0.8 / len(METHODS)
    # 3 method families (heuristic / CNN no-aug / CNN aug), dark=corrected
    # staff, light=uncorrected -- matches METHODS' corrected/uncorrected pairing
    colors = ["#2F5B8A", "#9DB8D6", "#B5651D", "#E8B98A", "#8B1E1E", "#E29A9A"]
    for i, method in enumerate(METHODS):
        ax.bar(x + (i - (len(METHODS) - 1) / 2) * width, exact_pct[i], width,
               label=method, color=colors[i % len(colors)])

    labels = [f"{c}\n(n={counts[c]})" for c in classes]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("exact-match %")
    ax.set_ylim(0, 105)
    ax.set_title("Exact-match accuracy by neume class and method (held-out test set)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = HERE / "results_by_class.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
