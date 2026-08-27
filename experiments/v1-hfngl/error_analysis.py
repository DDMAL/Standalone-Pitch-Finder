"""Error analysis: for test glyphs where a method is correct under
corrected staff but wrong under uncorrected staff, characterize (a) how the
crop geometry itself changed (height ratio, center shift) and (b) what
actually went wrong with the staff-line detection in that specific crop
window (line count, spacing) -- e.g. did detection produce extra/duplicate
lines, too few lines, or roughly the right lines just shifted. Uses
data.py's finalized hybrid crop for "uncorrected staff" (see its HYBRID_*
constants), so this now shows the failures that *survive* that fix -- a
reasonable starting point for the next round of improvements.

    python error_analysis.py
"""
from pathlib import Path

import numpy as np

from evaluate import get_test_predictions, uncorrected_crops, METHODS
from data import MANUAL_STAFF_PAGES, staff_line_crop_rows

HERE = Path(__file__).resolve().parent
FOCUS_METHOD = "CNN classifier aug"  # the model this analysis focuses on in detail


def line_spacing(rows):
    """Median gap between consecutive line rows, or nan if <2 lines."""
    rows = sorted(rows)
    if len(rows) < 2:
        return float("nan")
    return float(np.median(np.diff(rows)))


def main():
    test_rows, y_test, preds = get_test_predictions()
    X_test_u, valid_u, tops_u, bottoms_u, overridden_u = uncorrected_crops(test_rows)

    on_manual = np.array([r["page"] in MANUAL_STAFF_PAGES for r in test_rows])

    print(f"{'method':38s} {'n eligible':>10s} {'correct->wrong':>15s}")
    method_failure_sets = {}
    for base in ["heuristic", "CNN regression no-aug", "CNN regression aug",
                 "CNN classifier no-aug", "CNN classifier aug"]:
        pc, pu = preds[f"{base} (corrected staff)"], preds[f"{base} (uncorrected staff)"]
        eligible = on_manual & ~np.isnan(pu)
        failure = eligible & (pc == y_test) & (pu != y_test)
        method_failure_sets[base] = np.where(failure)[0]
        print(f"{base:38s} {eligible.sum():10d} {failure.sum():15d}")

    idx = method_failure_sets[FOCUS_METHOD]
    print(f"\n{'='*70}\ndetailed analysis for '{FOCUS_METHOD}': {len(idx)} correct->wrong cases\n{'='*70}")

    height_ratios, shifts, n_lines_c, n_lines_u, spacing_c, spacing_u = [], [], [], [], [], []
    print(f"\n{'page':22s} {'class':16s} {'truth':>5s} {'h.ratio':>7s} {'shift':>6s} "
          f"{'lines_c':>7s} {'lines_u':>7s} {'gap_c':>6s} {'gap_u':>6s} {'hybrid':>6s}")
    for i in idx:
        r = test_rows[i]
        top_c, bottom_c = r["box_top"], r["box_bottom"]
        top_u, bottom_u = tops_u[i], bottoms_u[i]
        height_c, height_u = bottom_c - top_c, bottom_u - top_u

        ratio = height_u / height_c
        shift = ((top_u + bottom_u) / 2 - (top_c + bottom_c) / 2) / height_c

        rows_c = staff_line_crop_rows(r["page"], True, r["ulx"], r["ncols"], top_c, bottom_c)
        rows_u = staff_line_crop_rows(r["page"], False, r["ulx"], r["ncols"], top_u, bottom_u)
        gap_c, gap_u = line_spacing(rows_c), line_spacing(rows_u)

        height_ratios.append(ratio); shifts.append(shift)
        n_lines_c.append(len(rows_c)); n_lines_u.append(len(rows_u))
        spacing_c.append(gap_c); spacing_u.append(gap_u)

        print(f"{r['page']:22s} {r['class_name']:16s} {y_test[i]:5.0f} {ratio:7.2f} {shift:6.2f} "
              f"{len(rows_c):7d} {len(rows_u):7d} {gap_c:6.1f} {gap_u:6.1f} {str(overridden_u[i]):>6s}")

    print(f"\nof these {len(idx)} surviving failures, the hybrid rule overrode "
          f"the crop for {overridden_u[idx].sum()} -- i.e. it still fell "
          f"through to the wrong answer even after intervening, vs. "
          f"{(~overridden_u[idx]).sum()} it never flagged as anomalous.")

    height_ratios, shifts = np.array(height_ratios), np.array(shifts)
    n_lines_c, n_lines_u = np.array(n_lines_c), np.array(n_lines_u)

    print(f"\n{'-'*70}\nsummary across {len(idx)} cases\n{'-'*70}")
    print(f"crop height ratio (uncorrected/corrected): mean={height_ratios.mean():.2f} "
          f"median={np.median(height_ratios):.2f} (1.0 = same size)")
    print(f"crop center shift (fraction of corrected height): mean={shifts.mean():+.2f} "
          f"median={np.median(shifts):+.2f} (0 = same center)")
    print(f"lines visible in crop, corrected: mean={n_lines_c.mean():.1f}  "
          f"uncorrected: mean={n_lines_u.mean():.1f}")
    more_lines = (n_lines_u > n_lines_c).sum()
    fewer_lines = (n_lines_u < n_lines_c).sum()
    same_lines = (n_lines_u == n_lines_c).sum()
    print(f"of these: uncorrected shows MORE lines in {more_lines}, FEWER in {fewer_lines}, "
          f"SAME count in {same_lines}")

    # crude bucketing of the dominant error mode per case
    print(f"\n{'-'*70}\nerror-mode breakdown (heuristic classification, not mutually exclusive)\n{'-'*70}")
    extreme_scale = np.sum((height_ratios < 0.6) | (height_ratios > 1.6))
    extreme_shift = np.sum(np.abs(shifts) > 0.5)
    line_count_mismatch = np.sum(n_lines_c != n_lines_u)
    print(f"extreme scale change (ratio <0.6 or >1.6): {extreme_scale}/{len(idx)}")
    print(f"extreme shift (>0.5 crop-heights off-center): {extreme_shift}/{len(idx)}")
    print(f"different number of visible lines than corrected: {line_count_mismatch}/{len(idx)}")


if __name__ == "__main__":
    main()
