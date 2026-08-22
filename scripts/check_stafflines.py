"""
Sanity-check a stafflines.json for lines that cross each other or sit
suspiciously close together -- the kind of leftover duplicate/near-duplicate
detections that confuse staff_regroup's line-gap estimate and grouping (see
fix_stafflines.py, built for exactly this problem).

For every pair of lines whose x-ranges overlap, compares their y at each
shared x (using each line's own per-x y_values, not assuming they're
straight) and flags:
  - crossing: the sign of (y_i - y_j) changes somewhere in the overlap,
    i.e. the two lines physically cross.
  - too close: the average vertical gap over the overlap is under
    CLOSE_FRACTION of the page's own median scale_unit (its own estimate of
    one diatonic step's pixel spacing) -- a real pair of adjacent staff
    lines is never this close.

Usage:
    python check_stafflines.py ../CantusMA1537_p22
    python check_stafflines.py --staff-json path/to/foo_stafflines.json
"""
import argparse
import json
import statistics
from pathlib import Path

import page_inputs as pi

CLOSE_FRACTION = 0.5  # flag pairs closer than this fraction of the page's median scale_unit


def find_staff_json(page: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    input_dir = page / pi.INPUT_DIR_NAME
    if not input_dir.is_dir():
        input_dir = page
    candidates = pi._find(input_dir, "*stafflines*.json")
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one stafflines file in {input_dir}, found {len(candidates)}: "
                         f"{candidates}. Use --staff-json to disambiguate.")
    return candidates[0]


def y_at(entry, x):
    cp = entry["centerline_page"]
    if x < cp["x_start"] or x > cp["x_end"]:
        return None
    idx = round(x - cp["x_start"])
    idx = max(0, min(idx, len(cp["y_values"]) - 1))
    return cp["y_values"][idx]


def compare_pair(a, b):
    """None if the two lines' x-ranges don't overlap, else (avg_dist, min_dist, crosses)."""
    ca, cb = a["centerline_page"], b["centerline_page"]
    x_lo = max(ca["x_start"], cb["x_start"])
    x_hi = min(ca["x_end"], cb["x_end"])
    if x_hi < x_lo:
        return None
    diffs = []
    step = max(1, (x_hi - x_lo) // 50)  # cap sampling for very wide overlaps
    for x in range(x_lo, x_hi + 1, step):
        ya, yb = y_at(a, x), y_at(b, x)
        if ya is not None and yb is not None:
            diffs.append(ya - yb)
    if not diffs:
        return None
    avg_dist = sum(abs(d) for d in diffs) / len(diffs)
    min_dist = min(abs(d) for d in diffs)
    crosses = min(diffs) <= 0 <= max(diffs) and max(diffs) - min(diffs) > 1e-6
    return avg_dist, min_dist, crosses


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("page", type=Path, nargs="?", default=None)
    parser.add_argument("--staff-json", type=Path, default=None)
    args = parser.parse_args()
    if args.page is None and args.staff_json is None:
        parser.error("pass a page folder or --staff-json")

    staff_json = find_staff_json(args.page, args.staff_json) if args.page else args.staff_json
    entries = json.loads(staff_json.read_text())
    entries = [e for e in entries if e.get("centerline_page", {}).get("y_values")]
    print(f"{staff_json}: {len(entries)} lines")

    scale_units = [e["scale_unit"] for e in entries if e.get("scale_unit")]
    median_scale = statistics.median(scale_units) if scale_units else 20.0
    close_threshold = CLOSE_FRACTION * median_scale
    print(f"median scale_unit: {median_scale:.1f}px  ->  flagging pairs closer than {close_threshold:.1f}px\n")

    crossing_pairs, close_pairs = [], []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            result = compare_pair(entries[i], entries[j])
            if result is None:
                continue
            avg_dist, min_dist, crosses = result
            if crosses:
                crossing_pairs.append((entries[i], entries[j], avg_dist, min_dist))
            elif avg_dist < close_threshold:
                close_pairs.append((entries[i], entries[j], avg_dist, min_dist))

    if crossing_pairs:
        print(f"--- {len(crossing_pairs)} CROSSING pairs (lines physically intersect) ---")
        for a, b, avg_dist, min_dist in sorted(crossing_pairs, key=lambda t: t[3]):
            print(f"  {a['id']:30s} x {b['id']:30s}  avg_gap={avg_dist:5.1f}px  min_gap={min_dist:5.1f}px")
    else:
        print("no crossing pairs")

    print()
    if close_pairs:
        print(f"--- {len(close_pairs)} TOO-CLOSE pairs (avg gap < {close_threshold:.1f}px) ---")
        for a, b, avg_dist, min_dist in sorted(close_pairs, key=lambda t: t[2]):
            print(f"  {a['id']:30s} x {b['id']:30s}  avg_gap={avg_dist:5.1f}px  min_gap={min_dist:5.1f}px")
    else:
        print("no suspiciously-close pairs")

    flagged = crossing_pairs + close_pairs
    if flagged and args.page:
        ids = sorted({a["id"] for a, b, *_ in flagged} | {b["id"] for a, b, *_ in flagged})
        print(f"\nto see all {len(ids)} flagged lines highlighted in the editor:")
        print(f"  python fix_stafflines.py {args.page} --highlight {','.join(ids)}")


if __name__ == "__main__":
    main()
