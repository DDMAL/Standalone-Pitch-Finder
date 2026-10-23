"""Cut one crop per usable note in notes.csv and cache them for training:
data/v2_crops/<manuscript>_<split>.npz (--mode staff, the default) or
data/v2_crops_bbox<img-h>/... (--mode bbox).

--mode staff: v1's expanded_boxes rule, with the MEI <sb> staff as the staff geometry:
  vertical    from the top staff line to the bottom one at the note's centre x,
              plus MARGIN_STEPS half-gaps on each side; widened to include the
              note's own box if it sticks out (refined = 1); re-centred on the
              page's median crop height if a degenerate staff makes it shorter
              than MEDIAN_FALLBACK_FRAC of that (refined = 1)
  horizontal  the note's own box (for a single-note neume, the neume's box)

--mode bbox: no staff at all. The crop is centred on the note's box and reaches
BBOX_HALF_UNITS units above and below it, a unit being the page's median note-box
height -- one step to within a few percent on both manuscripts (step / median
box height: Einsiedeln 1.04, MS73 1.02, p5-p95 0.98-1.10). 10 units each way keeps
the whole staff in the crop for any note from step -3 to 9. Same horizontal extent.
--mode icbox: like bbox, but on an IC-style box (ic_like_box: a virga's box runs down
its stem), centred on that box, and with a second image per crop -- the box's rows
as a 255-valued band ("masks") -- for models that take the box as an input channel.
--img-h sets the model window's height (128 = v1's input at half the pixels per
step; 256 = the staff crops' 12.8 px per step).

What is stored is a taller *context* crop: CONTEXT times the crop's height,
same centre, resized to (CTX_H, IMG_W), so the staff-error augmentation
(height x0.8-1.2, shift +-15%) can cut its window from it without reopening
the page. The un-augmented model input is the central IMG_H rows.

    python build_crops.py [--workers 6]
    python build_crops.py --mode bbox --img-h 256 --single-note-only
"""
import argparse
import csv
import collections
import math
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "data"
OUT = DATA / "v2_crops"
IMG_H, IMG_W = 128, 32          # model input, as v1
CONTEXT = 1.5                   # covers v1's jitter: 0.6 (max half-height) + 0.15 (max shift) = 0.75 each side
CTX_H = round(IMG_H * CONTEXT)  # 192
MARGIN_STEPS = 2.0              # v1 annotate_notecenters.MARGIN_STEPS (in half-gaps = steps)
MEDIAN_FALLBACK_FRAC = 0.6      # v1 annotate_notecenters.MEDIAN_FALLBACK_FRAC
BBOX_HALF_UNITS = 10.0          # --mode bbox: crop half-height, in page-median note-box heights
META = ["page", "id", "staff", "neume_id", "neume_size", "nc_index", "ligated", "liquescent", "tilt",
        "source", "box_repaired", "clef_shape", "pname", "oct"]


def staff_span(row, x):
    """(top line y, bottom line y, half gap) of the note's <sb> staff at x."""
    ulx, uly, lrx, lry = (float(row[k]) for k in ("staff_ulx", "staff_uly", "staff_lrx", "staff_lry"))
    t = math.tan(math.radians(float(row["staff_rotate"])))
    H = (lry - uly) - (lrx - ulx) * abs(t)
    x0 = ulx if t >= 0 else lrx
    top = uly + (x - x0) * t
    return top, top + H, H / 6


def crop_bounds(rows):
    """v1's vertical bounds for each note of one page: list of (top, bottom, refined)."""
    local = []
    for r in rows:
        x0, y0, x1, y1 = (float(r[k]) for k in ("box_ulx", "box_uly", "box_lrx", "box_lry"))
        top, bottom, hg = staff_span(r, (x0 + x1) / 2)
        local.append((top - MARGIN_STEPS * hg, bottom + MARGIN_STEPS * hg, y0, y1))
    med = float(np.median([b - t for t, b, *_ in local])) if local else 0.0
    out = []
    for t, b, y0, y1 in local:
        refined = (b - t) < MEDIAN_FALLBACK_FRAC * med
        if refined:
            mid = (t + b) / 2
            t, b = mid - med / 2, mid + med / 2
        if t > y0 or b < y1:
            refined = True
            t, b = min(t, y0), max(b, y1)
        out.append((t, b, int(refined)))
    return out


STEM_MAX_UNITS = 6.0  # longest stem searched for, in page-median note-box heights
STEM_GAP_UNITS = 0.4  # an ink-free run this long ends the stem
STEM_INK_FRAC = 0.25  # stem ink threshold: this far from the paper towards the head's ink


def ic_like_box(gray, r, unit):
    """The note's box as an IC-style glyph box: a virga's box extended down to the end
    of its stem (right side for tilt 's', left for 'n'); other notes keep the MEI box.
    The stem end is the last row, scanning down from the head, where a narrow strip on
    the stem side has at least 2 pixels darker than STEM_INK_FRAC of the way from the
    paper to the head's ink (stems are hairlines, much lighter than heads, in MS73
    especially); a gap of STEM_GAP_UNITS ends the scan. Returns (x0, y0, x1, y1, stem_found)."""
    x0, y0, x1, y1 = (float(r[k]) for k in ("box_ulx", "box_uly", "box_lrx", "box_lry"))
    if r["tilt"] not in ("s", "n"):
        return x0, y0, x1, y1, 0
    H, W = gray.shape
    w = x1 - x0
    sx0, sx1 = (x1 - 0.35 * w, x1 + 0.10 * w) if r["tilt"] == "s" else (x0 - 0.10 * w, x0 + 0.35 * w)
    sx0, sx1 = int(max(0, sx0)), int(min(W, max(sx1, sx0 + 2)))
    ya, yb = int(max(0, y1 - 0.2 * unit)), int(min(H, y1 + STEM_MAX_UNITS * unit))
    head = gray[int(max(0, y0)):int(min(H, y1)), int(max(0, x0)):int(min(W, x1))]
    around = gray[int(max(0, y0 - 2 * unit)):yb, int(max(0, x0 - w)):int(min(W, x1 + w))]
    if head.size == 0 or around.size == 0 or yb <= ya:
        return x0, y0, x1, y1, 0
    paper = np.percentile(around, 85)
    thr = paper - STEM_INK_FRAC * (paper - np.percentile(head, 20))
    ink = (gray[ya:yb, sx0:sx1] < thr).sum(1) >= 2
    end, gap = None, 0
    for k, v in enumerate(ink):
        if v:
            end, gap = ya + k, 0
        else:
            gap += 1
            if gap > STEM_GAP_UNITS * unit:
                break
    if end is None or end < y1 + 0.5 * unit:
        return x0, y0, x1, y1, 0
    return x0, y0, x1, float(end), 1


def bbox_bounds(rows, all_rows):
    """--mode bbox: (top, bottom, 0) centred on each note's box; unit from every note on the page."""
    unit = float(np.median([float(r["box_lry"]) - float(r["box_uly"]) for r in all_rows]))
    out = []
    for r in rows:
        cy = (float(r["box_uly"]) + float(r["box_lry"])) / 2
        out.append((cy - BBOX_HALF_UNITS * unit, cy + BBOX_HALF_UNITS * unit, 0))
    return out


def cut(image, x0, x1, top, bottom, ctx_h=CTX_H):
    """Context crop (ctx_h x IMG_W uint8) around [top, bottom]; outside the page is white."""
    c, h = (top + bottom) / 2, (bottom - top) * CONTEXT
    Y0, Y1 = int(round(c - h / 2)), int(round(c + h / 2))
    X0, X1 = int(round(x0)), max(int(round(x1)), int(round(x0)) + 1)
    H, W = image.shape
    pad = cv2.copyMakeBorder(image[max(0, Y0):min(H, Y1), max(0, X0):min(W, X1)],
                             max(0, -Y0), max(0, Y1 - H), max(0, -X0), max(0, X1 - W),
                             cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(pad, (IMG_W, ctx_h), interpolation=cv2.INTER_AREA)


def page_job(args):
    manuscript, image_name, all_rows, mode, img_h, single_only = args
    image = cv2.imread(str(DATA / "images" / manuscript / image_name), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_name)
    rows = [r for r in all_rows if r["neume_size"] == "1"] if single_only else all_rows
    ctx_h = round(img_h * CONTEXT)
    extra = {}
    if mode == "icbox":
        unit = float(np.median([float(r["box_lry"]) - float(r["box_uly"]) for r in all_rows]))
        boxes = [ic_like_box(image, r, unit) for r in rows]
        bounds, crops, masks = [], [], []
        for x0, y0, x1, y1, _ in boxes:
            cy = (y0 + y1) / 2
            t, b = cy - BBOX_HALF_UNITS * unit, cy + BBOX_HALF_UNITS * unit
            bounds.append((t, b, 0))
            crops.append(cut(image, x0, x1, t, b, ctx_h))
            c0 = cy - CONTEXT * (b - t) / 2  # top of the context crop
            to_row = lambda y: int(round((y - c0) / (CONTEXT * (b - t)) * ctx_h))
            m = np.zeros((ctx_h, IMG_W), np.uint8)
            m[max(0, to_row(y0)):max(0, to_row(y1)) + 1] = 255
            masks.append(m)
        crops = np.stack(crops) if rows else np.zeros((0, ctx_h, IMG_W), np.uint8)
        extra = dict(masks=np.stack(masks) if rows else np.zeros((0, ctx_h, IMG_W), np.uint8),
                     ic_box=np.array([b[:4] for b in boxes], np.float32).reshape(-1, 4),
                     stem_found=np.array([b[4] for b in boxes], np.int8))
        return manuscript, all_rows[0]["split"], crops, rows, bounds, extra
    bounds = crop_bounds(rows) if mode == "staff" else bbox_bounds(rows, all_rows)
    crops = np.stack([cut(image, float(r["box_ulx"]), float(r["box_lrx"]), t, b, ctx_h)
                      for r, (t, b, _) in zip(rows, bounds)]) if rows else np.zeros((0, ctx_h, IMG_W), np.uint8)
    return manuscript, all_rows[0]["split"], crops, rows, bounds, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--mode", choices=("staff", "bbox", "icbox"), default="staff")
    ap.add_argument("--img-h", type=int, default=IMG_H, help="model window height in px")
    ap.add_argument("--single-note-only", action="store_true")
    args = ap.parse_args()
    out_dir = OUT if args.mode == "staff" else DATA / f"v2_crops_{args.mode}{args.img_h}"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_page = collections.defaultdict(list)
    for r in csv.DictReader(open(HERE / "notes.csv")):
        if r["kind"] == "nc" and not r["exclude"]:
            by_page[(r["manuscript"], r["image"])].append(r)
    jobs = [(ms, img, rows, args.mode, args.img_h, args.single_note_only) for (ms, img), rows in sorted(by_page.items())]
    print(f"{sum(len(j[2]) for j in jobs)} notes on {len(jobs)} pages", flush=True)

    acc = collections.defaultdict(lambda: dict(crops=[], step=[], top=[], bottom=[], refined=[],
                                               **{k: [] for k in META}))
    with mp.Pool(args.workers) as pool:
        # imap, not imap_unordered: the cache order (and so training) must not depend on worker timing
        for i, (ms, split, crops, rows, bounds, extra) in enumerate(pool.imap(page_job, jobs), 1):
            if not rows:
                continue
            a = acc[(ms, split)]
            a["crops"].append(crops)
            for k, v in extra.items():
                a.setdefault("_" + k, []).append(v)
            a["step"] += [int(r["step"]) for r in rows]
            a["top"] += [b[0] for b in bounds]
            a["bottom"] += [b[1] for b in bounds]
            a["refined"] += [b[2] for b in bounds]
            for k in META:
                a[k] += [r[k] for r in rows]
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} pages", flush=True)

    for (ms, split), a in sorted(acc.items()):
        crops = np.concatenate(a.pop("crops"))
        extra = {k[1:]: np.concatenate(a.pop(k)) for k in [k for k in a if k.startswith("_")]}
        path = out_dir / f"{ms}_{split}.npz"
        np.savez(path, crops=crops, step=np.array(a.pop("step"), np.int8),
                 **{k: np.array(v) for k, v in a.items()}, **extra)
        print(f"{path.name}: {len(crops)} crops, refined {100 * np.mean(a['refined']):.1f}%, "
              f"{path.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
