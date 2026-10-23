"""Repair the octave-dragged zones in the 'Reviewed no pitch correction needed'
Einsiedeln MEI, two ways, and evaluate both against the page image.

Background (see e2e-omr-resources/resulting_mei_files/Einsiedeln/README.md and
the 'C clefs all good' note in the Reviewed folder): Rodan/Neon first assumed
C clefs default to C3, so everything after a C clef was encoded an octave too
low. Reviewers fixed the pitches by dragging neumes up an octave in Neon,
which also moved their zones up by 7 diatonic steps; clefs and text were not
dragged. Measured on 081r: same xml:ids, same pname, oct +1, x unchanged,
y up by 7 steps of each staff's own line spacing.

  (a) 'processed': take each element's zone from the pre-drag Processed MEI,
      matched by xml:id (pitches still come from the Reviewed file).
  (b) 'shift':     move each nc / custos / divLine under a C clef down by
      7 steps of its own staff's spacing, estimated from the Reviewed note
      zones themselves (which are exactly linear in pitch step and x).

Metrics, per element, against the Otsu-binarized page:
  ink    fraction of ink pixels inside the box
  align  vertical shift (within +-ALIGN_WIN px) that maximizes ink coverage;
         ~0 for a box that already sits on its glyph

Page images come from huggingface.co/datasets/DDMAL-lab/All-Images
("Pics Einsiedeln", private -- needs an HF token with DDMAL-lab access).

    python fix_zones.py --image-dir <dir with Einsiedeln__Stiftsbibliothek__Codex_611_<folio>.jpg>
"""
import argparse
import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
EINSIEDELN = HERE / "einsiedeln"
REVIEWED = EINSIEDELN / "mei" / "reviewed"    # as distributed (octave-dragged)
PROCESSED = EINSIEDELN / "mei" / "processed"  # same pages before the drag
M = "{http://www.music-encoding.org/ns/mei}"
XMLID = "{http://www.w3.org/XML/1998/namespace}id"
LETTERS = "cdefgab"
TARGETS = ("nc", "custos", "divLine")   # what the octave drag moved
CONTROLS = ("clef", "syl")              # what it did not
ALIGN_WIN = 60
OCTAVE_STEPS = 7

FOLIOS = ["081r", "085r", "095r", "125r", "128r", "165r", "180r", "200r", "230r", "260r"]


def parse(path):
    """Elements in document order, each tagged with its staff (sb) and the clef in force."""
    root = ET.parse(path).getroot()
    zones = {}
    for z in root.iter(M + "zone"):
        coords = [z.get(k) for k in ("ulx", "uly", "lrx", "lry")]
        if None in coords:  # Neon occasionally writes empty / partial zones; unusable
            continue
        zones[z.get(XMLID)] = tuple(float(c) for c in coords) + (float(z.get("rotate", 0)),)
    out, staff, sz, clef = [], -1, None, None
    for el in root.find(f".//{M}layer").iter():
        tag = el.tag.replace(M, "")
        z = zones.get((el.get("facs") or "").lstrip("#"))
        if tag == "sb":
            staff, sz = staff + 1, z
            continue
        if tag == "clef":  # @line is occasionally missing; shape alone still says "C clef"
            clef = (el.get("shape"), int(el.get("line")) if el.get("line") else None)
        if tag not in TARGETS + CONTROLS or z is None:
            continue
        rec = dict(id=el.get(XMLID), kind=tag, staff=staff, sz=sz, clef=clef, box=z[:4])
        if el.get("pname"):
            rec["pi"] = int(el.get("oct")) * 7 + LETTERS.index(el.get("pname"))
        out.append(rec)
    return out


def step_of(rec):
    shape, line = rec["clef"]
    ref = 4 * 7 + LETTERS.index("c") if shape == "C" else 3 * 7 + LETTERS.index("f")
    return 2 * (line - 1) + rec["pi"] - ref


def staff_step_px(recs):
    """px per diatonic step (positive) for each staff, from a robust fit of
    nc uly ~ step + x; falls back to the sb zone's de-rotated height."""
    out = {}
    for s in sorted({r["staff"] for r in recs}):
        ncs = [r for r in recs if r["kind"] == "nc" and r["staff"] == s and r["clef"] and r["clef"][1]]
        sz = next((r["sz"] for r in recs if r["staff"] == s and r["sz"]), None)
        est, src = None, None
        if len(ncs) >= 4:
            A = np.array([[1, step_of(r), r["box"][0] - sz[0]] for r in ncs])
            y = np.array([r["box"][1] for r in ncs])
            keep = np.ones(len(y), bool)
            for _ in range(3):
                coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
                keep = np.abs(y - A @ coef) < 5
                if keep.sum() < 4:
                    break
            if keep.sum() >= 4 and 17 < -coef[1] < 45:
                est, src = -coef[1], f"fit({keep.sum()}/{len(y)})"
        if est is None and sz is not None:
            ulx, uly, lrx, lry, rot = sz
            h = (lry - uly) - (lrx - ulx) * abs(math.tan(math.radians(rot)))
            est, src = h / 6, "sb-zone"
        out[s] = (est, src)
    return out


def ink_metrics(integ, box, shape):
    H, W = shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, x1 = max(0, x0), min(W - 1, x1)
    h = y1 - y0

    def cov(yy0):
        yy1 = yy0 + h
        if yy0 < 0 or yy1 >= H or x1 <= x0 or h <= 0:
            return -1.0
        s = integ[yy1, x1] - integ[yy0, x1] - integ[yy1, x0] + integ[yy0, x0]
        return s / ((x1 - x0) * h)

    ink = cov(y0)
    best = max(range(-ALIGN_WIN, ALIGN_WIN + 1), key=lambda d: cov(y0 + d))
    return ink, best


def run_folio(folio, image_dir, viz_dir):
    rev = parse(REVIEWED / f"CH-E_611_{folio}.mei")
    proc = {r["id"]: r for r in parse(PROCESSED / f"CH-E_611_{folio}.mei")}
    steps = staff_step_px(rev)

    gray = cv2.imread(str(image_dir / f"Einsiedeln__Stiftsbibliothek__Codex_611_{folio}.jpg"),
                      cv2.IMREAD_GRAYSCALE)
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    integ = cv2.integral(ink.astype(np.uint8))

    rows = []
    for r in rev:
        a = proc.get(r["id"], {}).get("box")
        b = None
        if r["kind"] in TARGETS:
            if r["clef"] and r["clef"][0] == "C":
                dy = OCTAVE_STEPS * steps[r["staff"]][0]
                b = (r["box"][0], r["box"][1] + dy, r["box"][2], r["box"][3] + dy)
            else:
                b = r["box"]  # not under a C clef: was never dragged
        else:
            b = r["box"]      # controls are left alone by (b)
        row = dict(folio=folio, id=r["id"], kind=r["kind"], staff=r["staff"],
                   clef=r["clef"][0] if r["clef"] else "", matched=a is not None)
        for name, box in (("orig", r["box"]), ("a", a), ("b", b)):
            if box is None:
                row[f"{name}_ink"], row[f"{name}_align"] = np.nan, np.nan
            else:
                row[f"{name}_ink"], row[f"{name}_align"] = ink_metrics(integ, box, gray.shape)
        row["a_minus_b_dy"] = (a[1] - b[1]) if a is not None else np.nan
        row["a_minus_b_dx"] = (a[0] - b[0]) if a is not None else np.nan
        row["box_a"], row["box_b"] = a, b
        rows.append(row)

    # how far each staff's notes were actually dragged (Reviewed - Processed),
    # in units of that staff's step; option (b) assumes exactly -7 everywhere
    drag = {}
    for s, (px, _) in steps.items():
        d = [r["box"][1] - proc[r["id"]]["box"][1] for r in rev
             if r["kind"] == "nc" and r["staff"] == s and r["id"] in proc]
        if d:
            drag[s] = np.median(d) / px
    for row in rows:
        row["staff_drag_steps"] = drag.get(row["staff"], np.nan)

    render(folio, image_dir, viz_dir, rows)
    return rows, steps, drag


def render(folio, image_dir, viz_dir, rows, scale=0.4):
    img = cv2.imread(str(image_dir / f"Einsiedeln__Stiftsbibliothek__Codex_611_{folio}.jpg"))
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def rect(box, color, t):
        x0, y0, x1, y1 = (int(round(v * scale)) for v in box)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, t)

    for r in rows:
        if r["kind"] not in TARGETS:
            continue
        if r["box_a"] is not None:
            rect(r["box_a"], (0, 170, 0), 3)       # green   = (a) processed zones
        rect(r["box_b"], (255, 0, 255), 1)         # magenta = (b) 7-step shift
    legend = [("GREEN thick: (a) zones from Processed MEI", (0, 140, 0)),
              ("MAGENTA thin: (b) Reviewed zones shifted down 7 steps", (200, 0, 200))]
    for i, (txt, col) in enumerate(legend):
        cv2.rectangle(img, (5, 5 + i * 28), (610, 29 + i * 28), (255, 255, 255), -1)
        cv2.putText(img, txt, (10, 24 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    viz_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(viz_dir / f"CH-E_611_{folio}_fix_compare.png"), img)


def summarize(rows, label):
    """ink: median coverage and % of boxes with coverage > 0.3.
    consistent: % of boxes whose ink-peak offset lies within 12px (~0.4 step)
    of that option's median offset -- i.e. sitting on its glyph with the same
    box convention as everyone else, whatever that convention is.
    a matched: % of elements whose xml:id exists (with a usable zone) in Processed.
    a~b: % agreeing within 6px / 15px (15px ~ half a step)."""
    def pct(x):
        return 100 * np.mean(x) if len(x) else float("nan")

    def consistent(vals):
        return pct(np.abs(vals - np.median(vals)) <= 12) if len(vals) else float("nan")

    print(f"\n=== {label} ===")
    print(f"{'kind':8s} {'n':>5s} | {'ink median':>17s} | {'% ink>0.3':>17s} | {'% consistent':>17s} | {'peak offset':>11s} | a matched | a~b <=6px <=15px")
    print(f"{'':8s} {'':>5s} | {'orig':>5s} {'(a)':>5s} {'(b)':>5s} | {'orig':>5s} {'(a)':>5s} {'(b)':>5s} | {'orig':>5s} {'(a)':>5s} {'(b)':>5s} | {'(a)':>5s} {'(b)':>5s} |")
    for kind in TARGETS + CONTROLS:
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        g = lambda k: np.array([r[k] for r in rs if not np.isnan(r[k])])
        d = np.abs(g("a_minus_b_dy"))
        print(f"{kind:8s} {len(rs):5d} | {np.median(g('orig_ink')):5.2f} {np.median(g('a_ink')):5.2f} {np.median(g('b_ink')):5.2f} | "
              f"{pct(g('orig_ink') > 0.3):5.1f} {pct(g('a_ink') > 0.3):5.1f} {pct(g('b_ink') > 0.3):5.1f} | "
              f"{consistent(g('orig_align')):5.1f} {consistent(g('a_align')):5.1f} {consistent(g('b_align')):5.1f} | "
              f"{np.median(g('a_align')):+5.0f} {np.median(g('b_align')):+5.0f} | "
              f"{pct(np.array([r['matched'] for r in rs])):6.1f}%   | {pct(d <= 6):5.1f}% {pct(d <= 15):5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", type=Path, required=True)
    args = ap.parse_args()
    viz_dir = EINSIEDELN / "fix_comparison"

    all_rows = []
    for folio in FOLIOS:
        rows, steps, drag = run_folio(folio, args.image_dir, viz_dir)
        all_rows += rows
        srcs = [src for _, src in steps.values()]
        odd = {s: round(v, 2) for s, v in drag.items() if abs(v + OCTAVE_STEPS) > 0.5}
        print(f"\n{folio}: {len(rows)} elements, staff spacing from fit on "
              f"{sum(s.startswith('fit') for s in srcs)}/{len(srcs)} staves; "
              f"actual drag per staff (steps) = {np.round(sorted(drag.values()), 2).tolist()}"
              + (f"  <-- staves NOT dragged by 7: {odd}" if odd else ""))
        summarize(rows, folio)
    summarize(all_rows, "ALL 10 FOLIOS")

    keys = [k for k in all_rows[0] if not k.startswith("box_")]
    with open(viz_dir / "per_element.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in keys})
    print(f"\nwrote {viz_dir / 'per_element.csv'}")


if __name__ == "__main__":
    main()
