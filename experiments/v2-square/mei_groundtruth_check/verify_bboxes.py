"""Verify MEI note bboxes against noteheads detected independently in the image.

Detector (uses no MEI note positions): Otsu-binarize, morphologically open
with a square of ~0.43 staff steps -- that erases staff lines, stems, hairlines
and most text strokes and leaves solid noteheads as blobs -- and keep the blobs
inside each staff band (from the <sb> zones, which match the real staff lines).
Blobs taller than ~1.5 steps (or much wider than a notehead) are stacked /
adjacent noteheads merged together and are split into one piece per notehead.
Blobs inside the MEI's lyric (<syl>) or clef boxes are dropped -- those
elements were never dragged, and using them leaks no note-position information.

Matching: per staff, optimal one-to-one assignment (Hungarian) between MEI
notes (nc / custos) and blobs, gated at +-0.75 zone widths and +-1.5 steps.
The box-centre vs ink-centroid convention offset is estimated per folio as the
median over all matches and removed. Each note is then:
  verified  residual within 0.5 step vertically (i.e. on the same line/space
            as the detected notehead) and 0.6 zone widths horizontally
  shifted   matched, but more than half a step away
  no_blob   nothing notehead-like nearby (oblique ligature strokes, faint ink, or a bad box)
Blobs left unclaimed are reported as possible missing notes (or detector false
positives: decoration, text residue).

Boxes evaluated: 'orig' = as distributed (Reviewed), 'a' = Processed zones by
xml:id, 'b' = Reviewed shifted down 7 steps under C clefs (see fix_zones.py).

    python verify_bboxes.py --image-dir <dir with Einsiedeln__Stiftsbibliothek__Codex_611_<folio>.jpg>
"""
import argparse
import csv
import math
from pathlib import Path

import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from fix_zones import EINSIEDELN, FOLIOS, M, OCTAVE_STEPS, PROCESSED, REVIEWED, XMLID, parse, staff_step_px

HERE = Path(__file__).resolve().parent
OUT = EINSIEDELN / "bbox_verification"
PITCHED = ("nc", "custos")
BAND_ABOVE, BAND_BELOW = 3.0, 1.5  # staff band margins, in steps
VERIFY_DY, VERIFY_DX = 0.5, 0.6    # steps / zone widths


def staff_band(sz, step):
    ulx, uly, lrx, lry, rot = sz
    t = math.tan(math.radians(rot))
    H = (lry - uly) - (lrx - ulx) * abs(t)
    x0 = ulx if t >= 0 else lrx

    def top(x):
        return uly + (x - x0) * t
    return ulx, lrx, (lambda x: top(x) - BAND_ABOVE * step), (lambda x: top(x) + H + BAND_BELOW * step)


def _inside_any(x, y, boxes, pad_x=0, pad_y=0):
    return any(b[0] - pad_x <= x <= b[2] + pad_x and b[1] - pad_y <= y <= b[3] + pad_y for b in boxes)


def band_mask(shape, staves):
    """Pixels inside any staff band, for fitting the threshold to the music area."""
    mask = np.zeros(shape, np.uint8)
    for ulx, lrx, lo, hi in staves.values():
        pts = np.array([[ulx, lo(ulx)], [lrx, lo(lrx)], [lrx, hi(lrx)], [ulx, hi(ulx)]], np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def binarize(gray, adaptive, region=None, bg_k=None):
    """region + bg_k: subtract a local paper estimate (grey closing, bg_k px) and
    fit Otsu to the contrast inside region only. MS73 needs both: a whole-page
    Otsu splits paper from the dark book border (~105) and lands below faded
    grey noteheads (~150-180), and paper brightness varies across the page."""
    if region is None:
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        sm = cv2.medianBlur(gray, 5)  # also evens out the hatched texture inside heads
        bg = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (bg_k, bg_k)))
        diff = cv2.subtract(bg, sm)
        t, _ = cv2.threshold(diff[region > 0].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, ink = cv2.threshold(diff, t, 255, cv2.THRESH_BINARY)
    if adaptive:
        # global Otsu loses faint light-brown noteheads; a local threshold keeps them
        loc = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 61, 12)
        ink = cv2.bitwise_or(ink, loc)
    return ink


def detect_blobs(gray, staves, step_med, masks, adaptive=False, head_h=None, local_bg=False):
    """masks: list of (boxes, pad_x, pad_y); blobs centred inside any are dropped.
    head_h: notehead height in px, which sizes the opening kernel and the
    merged-blob split. Defaults to one step.
    local_bg: threshold contrast against the local paper, fitted inside the
    staff bands (see binarize) -- for faded pages; Einsiedeln runs without it."""
    head_h = head_h or step_med
    region = band_mask(gray.shape, staves) if local_bg else None
    ink = binarize(gray, adaptive, region, int(3 * head_h) | 1)
    k = max(5, int(round(0.43 * head_h)) | 1)
    heads = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(heads, 8)
    w_unit = float(np.median(st[1:, 2])) * 1.3 if n > 1 else head_h
    pieces = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        kv, kh = max(1, round(h / head_h)), max(1, round(w / w_unit))
        # a merged blob holds several noteheads: one piece per notehead-sized slice
        if kv >= 2:
            groups = [(ys >= j * h / kv) & (ys < (j + 1) * h / kv) for j in range(kv)]
        elif kh >= 2:
            groups = [(xs >= j * w / kh) & (xs < (j + 1) * w / kh) for j in range(kh)]
        else:
            groups = [np.ones(len(xs), bool)]
        for g in groups:
            if g.sum() < 0.15 * k * k:
                continue
            pieces.append(dict(cx=x + xs[g].mean(), cy=y + ys[g].mean(), area=int(g.sum()),
                               w=int(np.ptp(xs[g])) + 1, h=int(np.ptp(ys[g])) + 1, merged=len(groups) > 1))
    blobs = []
    for b in pieces:
        if any(_inside_any(b["cx"], b["cy"], boxes, px, py) for boxes, px, py in masks):
            continue
        for s, (ulx, lrx, lo, hi) in staves.items():
            if ulx - 60 <= b["cx"] <= lrx + 60 and lo(b["cx"]) <= b["cy"] <= hi(b["cx"]):
                blobs.append(dict(b, staff=s))
                break
    return blobs


def match(elems, blobs, step_of_staff, wz):
    """Per-staff optimal one-to-one assignment in step-normalized distance."""
    GATE = 1e6
    out, used_b = {}, set()
    for s in {e["staff"] for e in elems}:
        ei = [i for i, e in enumerate(elems) if e["staff"] == s and e["box"] is not None]
        bi = [j for j, b in enumerate(blobs) if b["staff"] == s]
        if not ei or not bi:
            continue
        st = step_of_staff[s]
        C = np.full((len(ei), len(bi)), GATE)
        D = {}
        for a, i in enumerate(ei):
            x0, y0, x1, y1 = elems[i]["box"]
            ex, ey = (x0 + x1) / 2, (y0 + y1) / 2
            for c, j in enumerate(bi):
                dx, dy = blobs[j]["cx"] - ex, blobs[j]["cy"] - ey
                if abs(dx) <= 0.75 * wz and abs(dy) <= 1.5 * st:
                    C[a, c] = math.hypot(dx / wz, dy / st)
                    D[a, c] = (dx, dy)
        for a, c in zip(*linear_sum_assignment(C)):
            if C[a, c] < GATE:
                out[ei[a]] = (bi[c],) + D[a, c]
                used_b.add(bi[c])
    return out, used_b


def neume_sizes(path):
    root = ET.parse(path).getroot()
    sizes = {}
    for neume in root.iter(M + "neume"):
        ncs = neume.findall(M + "nc")
        for nc in ncs:
            sizes[nc.get(XMLID)] = len(ncs)
    return sizes


def evaluate(folio, variant, rev, proc, steps, blobs, wz, nsize):
    elems = []
    for r in rev:
        if r["kind"] not in PITCHED:
            continue
        box = r["box"]
        if variant == "a":
            box = proc[r["id"]]["box"] if r["id"] in proc else None
        elif variant == "b" and r["clef"] and r["clef"][0] == "C":
            dy = OCTAVE_STEPS * steps[r["staff"]][0]
            box = (box[0], box[1] + dy, box[2], box[3] + dy)
        elems.append(dict(r, box=box))
    step_of_staff = {s: v[0] for s, v in steps.items()}
    m, used_b = match(elems, blobs, step_of_staff, wz)

    dxs = np.array([m[i][1] for i in m])
    dys = np.array([m[i][2] / step_of_staff[elems[i]["staff"]] for i in m])
    bias_x = float(np.median(dxs)) if len(dxs) else 0.0
    bias_y = float(np.median(dys)) if len(dys) else 0.0  # in steps

    rows = []
    for i, e in enumerate(elems):
        row = dict(folio=folio, variant=variant, id=e["id"], kind=e["kind"], staff=e["staff"],
                   neume_size=nsize.get(e["id"], 0), box=e["box"], res_dx=np.nan, res_dy_steps=np.nan,
                   merged_blob=False)
        if e["box"] is None:
            row["status"] = "no_box"
        elif i not in m:
            row["status"] = "no_blob"
        else:
            j, dx, dy = m[i]
            rdx, rdy = dx - bias_x, dy / step_of_staff[e["staff"]] - bias_y
            row.update(res_dx=rdx, res_dy_steps=rdy, merged_blob=blobs[j]["merged"])
            ok = abs(rdy) <= VERIFY_DY and abs(rdx) <= VERIFY_DX * wz
            row["status"] = "verified" if ok else "shifted"
        rows.append(row)
    return rows, used_b, (bias_x, bias_y)


def notehead_like(b, ref_area):
    ar = b["w"] / max(1, b["h"])
    return 0.3 * ref_area <= b["area"] <= 3 * ref_area and 0.4 <= ar <= 2.5


def render(folio, image_dir, rows, blobs, missing, scale=0.4):
    img = cv2.imread(str(image_dir / f"Einsiedeln__Stiftsbibliothek__Codex_611_{folio}.jpg"))
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    color = {"verified": (0, 170, 0), "shifted": (0, 140, 255), "no_blob": (0, 0, 230)}
    for b in blobs:
        cv2.circle(img, (int(b["cx"] * scale), int(b["cy"] * scale)), 2, (255, 200, 0), -1)
    for r in rows:
        if r["box"] is None:
            continue
        x0, y0, x1, y1 = (int(round(v * scale)) for v in r["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), color[r["status"]], 2)
    for b in missing:
        x, y, d = int(b["cx"] * scale), int(b["cy"] * scale), 7
        cv2.line(img, (x - d, y - d), (x + d, y + d), (255, 0, 0), 2)
        cv2.line(img, (x - d, y + d), (x + d, y - d), (255, 0, 0), 2)
    legend = [("GREEN: verified (on a detected notehead)", (0, 150, 0)),
              ("ORANGE: matched blob but position off", (0, 120, 230)),
              ("RED: no notehead found near the box", (0, 0, 220)),
              ("BLUE X: notehead with no MEI note (missing note?)", (255, 0, 0)),
              ("cyan dot: detected notehead centroid", (200, 160, 0))]
    for i, (t, c) in enumerate(legend):
        cv2.rectangle(img, (5, 5 + i * 26), (560, 27 + i * 26), (255, 255, 255), -1)
        cv2.putText(img, t, (10, 23 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"CH-E_611_{folio}_verify.png"), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--adaptive", action="store_true",
                    help="also use a local threshold: recovers some faint noteheads (+0.5pt verified) "
                         "but picks up more noise (+1pt shifted, +50%% unclaimed blobs)")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    all_rows, summary = [], []
    for folio in FOLIOS:
        rev = parse(REVIEWED / f"CH-E_611_{folio}.mei")
        proc = {r["id"]: r for r in parse(PROCESSED / f"CH-E_611_{folio}.mei")}
        steps = staff_step_px(rev)
        step_med = float(np.median([v[0] for v in steps.values()]))
        wz = float(np.median([r["box"][2] - r["box"][0] for r in rev if r["kind"] == "nc"]))
        staves = {}
        for r in rev:
            if r["sz"] is not None and r["staff"] not in staves:
                staves[r["staff"]] = staff_band(r["sz"], steps[r["staff"]][0])
        gray = cv2.imread(str(args.image_dir / f"Einsiedeln__Stiftsbibliothek__Codex_611_{folio}.jpg"),
                          cv2.IMREAD_GRAYSCALE)
        syl = [r["box"] for r in rev if r["kind"] == "syl"]
        clefs = [r["box"] for r in rev if r["kind"] == "clef"]
        masks = [(syl, 4, 4), (clefs, 0.5 * wz, 1.2 * step_med)]
        blobs = detect_blobs(gray, staves, step_med, masks, adaptive=args.adaptive)
        nsize = neume_sizes(REVIEWED / f"CH-E_611_{folio}.mei")

        for variant in ("orig", "a", "b"):
            rows, used_b, bias = evaluate(folio, variant, rev, proc, steps, blobs, wz, nsize)
            all_rows += rows
            ref_area = np.median([blobs[i]["area"] for i in used_b]) if used_b else 1
            missing = [b for i, b in enumerate(blobs) if i not in used_b and notehead_like(b, ref_area)]
            st = [r["status"] for r in rows if r["kind"] == "nc"]
            summary.append(dict(folio=folio, variant=variant, n=len(st),
                                verified=st.count("verified"), shifted=st.count("shifted"),
                                no_blob=st.count("no_blob"), no_box=st.count("no_box"),
                                unclaimed_heads=len(missing), bias_dx=bias[0], bias_dy_steps=bias[1]))
            if variant == "a":
                render(folio, args.image_dir, rows, blobs, missing)

    print(f"{'folio':6s} {'var':4s} {'n':>4s} | {'verified':>9s} {'shifted':>8s} {'no_blob':>8s} | unclaimed heads | bias dy")
    for s in summary:
        n = s["n"]
        print(f"{s['folio']:6s} {s['variant']:4s} {n:4d} | {100*s['verified']/n:8.1f}% {100*s['shifted']/n:7.1f}% "
              f"{100*s['no_blob']/n:7.1f}% | {s['unclaimed_heads']:15d} | {s['bias_dy_steps']:+.2f} step")

    print("\nALL 10 FOLIOS, nc by neume size:")
    for label, sel in (("all notes", lambda r: True),
                       ("single-note neumes", lambda r: r["neume_size"] == 1),
                       ("notes in multi-note neumes", lambda r: r["neume_size"] > 1)):
        for v in ("orig", "a", "b"):
            rs = [r for r in all_rows if r["variant"] == v and r["kind"] == "nc" and sel(r)]
            c = {k: sum(r["status"] == k for r in rs) for k in ("verified", "shifted", "no_blob")}
            n = len(rs)
            print(f"  {label:27s} {v:4s} n={n:5d}  verified {100*c['verified']/n:5.1f}%  "
                  f"shifted {100*c['shifted']/n:5.1f}%  no_blob {100*c['no_blob']/n:5.1f}%")
    ua = sum(s["unclaimed_heads"] for s in summary if s["variant"] == "a")
    print(f"\nunclaimed notehead-like blobs under (a): {ua} total, {ua/len(FOLIOS):.0f} per folio")

    keys = [k for k in all_rows[0] if k != "box"]
    with open(OUT / "per_note.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in keys})
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()
