"""Check MS73 (CDN-Mlr MS73) MEI ground truth against its page images.

MEI: e2e-omr-resources/resulting_mei_files/MS73/Processed (corrected once in
Neon; there is no Reviewed folder in the repo). Per that folder's README its
coordinates are half the original image size -- so the page image is
downscaled to the MEI <surface> size before anything else.
Images: HF DDMAL-lab/All-Images, "Pics MS73", named either
MS73_<page>.jpg or _A_15th_century_Italian_antiphonal___manuscript__0_<page>_p.__0.jpg
(<page> is the number after the '-' in the MEI name; verified on page 096,
whose image matches that MEI's surface exactly while pages 95/97 don't).

Per page:
  flat      notes whose zone has zero height (seen on page 096 as thin dashes)
  octave    clef box vs. where a note on the clef line would sit, in steps,
            from a robust fit of note y on pitch step: ~0 means pitches use the
            correct C4 convention, ~+7 the old C3 bug seen in Einsiedeln
  verified  the notehead check from verify_bboxes.py; flat zones are given
            the median box size anchored at their recorded top-left corner

    python check_ms73.py --image-dir <dir with the MS73 page images>
"""
import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from fix_zones import M, parse, staff_step_px, step_of
from verify_bboxes import VERIFY_DX, VERIFY_DY, detect_blobs, match, staff_band

HERE = Path(__file__).resolve().parent
MS73 = HERE / "ms73"

cv2.setNumThreads(4)


def image_for(page, image_dir):
    for name in (f"MS73_{page:03d}.jpg", f"_A_15th_century_Italian_antiphonal___manuscript__0_{page}_p.__0.jpg"):
        if (image_dir / name).exists():
            return image_dir / name
    raise FileNotFoundError(f"no image for page {page} in {image_dir}")


def octave_offsets(rev, normal_ids):
    """Per clef: (clef box uly - predicted uly of a note on the clef line) in steps."""
    out = []
    for s in sorted({r["staff"] for r in rev}):
        ncs = [r for r in rev if r["kind"] == "nc" and r["staff"] == s and r["id"] in normal_ids
               and r["clef"] and r["clef"][1]]
        if len(ncs) < 5:
            continue
        sz = ncs[0]["sz"]
        A = np.array([[1, step_of(r), r["box"][0] - sz[0]] for r in ncs])
        y = np.array([r["box"][1] for r in ncs])
        keep = np.ones(len(y), bool)
        for _ in range(3):  # robust: drop gross outliers and refit
            coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
            r_ = np.abs(y - A @ coef)
            keep = r_ <= max(10.0, 3 * np.median(r_))
        if keep.sum() < 4 or not (8 < -coef[1] < 80):
            continue
        for cl in (r for r in rev if r["kind"] == "clef" and r["staff"] == s and r["clef"] and r["clef"][1]):
            pred = coef[0] + coef[1] * 2 * (cl["clef"][1] - 1) + coef[2] * (cl["box"][0] - sz[0])
            out.append((cl["box"][1] - pred) / -coef[1])
    return out


def check_page(mei_path, image_dir, out_dir):
    page = int(mei_path.stem.rsplit("-", 1)[1])
    rev = parse(mei_path)
    surf = ET.parse(mei_path).getroot().find(f".//{M}surface")
    sw, sh = int(surf.get("lrx")), int(surf.get("lry"))
    full = cv2.imread(str(image_for(page, image_dir)), cv2.IMREAD_GRAYSCALE)
    ratio = (full.shape[1] / sw, full.shape[0] / sh)
    gray = cv2.resize(full, (sw, sh), interpolation=cv2.INTER_AREA)

    ncs = [r for r in rev if r["kind"] in ("nc", "custos")]
    flat = {r["id"] for r in ncs if r["box"][3] - r["box"][1] < 1}
    normal = [r for r in ncs if r["id"] not in flat and r["kind"] == "nc"]
    head_h = float(np.median([r["box"][3] - r["box"][1] for r in normal]))
    wz = float(np.median([r["box"][2] - r["box"][0] for r in normal]))

    steps = staff_step_px([r for r in rev if r["id"] not in flat])
    step_med = float(np.median([v[0] for v in steps.values()]))
    staves = {}
    for r in rev:
        if r["sz"] is not None and r["staff"] not in staves and r["staff"] in steps:
            staves[r["staff"]] = staff_band(r["sz"], steps[r["staff"]][0])
    masks = [([r["box"] for r in rev if r["kind"] == "syl"], 4, 4),
             ([r["box"] for r in rev if r["kind"] == "clef"], 0.5 * wz, 0.6 * head_h)]
    blobs = detect_blobs(gray, staves, step_med, masks, head_h=head_h, local_bg=True)

    elems = []
    for r in ncs:
        if r["staff"] not in steps:
            continue
        box = r["box"]
        if r["id"] in flat:
            box = (box[0], box[1], box[0] + wz, box[1] + head_h)
        elems.append(dict(r, box=box, flat=r["id"] in flat))
    m, _ = match(elems, blobs, {s: v[0] for s, v in steps.items()}, wz)

    status, bias = {}, {}
    for grp in (False, True):
        idx = [i for i, e in enumerate(elems) if e["flat"] == grp and e["kind"] == "nc"]
        dys = [m[i][2] / steps[elems[i]["staff"]][0] for i in idx if i in m]
        dxs = [m[i][1] for i in idx if i in m]
        by, bx = (float(np.median(dys)), float(np.median(dxs))) if dys else (0.0, 0.0)
        bias[grp] = by
        for i in idx:
            if i not in m:
                status[i] = "no_blob"
                continue
            rdy = m[i][2] / steps[elems[i]["staff"]][0] - by
            rdx = m[i][1] - bx
            status[i] = "verified" if abs(rdy) <= VERIFY_DY and abs(rdx) <= VERIFY_DX * wz else "shifted"

    # pitch check: where the matched notehead sits on the <sb> staff vs. where its
    # pitch says it should (steps above the bottom line). Independent of the box.
    pitch_res = {}
    for i, e in enumerate(elems):
        if i in m and e["kind"] == "nc" and e["clef"] and e["clef"][1]:
            b = blobs[m[i][0]]
            pos = staff_pos(e["sz"], b["cx"], b["cy"])
            pitch_res[i] = pos - step_of(e)

    render(gray, elems, status, blobs, out_dir / f"{mei_path.stem}_verify.png", rev)
    oc = octave_offsets(rev, {r["id"] for r in normal})
    return dict(page=page, ratio=ratio, n=sum(e["kind"] == "nc" for e in elems), flat=len(flat & {r["id"] for r in ncs if r["kind"] == "nc"}),
                octave=float(np.median(oc)) if oc else float("nan"), n_clef_tests=len(oc),
                status=status, elems=elems, bias=bias, step=step_med, head_h=head_h, pitch_res=pitch_res)


def staff_pos(sz, x, y):
    """Steps above the bottom line of an <sb> zone (4 lines, rotated box)."""
    ulx, uly, lrx, lry, rot = sz
    t = math.tan(math.radians(rot))
    H = (lry - uly) - (lrx - ulx) * abs(t)
    x0 = ulx if t >= 0 else lrx
    bottom = uly + (x - x0) * t + H
    return (bottom - y) / (H / 6)


def staff_lines(sz):
    ulx, uly, lrx, lry, rot = sz
    t = math.tan(math.radians(rot))
    H = (lry - uly) - (lrx - ulx) * abs(t)
    x0 = ulx if t >= 0 else lrx
    return [((ulx, uly + (ulx - x0) * t + k * H / 3), (lrx, uly + (lrx - x0) * t + k * H / 3)) for k in range(4)]


def render(gray, elems, status, blobs, path, rev, scale=0.35):
    img = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for sz in {r["sz"] for r in rev if r["sz"] is not None}:  # the MEI's own staff lines
        for (xa, ya), (xb, yb) in staff_lines(sz):
            cv2.line(img, (int(xa * scale), int(ya * scale)), (int(xb * scale), int(yb * scale)), (200, 120, 255), 1)
    color = {"verified": (0, 170, 0), "shifted": (0, 140, 255), "no_blob": (0, 0, 230)}
    for b in blobs:
        cv2.circle(img, (int(b["cx"] * scale), int(b["cy"] * scale)), 3, (255, 200, 0), -1)
    for i, e in enumerate(elems):
        if i not in status:
            continue
        x0, y0, x1, y1 = (int(round(v * scale)) for v in e["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), color[status[i]], 3 if e["flat"] else 2)
        if e["flat"]:  # mark repaired zero-height boxes
            cv2.line(img, (x0, y0), (x1, y1), color[status[i]], 1)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", type=Path, required=True)
    args = ap.parse_args()
    out_dir = MS73 / "verify"
    out_dir.mkdir(exist_ok=True)

    res = [check_page(p, args.image_dir, out_dir) for p in sorted((MS73 / "mei").glob("*.mei"))]
    print(f"{'page':>4s} {'img/MEI':>9s} {'notes':>5s} {'flat':>9s} | {'octave (steps)':>15s} | "
          f"{'normal: verified shifted no_blob':>33s} | {'flat (repaired): verified shifted no_blob':>42s}")
    tot = {g: {"verified": 0, "shifted": 0, "no_blob": 0} for g in (False, True)}
    for r in res:
        cells = []
        for g in (False, True):
            st = [r["status"][i] for i, e in enumerate(r["elems"]) if i in r["status"] and e["flat"] == g]
            for k in tot[g]:
                tot[g][k] += st.count(k)
            n = len(st)
            cells.append("        -" if not n else
                         f"n={n:3d} {100*st.count('verified')/n:5.1f}% {100*st.count('shifted')/n:5.1f}% {100*st.count('no_blob')/n:5.1f}%")
        print(f"{r['page']:4d} {r['ratio'][0]:4.2f}x{r['ratio'][1]:4.2f} {r['n']:5d} {r['flat']:4d} ({100*r['flat']/max(1,r['n']):4.1f}%) | "
              f"{r['octave']:+6.2f} (n={r['n_clef_tests']:2d})   | {cells[0]:>33s} | {cells[1]:>42s}")
    for g, label in ((False, "normal boxes"), (True, "flat boxes (repaired)")):
        n = sum(tot[g].values())
        if n:
            print(f"ALL {label:22s} n={n:5d}  verified {100*tot[g]['verified']/n:5.1f}%  "
                  f"shifted {100*tot[g]['shifted']/n:5.1f}%  no_blob {100*tot[g]['no_blob']/n:5.1f}%")
    print("\npitch label vs. notehead position on the MEI staff (matched notes; residual in steps)")
    allr = []
    for r in res:
        v = np.array(list(r["pitch_res"].values()))
        if len(v):
            allr += list(v)
            print(f"  page {r['page']:3d}  n={len(v):3d}  median {np.median(v):+5.2f}  "
                  f"|res|<=0.5: {100*np.mean(np.abs(v - np.median(v)) <= 0.5):5.1f}% (after median)  "
                  f"{100*np.mean(np.abs(v) <= 0.5):5.1f}% (raw)  off by ~7: {np.sum(np.abs(np.abs(v) - 7) <= 1.5)}")
    allr = np.array(allr)
    print(f"  ALL n={len(allr)}  raw within 0.5 step: {100*np.mean(np.abs(allr) <= 0.5):.1f}%  within 1 step: {100*np.mean(np.abs(allr) <= 1):.1f}%")
    fb = [r["bias"][True] for r in res if r["flat"]]
    print(f"flat-box notehead offset from recorded top (median over pages): {np.median(fb):+.2f} steps"
          f"  [normal-box offset: {np.median([r['bias'][False] for r in res]):+.2f}]")


if __name__ == "__main__":
    main()
