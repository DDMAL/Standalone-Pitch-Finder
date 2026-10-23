"""Image check of Einsiedeln 'Reviewed_once' before adding it to the v2 dataset.

Reviewed_once (folios ~001r-080v) was reviewed before Neon's C-clef octave
fix, so everything under a C clef is encoded an octave low (C clef read as
C3) -- but it was never octave-dragged: 98% of its note zones are identical to
Processed (MEI-only comparison, see ../dataset/). Its own zones should
therefore be as good as option (a). This checks that against the page images,
side by side with the set already in use:

  reviewed      10 'Reviewed no pitch correction needed' folios, option (a)
                boxes (Processed zones by xml:id), C clef = C4
  reviewed_once 10 Reviewed_once folios spread over the set, own zones, C clef = C3

Per note: the box check of verify_bboxes.py (verified / shifted / no_blob),
and a pitch check -- where the matched notehead sits on the <sb> staff vs.
the stave step its pitch + clef give.

    python check_reviewed_once.py      # images from <repo>/data/images/einsiedeln
"""
import csv
from pathlib import Path

import cv2
import numpy as np

from check_ms73 import staff_pos
from fix_zones import EINSIEDELN, FOLIOS, LETTERS, PROCESSED, REVIEWED, parse, staff_step_px
from verify_bboxes import detect_blobs, evaluate, match, neume_sizes, render, staff_band
import verify_bboxes

DATA = Path(__file__).resolve().parents[3] / "data"
ONCE = DATA / "mei" / "einsiedeln" / "reviewed_once"
IMAGES = DATA / "images" / "einsiedeln"
OUT = EINSIEDELN / "reviewed_once_check"
N_PAGES = 10
C_REF = {"reviewed": 4 * 7, "reviewed_once": 3 * 7}  # pitch index of a C clef's line
F_REF = 3 * 7 + LETTERS.index("f")


def once_folios():
    full_reviewed = DATA / "mei" / "einsiedeln" / "reviewed"  # all 290, not just the 10 kept here
    folios = sorted(f.stem.rsplit("_", 1)[1] for f in ONCE.glob("*.mei") if not (full_reviewed / f.name).exists())
    idx = np.linspace(0, len(folios) - 1, N_PAGES).round().astype(int)
    return [folios[i] for i in idx]


def label(r, c_ref):
    shape, line = r["clef"]
    return 2 * (line - 1) + r["pi"] - (c_ref if shape == "C" else F_REF)


def check(folio, which):
    mei = (REVIEWED if which == "reviewed" else ONCE) / f"CH-E_611_{folio}.mei"
    rev = parse(mei)
    steps = staff_step_px(rev)
    step_med = float(np.median([v[0] for v in steps.values()]))
    wz = float(np.median([r["box"][2] - r["box"][0] for r in rev if r["kind"] == "nc"]))
    staves = {}
    for r in rev:
        if r["sz"] is not None and r["staff"] not in staves:
            staves[r["staff"]] = staff_band(r["sz"], steps[r["staff"]][0])
    gray = cv2.imread(str(IMAGES / f"Einsiedeln__Stiftsbibliothek__Codex_611_{folio}.jpg"), cv2.IMREAD_GRAYSCALE)
    masks = [([r["box"] for r in rev if r["kind"] == "syl"], 4, 4),
             ([r["box"] for r in rev if r["kind"] == "clef"], 0.5 * wz, 1.2 * step_med)]
    blobs = detect_blobs(gray, staves, step_med, masks)
    if which == "reviewed":
        proc = {r["id"]: r for r in parse(PROCESSED / mei.name)}
        variant = "a"
    else:
        proc, variant = {}, "orig"
    rows, used_b, _ = evaluate(folio, variant, rev, proc, steps, blobs, wz, neume_sizes(mei))

    # pitch check needs the matched blob, which evaluate() doesn't return: redo the match
    elems = [dict(r, box=(proc[r["id"]]["box"] if r["id"] in proc else None) if variant == "a" else r["box"])
             for r in rev if r["kind"] in ("nc", "custos")]
    m, _ = match(elems, blobs, {s: v[0] for s, v in steps.items()}, wz)
    pres = [staff_pos(e["sz"], blobs[m[i][0]]["cx"], blobs[m[i][0]]["cy"]) - label(e, C_REF[which])
            for i, e in enumerate(elems)
            if i in m and e["kind"] == "nc" and e["clef"] and e["clef"][1] and 1 <= e["clef"][1] <= 4]

    verify_bboxes.OUT = OUT  # render() writes to verify_bboxes.OUT
    render(folio, IMAGES, rows, blobs, [])
    (OUT / f"CH-E_611_{folio}_verify.png").rename(OUT / f"{which}_CH-E_611_{folio}_verify.png")
    st = [r["status"] for r in rows if r["kind"] == "nc"]
    return dict(which=which, folio=folio, n=len(st), verified=st.count("verified"), shifted=st.count("shifted"),
                no_blob=st.count("no_blob"), no_box=st.count("no_box"), pitch_n=len(pres),
                pitch_median=float(np.median(pres)) if pres else np.nan,
                pitch_ok=int(np.sum(np.abs(np.array(pres) - np.median(pres)) <= 0.5)) if pres else 0,
                pitch_off7=int(np.sum(np.abs(np.abs(np.array(pres)) - 7) <= 1)) if pres else 0)


def main():
    OUT.mkdir(exist_ok=True)
    res = [check(f, "reviewed") for f in FOLIOS] + [check(f, "reviewed_once") for f in once_folios()]
    print(f"{'set':14s} {'folio':6s} {'nc':>5s} | {'verified':>8s} {'shifted':>8s} {'no_blob':>8s} | "
          f"pitch: {'median':>6s} {'ok':>6s} {'off by 7':>8s}")
    for r in res:
        n = r["n"]
        print(f"{r['which']:14s} {r['folio']:6s} {n:5d} | {100*r['verified']/n:7.1f}% {100*r['shifted']/n:7.1f}% "
              f"{100*r['no_blob']/n:7.1f}% | {r['pitch_median']:+6.2f} {100*r['pitch_ok']/max(1,r['pitch_n']):5.1f}% "
              f"{r['pitch_off7']:8d}")
    print()
    for which in ("reviewed", "reviewed_once"):
        rs = [r for r in res if r["which"] == which]
        n, pn = sum(r["n"] for r in rs), sum(r["pitch_n"] for r in rs)
        print(f"ALL {which:14s} nc {n:5d}: verified {100*sum(r['verified'] for r in rs)/n:5.1f}%  "
              f"shifted {100*sum(r['shifted'] for r in rs)/n:5.1f}%  no_blob {100*sum(r['no_blob'] for r in rs)/n:5.1f}%  "
              f"no_box {sum(r['no_box'] for r in rs)} | pitch within 0.5 step (after per-page median) "
              f"{100*sum(r['pitch_ok'] for r in rs)/pn:5.1f}% of {pn}, off by an octave {sum(r['pitch_off7'] for r in rs)}")
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)


if __name__ == "__main__":
    main()
