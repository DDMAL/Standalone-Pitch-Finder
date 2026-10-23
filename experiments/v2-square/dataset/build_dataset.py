"""Build the v2 note index for Einsiedeln + MS73: one row per pitched element
(nc, custos) with its box in image pixels, its staff, and its label -- the
stave step, i.e. diatonic steps above the bottom staff line (bottom line = 0,
the same convention as v1). Then summarize it and split it by page into
train/val/test.

Sources (fetch them first with fetch_sources.py; all land in gitignored data/):
  Einsiedeln  (1) 'Reviewed no pitch correction needed': pitches from it, boxes and
              staff zones from 'Processed' matched by xml:id -- the Reviewed zones
              were dragged up an octave in Neon (option (a), see
              ../mei_groundtruth_check/README.md).
              (2) 'Reviewed_once' (pages not also in (1)): its own zones, which were
              never dragged. It was reviewed before Neon's C-clef fix, so pitches
              under a C clef are an octave low; each page's convention is read off
              its own geometry (c_convention) and oct is raised by 1 under C clefs
              where it is C3 (oct_shift = 1). Image check: check_reviewed_once.py.
              Coordinates are image pixels.
  MS73        'Processed' (no Reviewed folder exists). Coordinates are half the
              image size, so everything is scaled by MS73_SCALE. Zero-height note
              zones get the page's median note size at their recorded top-left
              corner (box_repaired = 1).

Split: by page, so no page contributes to two splits; per manuscript,
SPLIT_FRACS of the eligible pages, shuffled with a manuscript-specific seed
(so adding pages to one manuscript leaves the other's split alone). Pages
listed in TRAIN_ONLY never go to val/test.

Outputs (in this folder):
  notes.csv   one row per pitched element, excluded ones included with a reason
  pages.csv   one row per MEI page: image, counts, split
  stats.txt   the summary printed at the end

    python build_dataset.py
"""
import collections
import csv
import io
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "data"
M = "{http://www.music-encoding.org/ns/mei}"
XMLID = "{http://www.w3.org/XML/1998/namespace}id"
LETTERS = "cdefgab"
CLEF_REF = {"C": 4 * 7 + LETTERS.index("c"), "F": 3 * 7 + LETTERS.index("f")}  # C4 / F3 on the clef line
# plausible stave steps: up to ~1.5 line gaps beyond a 4-line staff. The 9
# Einsiedeln notes outside (-8..-5, 13), each on a different staff, are
# annotation errors; v1's classifier covered -1..8.
STEP_RANGE = (-3, 10)
MS73_SCALE = 2.0  # MS73 Processed README; measured 2.00 x 2.00 on the 11 pages checked
# Einsiedeln note-zone centres sit this many steps below where their pitch puts
# them (uniform, measured on images and in the MEI geometry); the right C-clef
# reading reproduces it, the wrong one is 7 steps away
EINSIEDELN_BOX_OFFSET = -0.4
SEED = 42
SPLIT_FRACS = (("train", 0.70), ("val", 0.15), ("test", 0.15))

# Einsiedeln folios that also sit as single-page images at the top level of the
# HF dataset, next to the pages our YOLO staff-line model was labelled on
# (A-Gu-30_032r, Aarau_MsMurF2_14v, ...). Possibly staff-model training pages:
# kept out of val/test so a detected-staff evaluation isn't flattered.
TRAIN_ONLY = {
    "einsiedeln": {"014r", "030r", "032r", "041v", "043r", "045v", "057r", "062r", "078r",
                   "080r", "095v", "113v", "122r", "175v", "209v"},
    "ms73": {52},
}


# ---------------------------------------------------------------- parsing

def zones_of(root):
    zones = {}
    for z in root.iter(M + "zone"):
        coords = [z.get(k) for k in ("ulx", "uly", "lrx", "lry")]
        if None in coords:  # Neon occasionally writes empty / partial zones
            continue
        zones[z.get(XMLID)] = tuple(float(c) for c in coords) + (float(z.get("rotate", 0)),)
    return zones


def walk(path):
    """Pitched elements in document order, with staff, staff zone, clef in force and neume."""
    root = ET.parse(path).getroot()
    zones = zones_of(root)
    surf = root.find(f".//{M}surface")
    neume_of = {}
    for neume in root.iter(M + "neume"):
        ncs = list(neume.iter(M + "nc"))
        for i, nc in enumerate(ncs):
            neume_of[nc.get(XMLID)] = (neume.get(XMLID), len(ncs), i)
    recs, staff, sb_facs, clef, clef_staff = [], -1, None, None, None
    for layer in root.iter(M + "layer"):
        for el in layer.iter():
            tag = el.tag.replace(M, "")
            facs = (el.get("facs") or "").lstrip("#")
            if tag == "sb":
                staff, sb_facs = staff + 1, facs
            elif tag == "clef":
                clef, clef_staff = (el.get("shape"), el.get("line")), staff
            elif tag in ("nc", "custos"):
                nid, nsize, ni = neume_of.get(el.get(XMLID), ("", 0, 0))
                recs.append(dict(
                    id=el.get(XMLID), kind=tag, staff=staff, sb_facs=sb_facs, facs=facs,
                    pname=el.get("pname"), oct=el.get("oct"),
                    clef_shape=clef[0] if clef else "", clef_line=clef[1] if clef else "",
                    clef_carried=int(clef is not None and clef_staff != staff),
                    neume_id=nid, neume_size=nsize, nc_index=ni,
                    ligated=int(el.get("ligated") == "true"), tilt=el.get("tilt") or "",
                    curve=el.get("curve") or "", liquescent=int(el.find(M + "liquescent") is not None)))
    return recs, zones, (int(surf.get("lrx")), int(surf.get("lry")))


def label_of(r):
    """(stave step, None) or (None, reason)."""
    if r["clef_shape"] not in CLEF_REF:
        return None, "no_clef"
    if not r["clef_line"] or not 1 <= int(r["clef_line"]) <= 4:
        return None, "bad_clef_line"
    if r["pname"] not in LETTERS or not (r["oct"] or "").isdigit():
        return None, "no_pitch"
    pi = int(r["oct"]) * 7 + LETTERS.index(r["pname"])
    step = 2 * (int(r["clef_line"]) - 1) + pi - CLEF_REF[r["clef_shape"]]
    if not STEP_RANGE[0] <= step <= STEP_RANGE[1]:
        return step, "step_outlier"
    return step, None


def staff_pos(z, sb):
    """Box centre in steps above the bottom line of an <sb> zone (4 lines, rotated box)."""
    ulx, uly, lrx, lry, rot = sb
    t = np.tan(np.radians(rot))
    H = (lry - uly) - (lrx - ulx) * abs(t)
    x0 = ulx if t >= 0 else lrx
    cx, cy = (z[0] + z[2]) / 2, (z[1] + z[3]) / 2
    return (uly + (cx - x0) * t + H - cy) / (H / 6)


def c_convention(recs, zones):
    """'C3' or 'C4': which reading of the C clef puts this page's note zones where
    their pitches say. Neon places a zone from its pitch, so the right reading
    lands on EINSIEDELN_BOX_OFFSET and the wrong one 7 steps away. '' if the page
    has no measurable note under a C clef, None if neither reading fits."""
    fits = {}
    for name, ref in (("C3", 3 * 7), ("C4", 4 * 7)):
        d = []
        for r in recs:
            if r["kind"] != "nc" or r["clef_shape"] != "C" or r["clef_line"] not in ("1", "2", "3", "4"):
                continue
            z, sb = zones.get(r["facs"]), zones.get(r["sb_facs"])
            if z and sb and z[3] - z[1] >= 1 and r["pname"] in LETTERS:
                pi = int(r["oct"]) * 7 + LETTERS.index(r["pname"])
                d.append(staff_pos(z, sb) - (2 * (int(r["clef_line"]) - 1) + pi - ref))
        if not d:
            return ""
        fits[name] = abs(float(np.median(d)) - EINSIEDELN_BOX_OFFSET)
    best = min(fits, key=fits.get)
    return best if fits[best] < 1.5 else None


# ---------------------------------------------------------------- images

def listing(name):
    entries = json.loads((DATA / "hf_listings" / f"{name}.json").read_text())
    return {e["path"].split("/", 1)[1] for e in entries if e["type"] == "file"}


def einsiedeln_image(folio, names):
    short = folio.lstrip("0")
    for cand in (f"Einsiedeln__Stiftsbibliothek__Codex_611_{folio}.jpg",
                 f"Einsiedeln__Stiftsbibliothek__Codex_611_{folio} copy.jpg"):
        if cand in names:
            return cand
    alt = [n for n in names if re.fullmatch(rf"Einsiedeln__Stiftsbibliothek__Codex_611_89__0_\d+_{short}_0\.jpg", n)]
    return alt[0] if alt else ""


def ms73_image(page, names):
    for cand in (f"MS73_{page:03d}.jpg", f"_A_15th_century_Italian_antiphonal___manuscript__0_{page}_p.__0.jpg"):
        if cand in names:
            return cand
    return ""


# ---------------------------------------------------------------- per page

def box_px(z, scale):
    return tuple(round(v * scale, 1) for v in z[:4])


def einsiedeln_pages():
    names = listing("einsiedeln")
    reviewed = sorted((DATA / "mei/einsiedeln/reviewed").glob("*.mei"))
    for path in reviewed:
        folio = path.stem.rsplit("_", 1)[1]
        recs, _, surface = walk(path)
        _, pzones, _ = walk(DATA / "mei/einsiedeln/processed" / path.name)
        for r in recs:
            r["zone"] = pzones.get(r["facs"])        # option (a): Processed zone by id
            r["sb_zone"] = pzones.get(r["sb_facs"])  # staff zones were not dragged; same source anyway
            r["repaired"] = r["oct_shift"] = 0
        yield dict(manuscript="einsiedeln", page=folio, source="reviewed", c_convention="C4",
                   mei=f"einsiedeln/reviewed/{path.name}", image=einsiedeln_image(folio, names),
                   surface=surface, scale=1.0), recs
    done = {p.name for p in reviewed}
    for path in sorted((DATA / "mei/einsiedeln/reviewed_once").glob("*.mei")):
        if path.name in done:  # 253r is in both; the fully reviewed version wins
            continue
        folio = path.stem.rsplit("_", 1)[1]
        recs, zones, surface = walk(path)
        conv = c_convention(recs, zones)
        for r in recs:
            r["zone"], r["sb_zone"] = zones.get(r["facs"]), zones.get(r["sb_facs"])
            r["repaired"] = 0
            r["oct_shift"] = int(conv == "C3" and r["clef_shape"] == "C" and (r["oct"] or "").isdigit())
            if r["oct_shift"]:
                r["oct"] = str(int(r["oct"]) + 1)
        yield dict(manuscript="einsiedeln", page=folio, source="reviewed_once",
                   c_convention="unclear" if conv is None else conv,
                   mei=f"einsiedeln/reviewed_once/{path.name}", image=einsiedeln_image(folio, names),
                   surface=surface, scale=1.0), recs


def ms73_pages():
    names = listing("ms73")
    parsed, seen = [], {}
    for path in sorted((DATA / "mei/ms73/processed").glob("*.mei")):
        page = int(re.search(r"[-_](\d{3})$", path.stem).group(1))
        if page in seen:  # CDN_Mlr_MS73_109r-220.mei is a byte-identical copy of CDN-Mlr_...
            if path.read_bytes() != seen[page].read_bytes():
                raise ValueError(f"two different MEI files for MS73 page {page}: {seen[page].name}, {path.name}")
            print(f"skipping {path.name}: identical copy of {seen[page].name}")
            continue
        seen[page] = path
        recs, zones, surface = walk(path)
        sized = [zones[r["facs"]] for r in recs if r["kind"] == "nc" and r["facs"] in zones
                 and zones[r["facs"]][3] - zones[r["facs"]][1] >= 1]
        parsed.append((path, recs, zones, surface, sized))
    every = [z for *_, sized in parsed for z in sized]  # fallback for a page with no sized note (162)
    for path, recs, zones, surface, sized in parsed:
        page = int(re.search(r"[-_](\d{3})$", path.stem).group(1))
        ref = sized or every
        wz = float(np.median([z[2] - z[0] for z in ref]))
        hz = float(np.median([z[3] - z[1] for z in ref]))
        for r in recs:
            z = zones.get(r["facs"])
            r["oct_shift"] = 0
            r["repaired"] = int(z is not None and z[3] - z[1] < 1)
            if r["repaired"]:
                z = (z[0], z[1], z[0] + wz, z[1] + hz, z[4])
            r["zone"] = z
            r["sb_zone"] = zones.get(r["sb_facs"])
        yield dict(manuscript="ms73", page=page, source="processed", c_convention="C4",
                   mei=f"ms73/processed/{path.name}",
                   image=ms73_image(page, names), surface=surface, scale=MS73_SCALE), recs


# ---------------------------------------------------------------- split

def assign_splits(pages):
    for ms in sorted({p["manuscript"] for p in pages}):
        rng = random.Random(f"{SEED}:{ms}")
        mine = [p for p in pages if p["manuscript"] == ms]
        for p in mine:
            if not p["image"]:
                p["split"], p["split_note"] = "", "no_image"
            elif p["c_convention"] == "unclear":
                p["split"], p["split_note"] = "", "octave_unclear"
            elif p["n_usable"] == 0:
                p["split"], p["split_note"] = "", "no_usable_notes"
            elif p["page"] in TRAIN_ONLY.get(ms, ()):
                p["split"], p["split_note"] = "train", "train_only"
        free = [p for p in mine if "split" not in p]
        rng.shuffle(free)
        n_fixed_train = sum(p.get("split") == "train" for p in mine)
        n_total = len(free) + n_fixed_train
        n_val = round(SPLIT_FRACS[1][1] * n_total)
        n_test = round(SPLIT_FRACS[2][1] * n_total)
        for i, p in enumerate(free):
            p["split"] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            p["split_note"] = ""


# ---------------------------------------------------------------- report

def report(pages, notes):
    def pct(a, b):
        return f"{100 * a / b:5.1f}%" if b else "    -"

    for ms in ("einsiedeln", "ms73"):
        P = [p for p in pages if p["manuscript"] == ms]
        N = [n for n in notes if n["manuscript"] == ms]
        nc = [n for n in N if n["kind"] == "nc"]
        print(f"\n================ {ms.upper()} ================")
        print(f"MEI pages {len(P)}; with image {sum(bool(p['image']) for p in P)}; "
              f"in split {sum(bool(p['split']) for p in P)}")
        for p in P:
            if not p["split"]:
                print(f"  excluded page {p['page']}: {p['split_note']}")
        print(f"pitched elements: nc {len(nc)}, custos {len(N) - len(nc)}")
        reasons = collections.Counter(n["exclude"] for n in nc if n["exclude"])
        usable = [n for n in nc if not n["exclude"]]
        print(f"usable nc {len(usable)} ({pct(len(usable), len(nc))}); excluded: "
              + (", ".join(f"{k} {v}" for k, v in reasons.most_common()) or "none"))
        if not usable:
            continue
        per_page = collections.Counter(n["page"] for n in usable)
        print(f"usable nc per page: median {int(np.median(list(per_page.values())))}, "
              f"min {min(per_page.values())}, max {max(per_page.values())}")
        print(f"neumes: single-note {pct(sum(n['neume_size'] == 1 for n in usable), len(usable))} of notes, "
              f"ligated {pct(sum(n['ligated'] for n in usable), len(usable))}, "
              f"liquescent {pct(sum(n['liquescent'] for n in usable), len(usable))}, "
              f"inclinatum (tilt=se) {pct(sum(n['tilt'] == 'se' for n in usable), len(usable))}, "
              f"box repaired (zero height) {pct(sum(n['box_repaired'] for n in usable), len(usable))}")
        clefs = collections.Counter(n["clef_shape"] for n in usable)
        print("clef in force: " + ", ".join(f"{k} {pct(v, len(usable))}" for k, v in clefs.most_common())
              + f"; carried over from a previous staff {pct(sum(n['clef_carried'] for n in usable), len(usable))}")
        steps = collections.Counter(n["step"] for n in usable)
        lo, hi = min(steps), max(steps)
        out_v1 = sum(v for k, v in steps.items() if not -1 <= k <= 8)
        print(f"stave step (label) range {lo}..{hi}; outside v1's -1..8: {out_v1} notes ({pct(out_v1, len(usable))})")
        print("  step:  " + " ".join(f"{k:>5d}" for k in range(lo, hi + 1)))
        print("  share: " + " ".join(f"{100 * steps[k] / len(usable):5.1f}" for k in range(lo, hi + 1)))
        src = collections.Counter(p["source"] for p in P if p["split"])
        conv = collections.Counter(p["c_convention"] for p in P if p["split"] and p["source"] == "reviewed_once")
        print("pages in split by source: " + ", ".join(f"{k} {v}" for k, v in src.items())
              + (f"; reviewed_once C-clef convention: " + ", ".join(f"{k or 'no C clef'} {v}" for k, v in conv.items())
                 if conv else ""))
        print("split   pages  usable nc   single  ligated  repaired | step share (-2..9)")
        for s, _ in SPLIT_FRACS:
            sp = {p["page"] for p in P if p["split"] == s}
            u = [n for n in usable if n["page"] in sp]
            st = collections.Counter(n["step"] for n in u)
            print(f"{s:6s} {len(sp):6d} {len(u):10d}   {pct(sum(n['neume_size'] == 1 for n in u), len(u))}  "
                  f"{pct(sum(n['ligated'] for n in u), len(u))}   {pct(sum(n['box_repaired'] for n in u), len(u))} | "
                  + " ".join(f"{100 * st[k] / max(1, len(u)):4.1f}" for k in range(-2, 10)))
        fixed = sorted(str(p["page"]) for p in P if p["split_note"] == "train_only")
        if fixed:
            print(f"train_only pages present: {', '.join(fixed)}")


def main():
    pages, notes = [], []
    for meta, recs in list(einsiedeln_pages()) + list(ms73_pages()):
        s = meta["scale"]
        for r in recs:
            step, why = label_of(r)  # an outlier keeps its step for inspection
            if r["zone"] is None:
                why = why or "no_box"
            elif r["sb_zone"] is None:
                why = why or "no_staff_zone"
            elif r["zone"][3] - r["zone"][1] < 1:
                why = why or "zero_height_box"
            z = box_px(r["zone"], s) if r["zone"] else ("",) * 4
            sz = box_px(r["sb_zone"], s) + (r["sb_zone"][4],) if r["sb_zone"] else ("",) * 5
            notes.append(dict(
                manuscript=meta["manuscript"], page=meta["page"], source=meta["source"], image=meta["image"],
                id=r["id"], kind=r["kind"],
                staff=r["staff"], neume_id=r["neume_id"], neume_size=r["neume_size"], nc_index=r["nc_index"],
                pname=r["pname"], oct=r["oct"], oct_shift=r["oct_shift"],
                clef_shape=r["clef_shape"], clef_line=r["clef_line"],
                clef_carried=r["clef_carried"], step="" if step is None else step,
                ligated=r["ligated"], tilt=r["tilt"], curve=r["curve"], liquescent=r["liquescent"],
                box_ulx=z[0], box_uly=z[1], box_lrx=z[2], box_lry=z[3], box_repaired=r["repaired"],
                staff_ulx=sz[0], staff_uly=sz[1], staff_lrx=sz[2], staff_lry=sz[3], staff_rotate=sz[4],
                exclude=why or ""))
        mine = notes[len(notes) - len(recs):]
        pages.append(dict(meta, n_nc=sum(n["kind"] == "nc" for n in mine),
                          n_usable=sum(n["kind"] == "nc" and not n["exclude"] for n in mine),
                          n_staves=len({r["staff"] for r in recs})))
    for ms in {p["manuscript"] for p in pages}:  # a page or image in two splits would leak
        for key in ("page", "image"):
            vals = [p[key] for p in pages if p["manuscript"] == ms and p[key] != ""]
            dup = {v for v in vals if vals.count(v) > 1}
            if dup:
                raise ValueError(f"{ms}: {key} used by more than one MEI page: {sorted(dup)}")
    assign_splits(pages)
    split_of = {(p["manuscript"], p["page"]): p["split"] for p in pages}
    for n in notes:
        n["split"] = split_of[(n["manuscript"], n["page"])]
        if not n["split"] and not n["exclude"]:
            n["exclude"] = "page_excluded"

    with open(HERE / "notes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split"] + [k for k in notes[0] if k != "split"])
        w.writeheader()
        w.writerows(notes)
    with open(HERE / "pages.csv", "w", newline="") as f:
        cols = ["manuscript", "page", "source", "c_convention", "split", "split_note", "image", "mei",
                "n_nc", "n_usable", "n_staves",
                "surface_w", "surface_h", "scale"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in pages:
            w.writerow(dict({k: p.get(k, "") for k in cols}, surface_w=p["surface"][0], surface_h=p["surface"][1]))

    buf = io.StringIO()
    with redirect_stdout(buf):
        report(pages, notes)
    (HERE / "stats.txt").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())


if __name__ == "__main__":
    main()
