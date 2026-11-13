"""Out-of-distribution test: Salzinnes (CDN-Hsmu M2149.L4), a square-notation
manuscript none of the models has seen.

Only the 6 pages whose images are on the HF dataset are used (DDMAL-lab/All-Images,
all-images-used-for-mothra/CDN-Hsmu_M2149.L4_*_resized*.png -> data/images/salzinnes/),
with their MEI from e2e-omr-resources resulting_mei_files/Salzinnes/reviewed
(-> data/mei/salzinnes/reviewed/). Checked before use:
  - the images are exactly 1/4 of the MEI <surface> (0.2501 both ways, all 6 pages)
  - reviewed and processed share their zones (a few re-drawn) and every octave:
    no octave drag as in Einsiedeln's Reviewed set
  - C clefs are written in the old C3 convention: read as C3, every page's note
    zones sit where their pitch puts them (median -0.45 step, 100% within half a
    step of it, as Einsiedeln's); read as C4 they are 6.5 steps off. So, as for
    Einsiedeln's Reviewed_once, oct is raised by 1 under C clefs before labelling.

Single-note neumes, crops cut exactly as for training (dataset/build_crops.py),
staff from the MEI <sb> zones; every checkpoint in train.CONFIGS is scored.

    python eval_ood.py   # -> results/ood_salzinnes.csv, figures/ood_salzinnes_overlay.jpg
"""
import collections
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dataset"))
from build_crops import CONTEXT, bbox_bounds, crop_bounds, cut, ic_like_box  # noqa: E402
from build_dataset import M, c_convention, label_of, walk  # noqa: E402

from data import model_input  # noqa: E402
from train import CKPT, CONFIGS, build_model, device, predict_steps  # noqa: E402

DATA = HERE.parents[1] / "data"
MEI_DIR = DATA / "mei" / "salzinnes" / "reviewed"
IMG_DIR = DATA / "images" / "salzinnes"
TILT = {"": "punctum", "s": "virga", "n": "reverse virga", "se": "inclinatum"}


def page_rows(mei_path, image_path):
    """notes.csv-style rows (image pixels) for one page; excluded notes dropped."""
    recs, zones, (sw, sh) = walk(mei_path)
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    sx, sy = gray.shape[1] / sw, gray.shape[0] / sh
    conv = c_convention(recs, zones)
    assert conv in ("C3", ""), f"{mei_path.name}: C clefs read as {conv}"
    rows, dropped = [], collections.Counter()
    for r in recs:
        if r["kind"] != "nc":
            continue
        if conv == "C3" and r["clef_shape"] == "C" and (r["oct"] or "").isdigit():
            r["oct"] = str(int(r["oct"]) + 1)
        step, why = label_of(r)
        z, sb = zones.get(r["facs"]), zones.get(r["sb_facs"])
        why = why or ("no_box" if z is None else "no_staff_zone" if sb is None else None)
        if why:
            dropped[why] += 1
            continue
        rows.append(dict(id=r["id"], staff=r["staff"], step=step, tilt=r["tilt"], neume_size=str(r["neume_size"]),
                         neume_id=r["neume_id"], nc_index=r["nc_index"], ligated=r["ligated"],
                         pname=r["pname"], oct=r["oct"],
                         box_ulx=z[0] * sx, box_uly=z[1] * sy, box_lrx=z[2] * sx, box_lry=z[3] * sy,
                         staff_ulx=sb[0] * sx, staff_uly=sb[1] * sy, staff_lrx=sb[2] * sx, staff_lry=sb[3] * sy,
                         staff_rotate=sb[4]))
    return gray, rows, dropped


def overlay(gray, rows, path):
    """MEI staff lines and note boxes drawn on the page, to check the 1/4 scaling."""
    from build_crops import staff_span
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for r in rows:
        x0, x1 = int(r["staff_ulx"]), int(r["staff_lrx"])
        for x_a, x_b in ((x0, x1),):
            ta, ba, _ = staff_span(r, x_a)
            tb, bb, _ = staff_span(r, x_b)
            for k in range(4):
                cv2.line(img, (x_a, int(ta + k * (ba - ta) / 3)), (x_b, int(tb + k * (bb - tb) / 3)), (255, 90, 0), 1)
        cv2.rectangle(img, (int(r["box_ulx"]), int(r["box_uly"])), (int(r["box_lrx"]), int(r["box_lry"])),
                      (0, 0, 230) if r["neume_size"] == "1" else (0, 160, 0), 1)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    dev = device()
    pages = []
    for img in sorted(IMG_DIR.glob("*.png")):
        folio = re.search(r"_(\d{3}[rv])_resized", img.name).group(1)
        pages.append((folio, MEI_DIR / f"CDN-Hsmu_M2149.L4_{folio}.mei", img))

    singles, crops = [], collections.defaultdict(list)
    n_all, dropped = 0, collections.Counter()
    for k, (folio, mei, img) in enumerate(pages):
        gray, rows, drop = page_rows(mei, img)
        n_all += len(rows)
        dropped += drop
        if k == 0:
            overlay(gray, rows, HERE / "figures" / "ood_salzinnes_overlay.jpg")
        ss = [r for r in rows if r["neume_size"] == "1"]
        for r in ss:
            r["folio"] = folio
        singles += ss
        staff_b = crop_bounds(ss)
        crops["staff"] += [cut(gray, r["box_ulx"], r["box_lrx"], t, b) for r, (t, b, _) in zip(ss, staff_b)]
        for h in (128, 256):
            ctx_h = round(h * CONTEXT)
            crops[f"bbox{h}"] += [cut(gray, r["box_ulx"], r["box_lrx"], t, b, ctx_h)
                                  for r, (t, b, _) in zip(ss, bbox_bounds(ss, rows))]
        unit = float(np.median([r["box_lry"] - r["box_uly"] for r in rows]))
        for r in ss:  # IC-style box + its mask, as build_crops --mode icbox
            x0, y0, x1, y1, _ = ic_like_box(gray, {**{k_: str(v) for k_, v in r.items()}}, unit)
            cy = (y0 + y1) / 2
            t, b = cy - 10 * unit, cy + 10 * unit
            ctx_h = round(128 * CONTEXT)
            crops["ic128"].append(cut(gray, x0, x1, t, b, ctx_h))
            c0 = cy - CONTEXT * (b - t) / 2
            to_row = lambda y: int(round((y - c0) / (CONTEXT * (b - t)) * ctx_h))
            m = np.zeros((ctx_h, 32), np.uint8)
            m[max(0, to_row(y0)):max(0, to_row(y1)) + 1] = 255
            crops["ic128_mask"].append(m)
    truth = np.array([r["step"] for r in singles])
    print(f"Salzinnes, {len(pages)} pages: {n_all} usable notes, {len(singles)} single-note neumes; dropped {dict(dropped)}")
    print("labels:", dict(sorted(collections.Counter(truth.tolist()).items())))

    source = {"staff": "staff", "bbox128": "bbox128", "bbox256": "bbox256", "icbox128": "ic128"}
    out = []
    for name, c in CONFIGS.items():
        if not (CKPT / f"{name}.pt").exists():
            continue
        kind = Path(c["crops"]).name.replace("v2_crops_", "").replace("v2_crops", "staff")
        arr = {"crops": np.stack(crops[source[kind]])}
        if c["mask"]:
            arr["masks"] = np.stack(crops["ic128_mask"])
        m = build_model(name).to(dev)
        m.load_state_dict(torch.load(CKPT / f"{name}.pt", map_location=dev))
        pred = predict_steps(m, torch.tensor(model_input(arr, c["mask"]), device=dev), c["classifier"], dev, out_h=c["out_h"])
        err = np.abs(pred - truth)
        row = dict(model=name, n=len(truth), exact=float(np.mean(err == 0)), within1=float(np.mean(err <= 1)))
        for t in TILT:
            sel = np.array([r["tilt"] == t for r in singles])
            if sel.sum() >= 10:
                row[f"exact_{TILT[t]}"] = float(np.mean(err[sel] == 0))
        for f, *_ in pages:
            sel = np.array([r["folio"] == f for r in singles])
            row[f"exact_{f}"] = float(np.mean(err[sel] == 0))
        out.append(row)
        np.save(HERE / "results" / f"ood_salzinnes_pred_{name}.npy", pred)
    np.save(HERE / "results" / "ood_salzinnes_truth.npy", truth)

    keys = list(dict.fromkeys(k for r in out for k in r))
    with open(HERE / "results" / "ood_salzinnes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    folios = [p[0] for p in pages]
    print(f"\n{'model':28s} {'exact':>7s} {'±1':>6s} | " + " ".join(f"{f:>6s}" for f in folios) + " | virga  punctum")
    for r in out:
        print(f"{r['model']:28s} {100 * r['exact']:6.2f}% {100 * r['within1']:5.1f}% | "
              + " ".join(f"{100 * r[f'exact_{f}']:5.1f}%" for f in folios)
              + f" | {100 * r.get('exact_virga', np.nan):5.1f}% {100 * r.get('exact_punctum', np.nan):5.1f}%")


if __name__ == "__main__":
    main()
