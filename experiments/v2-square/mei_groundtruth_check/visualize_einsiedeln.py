"""Visualize ALL ground truth encoded in an Einsiedeln MEI file (bbox + pitch +
clef + custos + divLine) on top of its page image, as distributed -- i.e. the
'Reviewed' zones, octave drag included (see fix_zones.py).

MEI source: github.com/DDMAL/e2e-omr-resources, resulting_mei_files/Einsiedeln/
            Reviewed no pitch correction needed/CH-E_611_<folio>.mei (public repo)
Page image: huggingface.co/datasets/DDMAL-lab/All-Images,
            "Pics Einsiedeln/Einsiedeln__Stiftsbibliothek__Codex_611_<folio>.jpg"
            (private dataset -- needs an HF token with DDMAL-lab org access;
            not vendored into this repo). Fetch with:
              curl -sL "https://huggingface.co/datasets/DDMAL-lab/All-Images/resolve/main/Pics%20Einsiedeln/Einsiedeln__Stiftsbibliothek__Codex_611_081r.jpg" \\
                -H "Authorization: Bearer $HF_TOKEN" -o <image dir>/Einsiedeln__Stiftsbibliothek__Codex_611_081r.jpg

    python visualize_einsiedeln.py --image-dir <dir> [folio ...]
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

from fix_zones import EINSIEDELN, FOLIOS, M, REVIEWED, XMLID

OUT = EINSIEDELN / "groundtruth_viz"
SCALE = 0.4  # full page is 4872x6496 -- downscale for a manageable file

NOTE_COLOR = (0, 60, 255)      # bold orange-red -- per-note (nc) bbox + pitch label
NEUME_COLOR = (180, 180, 180)  # neutral grey -- neume-level (grouped) bbox
CLEF_COLOR = (255, 140, 0)     # blue-orange -- clef bbox
CUSTOS_COLOR = (0, 200, 200)   # yellow -- custos bbox
DIVLINE_COLOR = (128, 0, 128)  # purple -- divLine (barline) bbox


def render(folio, image_dir):
    root = ET.parse(REVIEWED / f"CH-E_611_{folio}.mei").getroot()
    zones = {}
    for z in root.iter(M + "zone"):
        if z.get("ulx") is None:
            continue
        zones[z.get(XMLID)] = (int(float(z.get("ulx"))), int(float(z.get("uly"))),
                               int(float(z.get("lrx"))), int(float(z.get("lry"))))

    def zone_for(el):
        return zones.get((el.get("facs") or "").lstrip("#"))

    img = cv2.imread(str(image_dir / f"Einsiedeln__Stiftsbibliothek__Codex_611_{folio}.jpg"))
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

    def draw_box(bbox, color, thickness, label=None, label_color=None):
        ulx, uly, lrx, lry = bbox
        p1 = (round(ulx * SCALE), round(uly * SCALE))
        p2 = (round(lrx * SCALE), round(lry * SCALE))
        cv2.rectangle(img, p1, p2, color, thickness)
        if label:
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            org = (p1[0], max(th + 2, p1[1] - 3))
            cv2.rectangle(img, (org[0] - 1, org[1] - th - baseline), (org[0] + tw + 1, org[1] + baseline), (255, 255, 255), -1)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color or color, 1, cv2.LINE_AA)

    n = dict(notes=0, neumes=0, clefs=0, custos=0, divLines=0)
    for neume in root.iter(M + "neume"):
        nc_boxes = []
        for nc in neume.findall(M + "nc"):
            bbox = zone_for(nc)
            if bbox is None:
                continue
            liq = "~" if nc.find(M + "liquescent") is not None else ""
            draw_box(bbox, NOTE_COLOR, 2, f"{nc.get('pname', '?')}{nc.get('oct', '?')}{liq}")
            nc_boxes.append(bbox)
            n["notes"] += 1
        if nc_boxes:
            draw_box((min(b[0] for b in nc_boxes), min(b[1] for b in nc_boxes),
                      max(b[2] for b in nc_boxes), max(b[3] for b in nc_boxes)), NEUME_COLOR, 3)
            n["neumes"] += 1
    for clef in root.iter(M + "clef"):
        bbox = zone_for(clef)
        if bbox:
            draw_box(bbox, CLEF_COLOR, 3, f"clef.{clef.get('shape', '?').lower()}{clef.get('line', '?')}")
            n["clefs"] += 1
    for custos in root.iter(M + "custos"):
        bbox = zone_for(custos)
        if bbox:
            draw_box(bbox, CUSTOS_COLOR, 3, f"custos {custos.get('pname', '?')}{custos.get('oct', '?')}", (0, 140, 140))
            n["custos"] += 1
    for dl in root.iter(M + "divLine"):
        bbox = zone_for(dl)
        if bbox:
            draw_box(bbox, DIVLINE_COLOR, 2)
            n["divLines"] += 1

    out = OUT / f"CH-E_611_{folio}_groundtruth_viz.png"
    cv2.imwrite(str(out), img)
    print(f"{folio}: " + " ".join(f"{k}={v}" for k, v in n.items()) + f" -> {out.relative_to(EINSIEDELN.parent)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("folios", nargs="*", default=FOLIOS)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for folio in args.folios:
        render(folio, args.image_dir)


if __name__ == "__main__":
    main()
