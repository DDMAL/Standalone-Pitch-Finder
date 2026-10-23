"""Draw an MS73 page's MEI ground truth on its image: staff lines (from the
<sb> zones), note boxes + pitch, clefs, custos.

MEI coordinates are half the image size (see check_ms73.py), so the image is
first brought to the MEI <surface> size. Zero-height note zones -- 57% of notes
in the 10-page sample -- are drawn in a separate colour at the page's median
note size, anchored at the recorded top-left corner, with the recorded flat
line kept visible inside.

    python visualize_ms73.py --image-dir <dir with the MS73 images> [page ...]
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from check_ms73 import MS73, image_for, staff_lines

HERE = Path(__file__).resolve().parent
M = "{http://www.music-encoding.org/ns/mei}"
XMLID = "{http://www.w3.org/XML/1998/namespace}id"
SCALE = 0.45  # of the MEI (half-size) coordinates -> ~2100 px wide

NOTE = (0, 60, 255)       # orange-red: note box with a real height
FLAT = (200, 0, 200)      # magenta: zero-height note box, given the median size
CLEF = (255, 120, 0)      # blue
CUSTOS = (0, 170, 200)    # yellow
STAFF = (255, 90, 0)     # blue, 1px: MS73 staff lines are red ink, a thick line beside one reads as a second line
NEUME = (170, 170, 170)   # grey: neume group


def draw_label(img, text, x, y, color):
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    (tw, th), bl = cv2.getTextSize(text, font, fs, 1)
    y = max(th + 2, y)
    cv2.rectangle(img, (x - 1, y - th - bl), (x + tw + 1, y + bl - 1), (255, 255, 255), -1)
    cv2.putText(img, text, (x, y), font, fs, color, 1, cv2.LINE_AA)


def render(mei_path, image_dir, out_dir):
    page = int(mei_path.stem.rsplit("-", 1)[1])
    root = ET.parse(mei_path).getroot()
    surf = root.find(f".//{M}surface")
    sw, sh = int(surf.get("lrx")), int(surf.get("lry"))
    zones = {}
    for z in root.iter(M + "zone"):
        if z.get("ulx") is None:
            continue
        zones[z.get(XMLID)] = tuple(float(z.get(k)) for k in ("ulx", "uly", "lrx", "lry")) + (float(z.get("rotate", 0)),)

    def zone(el):
        return zones.get((el.get("facs") or "").lstrip("#"))

    ncs = [(nc, zone(nc)) for nc in root.iter(M + "nc") if zone(nc)]
    real = [z for _, z in ncs if z[3] - z[1] >= 1]
    wz = float(np.median([z[2] - z[0] for z in real]))
    hz = float(np.median([z[3] - z[1] for z in real]))

    full = cv2.imread(str(image_for(page, image_dir)))
    img = cv2.resize(full, (round(sw * SCALE), round(sh * SCALE)), interpolation=cv2.INTER_AREA)
    s = lambda v: int(round(v * SCALE))

    for sb in root.iter(M + "sb"):
        z = zone(sb)
        if z:
            for (xa, ya), (xb, yb) in staff_lines(z):
                cv2.line(img, (s(xa), s(ya)), (s(xb), s(yb)), STAFF, 1, cv2.LINE_AA)

    n_flat = 0
    for neume in root.iter(M + "neume"):
        boxes = []
        for nc in neume.findall(M + "nc"):
            z = zone(nc)
            if not z:
                continue
            flat = z[3] - z[1] < 1
            box = (z[0], z[1], z[0] + wz, z[1] + hz) if flat else z[:4]
            boxes.append(box)
            color = FLAT if flat else NOTE
            cv2.rectangle(img, (s(box[0]), s(box[1])), (s(box[2]), s(box[3])), color, 3)
            if flat:
                n_flat += 1
                cv2.line(img, (s(z[0]), s(z[1])), (s(z[2]), s(z[1])), color, 1)
            liq = "~" if nc.find(M + "liquescent") is not None else ""
            draw_label(img, f"{nc.get('pname', '?')}{nc.get('oct', '?')}{liq}", s(box[0]), s(box[1]) - 4, color)
        if len(boxes) > 1:
            cv2.rectangle(img, (s(min(b[0] for b in boxes)) - 4, s(min(b[1] for b in boxes)) - 4),
                          (s(max(b[2] for b in boxes)) + 4, s(max(b[3] for b in boxes)) + 4), NEUME, 1)

    for clef in root.iter(M + "clef"):
        z = zone(clef)
        if z:
            cv2.rectangle(img, (s(z[0]), s(z[1])), (s(z[2]), s(z[3])), CLEF, 3)
            draw_label(img, f"{clef.get('shape', '?')}{clef.get('line', '?')}", s(z[0]), s(z[3]) + 16, CLEF)
    for cu in root.iter(M + "custos"):
        z = zone(cu)
        if z:
            box = (z[0], z[1], z[0] + wz, z[1] + hz) if z[3] - z[1] < 1 else z[:4]
            cv2.rectangle(img, (s(box[0]), s(box[1])), (s(box[2]), s(box[3])), CUSTOS, 3)

    out = out_dir / f"{mei_path.stem}_groundtruth_viz.png"
    cv2.imwrite(str(out), img)
    print(f"page {page:3d}: {len(ncs)} notes, {n_flat} zero-height -> {out.relative_to(HERE)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("pages", nargs="*", type=int, help="page numbers (default: all in mei/)")
    args = ap.parse_args()
    out_dir = MS73 / "groundtruth_viz"
    out_dir.mkdir(exist_ok=True)
    for p in sorted((MS73 / "mei").glob("*.mei")):
        if not args.pages or int(p.stem.rsplit("-", 1)[1]) in args.pages:
            render(p, args.image_dir, out_dir)


if __name__ == "__main__":
    main()
