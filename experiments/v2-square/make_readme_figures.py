"""The two figures in README.md (figures/readme/), from a real test page:

  input_crops.png      one note on its page: its box, the staff-based crop and the
                       staff-free crop, and the two 128 x 32 inputs they become
  staff_dependence.png the same note under a correct staff and a staff detected 2
                       steps too high: the staff-based input moves with the staff,
                       the staff-free input doesn't use it

    python make_readme_figures.py
"""
import collections
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dataset"))
from build_crops import BBOX_HALF_UNITS, crop_bounds, cut, staff_span  # noqa: E402

from data import batch_input  # noqa: E402
from train import CKPT, build_model, device, predict_steps  # noqa: E402

DATA = HERE.parents[1] / "data"
OUT = HERE / "figures" / "readme"
FONT = next((f for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
                         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if Path(f).exists()), None)
UP = 3                      # display upscale of 128 x 32 inputs
RED, BLUE, GREEN, ORANGE = (40, 40, 220), (190, 90, 30), (40, 150, 40), (0, 140, 255)
STAFF_MODEL, FREE_MODEL = "cls_aug", "cls_aug_bbox128_shift15"


def font(size):
    return ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()


def text(img, lines, xy, size=17, color=(30, 30, 30)):
    """Draw lines of text on a BGR array (PIL underneath)."""
    pil = Image.fromarray(img[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    x, y = xy
    for k, s in enumerate(lines):
        d.text((x, y + k * int(size * 1.35)), s, fill=color[::-1], font=font(size))
    img[:] = np.array(pil)[:, :, ::-1]


def canvas(h, w):
    return np.full((h, w, 3), 255, np.uint8)


def paste(dst, src, x, y):
    dst[y:y + src.shape[0], x:x + src.shape[1]] = src


def window(ctx):
    """Model input (128 x 32, [0,1]) from a cached-style context crop, upscaled for display."""
    w = batch_input(torch.tensor(ctx[None]))[0, 0].numpy()
    g = (w * 255).round().astype(np.uint8)
    return cv2.cvtColor(cv2.resize(g, (32 * UP, 128 * UP), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)


def frame(img, color=(200, 200, 200), t=1):
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, t)
    return img


def test_rows():
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(HERE / "dataset" / "notes.csv")):
        if r["split"] == "test" and r["kind"] == "nc" and not r["exclude"]:
            by[(r["manuscript"], r["image"])].append(r)
    return by


def staff_crop(gray, r, page_rows, shift_steps=0.0):
    t, b, _ = crop_bounds([r])[0]
    hg = staff_span(r, 0)[2]
    return cut(gray, float(r["box_ulx"]), float(r["box_lrx"]), t + shift_steps * hg, b + shift_steps * hg)


def free_crop(gray, r, page_rows):
    unit = float(np.median([float(x["box_lry"]) - float(x["box_uly"]) for x in page_rows]))
    cy = (float(r["box_uly"]) + float(r["box_lry"])) / 2
    return cut(gray, float(r["box_ulx"]), float(r["box_lrx"]), cy - BBOX_HALF_UNITS * unit, cy + BBOX_HALF_UNITS * unit), unit



def load(name, dev):
    m = build_model(name).to(dev)
    m.load_state_dict(torch.load(CKPT / f"{name}.pt", map_location=dev))
    return m


def predict(m, crops, dev):
    return predict_steps(m, torch.tensor(np.stack(crops), device=dev), True, dev)


def pick_note(by, dev, staff_m, free_m):
    """A virga on a space of an Einsiedeln test page that both models get right, and
    that the staff model gets wrong once the staff is detected 2 steps too high."""
    for (ms, image), page_rows in sorted(by.items()):
        if ms != "einsiedeln":
            continue
        gray = cv2.imread(str(DATA / "images" / ms / image), cv2.IMREAD_GRAYSCALE)
        cands = [r for r in page_rows if r["neume_size"] == "1" and r["tilt"] == "s" and r["step"] in ("3", "5")]
        if len(cands) < 5:
            continue
        clean = predict(staff_m, [staff_crop(gray, r, page_rows) for r in cands], dev)
        moved = predict(staff_m, [staff_crop(gray, r, page_rows, -2.0) for r in cands], dev)
        free = predict(free_m, [free_crop(gray, r, page_rows)[0] for r in cands], dev)
        for r, c, mv, f in zip(cands, clean, moved, free):
            s = int(r["step"])
            if c == s and f == s and mv != s:
                return gray, r, page_rows, int(mv)
    raise RuntimeError("no suitable note found")


def fig_input(gray_bgr, gray, r, page_rows):
    x0, y0, x1, y1 = (float(r[k]) for k in ("box_ulx", "box_uly", "box_lrx", "box_lry"))
    t_s, b_s, _ = crop_bounds([r])[0]
    ctx_f, unit = free_crop(gray, r, page_rows)
    cy = (y0 + y1) / 2
    t_f, b_f = cy - BBOX_HALF_UNITS * unit, cy + BBOX_HALF_UNITS * unit
    w = x1 - x0
    X0, Y0 = int(x0 - 7 * w), int(t_f - unit)
    X1, Y1 = int(x1 + 7 * w), int(b_f + unit)
    ex = gray_bgr[max(0, Y0):Y1, max(0, X0):X1].copy()
    P = lambda x, y: (int(round(x - X0)), int(round(y - Y0)))
    th = max(2, int(unit / 6))
    cv2.rectangle(ex, P(x0 - 0.15 * w, t_f), P(x1 + 0.15 * w, b_f), GREEN, th)
    cv2.rectangle(ex, P(x0, t_s), P(x1, b_s), BLUE, th)
    cv2.rectangle(ex, P(x0, y0), P(x1, y1), RED, th)
    s = 128 * UP / ex.shape[0]
    ex = cv2.resize(ex, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    a, b = frame(window(staff_crop(gray, r, page_rows))), frame(window(ctx_f))
    W = ex.shape[1] + 2 * a.shape[1] + 140
    img = canvas(128 * UP + 175, W)
    paste(img, frame(ex), 10, 70)
    xa = ex.shape[1] + 60
    paste(img, a, xa, 70)
    paste(img, b, xa + a.shape[1] + 50, 70)
    text(img, ["One note, two ways to cut the model input"], (10, 12), 21)
    text(img, ["page detail"], (10, 44), 15, (90, 90, 90))
    text(img, ["staff-based", "input"], (xa - 6, 70 + a.shape[0] + 8), 15, BLUE)
    text(img, ["staff-free", "input"], (xa + a.shape[1] + 44, 70 + b.shape[0] + 8), 15, GREEN)
    text(img, ["red: the note's box (from the MEI)     blue: staff-based crop = the staff, plus 2 steps above and below",
               "green: staff-free crop = 10 box-heights above and below the box; no staff used"],
         (10, 70 + ex.shape[0] + 52), 14, (60, 60, 60))
    return img


def fig_dependence(gray, r, page_rows, wrong, dev, staff_m, free_m):
    s = int(r["step"])
    a = frame(window(staff_crop(gray, r, page_rows)))
    b = frame(window(staff_crop(gray, r, page_rows, -2.0)))
    ctx_f, unit = free_crop(gray, r, page_rows)
    c = frame(window(ctx_f))
    for img_ in (a, b):  # where the staff-based model expects the 4 lines
        for k in range(4):
            y = (6 + 2 - k * 2) / 10 * 128
            cv2.line(img_, (0, int(y * UP)), (12, int(y * UP)), ORANGE, 3)
    gap = 190
    img = canvas(128 * UP + 250, 40 + 3 * gap + 40)
    xs = [40, 40 + gap, 40 + 2 * gap + 40]
    for x, im in zip(xs, (a, b, c)):
        paste(img, im, x, 80)
    text(img, ["What 'depending on the staff' means"], (10, 12), 21)
    text(img, ["staff-based model"], (xs[0], 50), 15, BLUE)
    text(img, ["staff-free model"], (xs[2], 50), 15, GREEN)
    y = 80 + a.shape[0] + 10
    text(img, ["staff correct", f"predicts {s:+d}  (right)"], (xs[0] - 10, y), 15)
    text(img, ["staff detected", "2 steps too high", f"predicts {wrong:+d}  (wrong)"], (xs[1] - 10, y), 15, (40, 40, 200))
    text(img, ["no staff used", f"predicts {s:+d}  (right)"], (xs[2] - 10, y), 15)
    text(img, [f"Einsiedeln {r['page']}, a virga, label {s:+d}. Orange ticks: where the staff-based model expects the 4 staff lines.",
               "2 steps is one line spacing, so lines still fall on the ticks and the strip looks normal -- but every line is",
               "the wrong one (the real bottom line has slid out of view), and the model reads the note one line too low."],
         (10, y + 72), 13, (90, 90, 90))
    return img



def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = device()
    staff_m, free_m = load(STAFF_MODEL, dev), load(FREE_MODEL, dev)
    by = test_rows()
    gray, r, page_rows, wrong = pick_note(by, dev, staff_m, free_m)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = cv2.imread(str(DATA / "images" / "einsiedeln" / r["image"]))
    cv2.imwrite(str(OUT / "input_crops.png"), fig_input(color if color is not None else bgr, gray, r, page_rows))
    cv2.imwrite(str(OUT / "staff_dependence.png"), fig_dependence(gray, r, page_rows, wrong, dev, staff_m, free_m))
    for f in sorted(OUT.glob("*.png")):
        print(f"{f.relative_to(HERE)}: {cv2.imread(str(f)).shape[1::-1]}, {f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
