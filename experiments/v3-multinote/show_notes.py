"""Figure (step 1): what a staff-free note model sees and outputs for the notes
of multi-note neumes. Each note is cut exactly as v2 cuts a
single-note neume (v2's build_crops.py --mode bbox): its own MEI box, 10
page-median box heights above and below, no staff used. By default the model is
v2's final one, never trained on multi-note neumes; --model picks another (e.g.
cls_aug_bbox128_all). Neumes are drawn at random (fixed seed) from the test
pages, one per (manuscript, contour) in PICKS; it is not an evaluation.

  figures/notes_io_<model>.png

    python show_notes.py [--model cls_aug_bbox128_shift15]
"""
import argparse
import collections
import csv
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from paths import DATA, FIGURES as OUT, NOTES_CSV
from build_crops import BBOX_HALF_UNITS, cut, staff_span  # v2's crop rules
from data import batch_input  # v2's data.py
from model import STEP_MAX, STEP_MIN  # v2's model.py
from train import device  # v2's train.py
from train_notes import load_model
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
MODEL = "cls_aug_bbox128_shift15"
SEED = 0
UP = 3                       # display upscale of the 128 x 32 input
H = 128 * UP
BAR_W = 175
PICKS = [("einsiedeln", (0, -1), "clivis"), ("einsiedeln", (0, 1), "podatus"),
         ("einsiedeln", (0, 1, 0), "torculus"), ("einsiedeln", (0, -1, -2), "climacus"),
         ("ms73", (0, 1), "podatus"), ("ms73", (0, -1, 0), "porrectus")]
NOTE_COLORS = [(40, 40, 220), (30, 140, 30), (210, 110, 20), (160, 40, 160)]  # BGR, by note order
STAFF = (200, 160, 120)
GREEN, RED, GREY = (40, 150, 40), (40, 40, 210), (175, 175, 175)


def font(size):
    return ImageFont.truetype(FONT, size)


def text(img, lines, xy, size=16, color=(30, 30, 30)):
    pil = Image.fromarray(img[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    for k, s in enumerate(lines):
        d.text((xy[0], xy[1] + k * int(size * 1.4)), s, fill=color[::-1], font=font(size))
    img[:] = np.array(pil)[:, :, ::-1]


def canvas(h, w):
    return np.full((h, w, 3), 255, np.uint8)


def frame(img, color=(200, 200, 200)):
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, 1)
    return img


def box(r):
    return tuple(float(r[k]) for k in ("box_ulx", "box_uly", "box_lrx", "box_lry"))


def test_neumes():
    """{(manuscript, image): (page rows, {neume_id: notes in order})} over the test pages."""
    pages = collections.defaultdict(list)
    for r in csv.DictReader(open(NOTES_CSV)):
        if r["split"] == "test" and r["kind"] == "nc" and not r["exclude"]:
            pages[(r["manuscript"], r["image"])].append(r)
    out = {}
    for key, rows in pages.items():
        by = collections.defaultdict(list)
        for r in rows:
            by[r["neume_id"]].append(r)
        out[key] = rows, {n: sorted(v, key=lambda r: int(r["nc_index"])) for n, v in by.items()}
    return out


def pick(neumes, ms, contour, rng):
    cands = [(key, notes) for key, (_, by) in sorted(neumes.items()) if key[0] == ms
             for _, notes in sorted(by.items())
             if len(notes) == len(contour) and int(notes[0]["neume_size"]) == len(contour)
             and tuple(int(n["step"]) - int(notes[0]["step"]) for n in notes) == contour]
    return rng.choice(cands) if cands else None


def detail(page_bgr, notes, unit):
    """The page around the neume, as tall as one model window: MEI staff lines
    (for reading the labels only) and each note's box, numbered in order."""
    bx = [box(r) for r in notes]
    x0, x1 = min(b[0] for b in bx) - 4 * unit, max(b[2] for b in bx) + 4 * unit
    cy = (min(b[1] for b in bx) + max(b[3] for b in bx)) / 2
    y0, y1 = cy - BBOX_HALF_UNITS * unit, cy + BBOX_HALF_UNITS * unit
    X0, Y0, X1, Y1 = (int(round(v)) for v in (x0, y0, x1, y1))
    Hh, W = page_bgr.shape[:2]
    crop = cv2.copyMakeBorder(page_bgr[max(0, Y0):min(Hh, Y1), max(0, X0):min(W, X1)],
                              max(0, -Y0), max(0, Y1 - Hh), max(0, -X0), max(0, X1 - W),
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))
    s = H / crop.shape[0]
    img = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    P = lambda x, y: (int(round((x - X0) * s)), int(round((y - Y0) * s)))
    r = notes[0]
    ta, ba, _ = staff_span(r, x0)
    tb, bb, _ = staff_span(r, x1)
    for k in range(4):
        cv2.line(img, P(x0, ta + k * (ba - ta) / 3), P(x1, tb + k * (bb - tb) / 3), STAFF, 1, cv2.LINE_AA)
    for k, b in enumerate(bx):
        c = NOTE_COLORS[k % len(NOTE_COLORS)]
        cv2.rectangle(img, P(b[0], b[1]), P(b[2], b[3]), c, 2)
    for k, b in enumerate(bx):
        c = NOTE_COLORS[k % len(NOTE_COLORS)]
        px, py = P(b[0], b[1])
        text(img, [str(k + 1)], (px - 13, py - 20), 15, c)
    return frame(img)


def model_view(window, color):
    """128 x 32 model input, upscaled; ticks mark its centre row (= the note's box centre)."""
    g = (window * 255).round().astype(np.uint8)
    img = cv2.cvtColor(cv2.resize(g, (32 * UP, H), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    for x in (0, img.shape[1] - 12):
        cv2.line(img, (x, H // 2), (x + 11, H // 2), color, 3)
    return frame(img, color)


def bars(prob, label):
    """The 13 output probabilities, highest step on top; the argmax is the prediction."""
    img = canvas(H, BAR_W)
    steps = list(range(STEP_MAX, STEP_MIN - 1, -1))
    rh = H / len(steps)
    pred = int(np.argmax(prob)) + STEP_MIN
    for k, st in enumerate(steps):
        p = float(prob[st - STEP_MIN])
        y = int(k * rh)
        color = (GREEN if pred == label else RED) if st == pred else GREY
        cv2.rectangle(img, (38, y + 5), (38 + max(1, int(p * 90)), int(y + rh) - 5), color, -1)
    for k, st in enumerate(steps):
        p = float(prob[st - STEP_MIN])
        y = int(k * rh)
        text(img, [f"{st:+d}"], (4, y + 4), 14, GREEN if st == label else (90, 90, 90))
        if p >= 0.05:
            text(img, [f"{p:.0%}"], (38 + int(p * 90) + 4, y + 4), 12, (60, 60, 60))
    return img


def neume_row(page_bgr, page_rows, notes, name, model, dev):
    unit = float(np.median([box(r)[3] - box(r)[1] for r in page_rows]))
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    ctx = []
    for r in notes:
        x0, y0, x1, y1 = box(r)
        cy = (y0 + y1) / 2
        ctx.append(cut(gray, x0, x1, cy - BBOX_HALF_UNITS * unit, cy + BBOX_HALF_UNITS * unit))
    x = batch_input(torch.tensor(np.stack(ctx)), device=dev)
    with torch.no_grad():
        prob = torch.softmax(model(x), 1).cpu().numpy()
    windows = x[:, 0].cpu().numpy()

    det = detail(page_bgr, notes, unit)
    unit_w = 32 * UP + 12 + BAR_W + 36
    row = canvas(H + 112, det.shape[1] + 40 + unit_w * len(notes))
    row[46:46 + H, :det.shape[1]] = det
    contour = ", ".join(f"{int(n['step']) - int(notes[0]['step']):+d}" if k else "0" for k, n in enumerate(notes))
    lig = ", ligated" if any(n["ligated"] == "1" for n in notes) else ""
    ms = {"einsiedeln": "Einsiedeln", "ms73": "MS73"}[notes[0]["manuscript"]]
    text(row, [f"{ms} {notes[0]['page']} · {name} (steps from the first note: {contour}){lig}"], (0, 6), 18)
    text(row, ["page detail"], (0, 46 + H + 6), 14, (110, 110, 110))
    n_ok = 0
    for k, (r, w, p) in enumerate(zip(notes, windows, prob)):
        c = NOTE_COLORS[k % len(NOTE_COLORS)]
        label, pred = int(r["step"]), int(np.argmax(p)) + STEP_MIN
        n_ok += pred == label
        x = det.shape[1] + 40 + k * unit_w
        mv = model_view(w, c)
        row[46:46 + H, x:x + mv.shape[1]] = mv
        row[46:46 + H, x + mv.shape[1] + 12:x + mv.shape[1] + 12 + BAR_W] = bars(p, label)
        verdict = ("right", GREEN) if pred == label else ("wrong", RED)
        text(row, [f"note {k + 1}: input"], (x, 46 + H + 6), 14, c)
        text(row, [f"label {label:+d}  model {pred:+d}  {verdict[0]}"], (x, 46 + H + 30), 15, verdict[1])
    return row, n_ok


def header(width):
    img = canvas(150, width)
    text(img, ["Multi-note neumes, one crop per note: input and output of a staff-free note model"], (10, 10), 22)
    text(img, ["Input: one 128 x 32 grey crop per note, centred on the note's own MEI box, 10 units above and below (a unit is the",
               "          page's median note-box height, about one step), as wide as the box. No staff lines. Ticks mark the crop's centre row.",
               "Output: probabilities over 13 positions (-3 to +9; 0 = bottom line, +1 per line or space up); the largest is the prediction.",
               "Green: the label. The light blue lines in the page detail are the MEI staff, for reading the labels only."], (10, 48), 15, (70, 70, 70))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = device()
    model, _ = load_model(args.model, dev)
    neumes = test_neumes()
    rng = random.Random(SEED)
    rows, total, ok = [], 0, 0
    for ms, contour, name in PICKS:
        got = pick(neumes, ms, contour, rng)
        if got is None:
            print(f"no {ms} neume with contour {contour}")
            continue
        (ms_, image), notes = got
        page_bgr = cv2.imread(str(DATA / "images" / ms_ / image))
        row, n_ok = neume_row(page_bgr, neumes[(ms_, image)][0], notes, name, model, dev)
        rows.append(row)
        total, ok = total + len(notes), ok + n_ok
        print(f"{ms_} {notes[0]['page']} {name}: {n_ok}/{len(notes)}")
    width = max(r.shape[1] for r in rows) + 20
    parts = [header(width)] + [np.pad(r, ((0, 20), (10, width - r.shape[1] - 10), (0, 0)), constant_values=255)
                               for r in rows]
    out = OUT / f"notes_io_{args.model}.png"
    cv2.imwrite(str(out), np.vstack(parts))
    print(f"{ok}/{total} notes right; wrote {out}")


if __name__ == "__main__":
    main()
