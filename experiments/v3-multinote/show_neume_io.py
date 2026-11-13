"""Figure (step 2): what NeumeCNN sees and outputs. One crop
per neume (dataset/build_neume_crops.py): the union of its notes' MEI boxes, 10
units above and below, the neume's width at true scale on a 64-px canvas, no
staff used. The output is MAX_NOTES slots in note order, each a step or "none".
Neumes are drawn at random (fixed seed) from the test pages, one per
(manuscript, contour) in PICKS. Without a trained checkpoint only the targets
are drawn.

  figures/neume_io.png

    python show_neume_io.py [--model neume_shift15_hjit]
"""
import argparse
import random
import sys

import cv2
import numpy as np
import torch

from paths import CKPT, DATA, FIGURES as OUT, HERE
from build_crops import BBOX_HALF_UNITS, staff_span  # v2's crop rules
from model import STEP_MIN  # v2's model.py
from neume_data import model_input
from neume_model import MAX_NOTES, NEUME_W, NONE
from show_notes import GREEN, NOTE_COLORS, RED, STAFF, box, canvas, frame, test_neumes, text
from train import device  # v2's train.py
from train_neume import CONFIGS, CROPS_CTX, FINAL, load_model

sys.path.insert(0, str(HERE / "dataset"))
from build_neume_crops import cut_neume, cut_neume_ctx  # noqa: E402
SEED = 0
UP = 3
H = 128 * UP
NEUME_BOX = (0, 120, 230)  # BGR orange
PICKS = [("einsiedeln", (0, -1), "clivis"), ("einsiedeln", (0, 1), "podatus"),
         ("einsiedeln", (0, 1, 0), "torculus"), ("einsiedeln", (0, -1, -2), "climacus"),
         ("einsiedeln", (0, 1, 2), "scandicus"), ("einsiedeln", (0, 1, 0, -1), "pes subbipunctis"),
         ("ms73", (0, 1), "podatus"), ("ms73", (0, -1, 0), "porrectus")]


def pick(neumes, ms, contour, rng):
    cands = [(key, notes) for key, (_, by) in sorted(neumes.items()) if key[0] == ms
             for _, notes in sorted(by.items())
             if len(notes) == len(contour) and int(notes[0]["neume_size"]) == len(contour)
             and tuple(int(n["step"]) - int(notes[0]["step"]) for n in notes) == contour]
    return rng.choice(cands) if cands else None


def detail(page_bgr, notes, unit):
    """The page around the neume, as tall as the model window: MEI staff lines (for
    reading the labels only), each note's box (thin, numbered) and the neume's box."""
    bx = np.array([box(r) for r in notes])
    nx0, ny0, nx1, ny1 = bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max()
    x0, x1 = nx0 - 4 * unit, nx1 + 4 * unit
    cy = (ny0 + ny1) / 2
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
        cv2.rectangle(img, P(b[0], b[1]), P(b[2], b[3]), NOTE_COLORS[k % len(NOTE_COLORS)], 1)
    cv2.rectangle(img, P(nx0, ny0), P(nx1, ny1), NEUME_BOX, 2)
    for k, b in enumerate(bx):
        px, py = P(b[0], b[1])
        text(img, [str(k + 1)], (px - 12, py - 19), 14, NOTE_COLORS[k % len(NOTE_COLORS)])
    return frame(img)


def model_view(window):
    g = (window * 255).round().astype(np.uint8)
    img = cv2.cvtColor(cv2.resize(g, (NEUME_W * UP, H), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    return frame(img, NEUME_BOX)


def slots(labels, prob):
    """MAX_NOTES rows: slot, label, prediction (argmax and its probability)."""
    w = 250 if prob is not None else 150
    img = canvas(H, w)
    rh = H // MAX_NOTES
    text(img, ["slot   label" + ("     model" if prob is not None else "")], (6, 0), 13, (110, 110, 110))
    for k in range(MAX_NOTES):
        y = 20 + k * (H - 20) // MAX_NOTES
        lab = f"{labels[k]:+d}" if k < len(labels) else "none"
        color = NOTE_COLORS[k % len(NOTE_COLORS)] if k < len(labels) else (150, 150, 150)
        text(img, [f"{k + 1}"], (14, y), 15, color)
        text(img, [lab], (62, y), 15, color)
        if prob is not None:
            c = int(np.argmax(prob[k]))
            p = f"{c + STEP_MIN:+d}" if c != NONE else "none"
            ok = (c == NONE and k >= len(labels)) or (k < len(labels) and c != NONE and c + STEP_MIN == labels[k])
            text(img, [f"{p}  ({prob[k][c]:.0%})"], (128, y), 15, GREEN if ok else RED)
    return img


def neume_row(page_bgr, page_rows, notes, name, model, model_name, dev):
    unit = float(np.median([box(r)[3] - box(r)[1] for r in page_rows]))
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    bx = np.array([box(r) for r in notes])
    nb = (bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max())
    cfg = CONFIGS.get(model_name, CONFIGS[FINAL])  # no checkpoint: draw targets, cut as the final model
    if cfg["crops"] == CROPS_CTX:
        ctx, *g, _ = cut_neume_ctx(gray, *nb, unit)
        geom = tuple(torch.tensor([v], dtype=torch.float32, device=dev) for v in g)
    else:
        (ctx, _), geom = cut_neume(gray, *nb, unit), None
    x = model_input(cfg, torch.tensor(ctx[None], device=dev), geom, dev=dev)
    prob = None
    if model is not None:
        with torch.no_grad():
            prob = torch.softmax(model(x), 2)[0].cpu().numpy()
    labels = [int(r["step"]) for r in notes]
    det, mv, sl = detail(page_bgr, notes, unit), model_view(x[0, 0].cpu().numpy()), slots(labels, prob)
    row = canvas(H + 80, det.shape[1] + mv.shape[1] + sl.shape[1] + 90)
    row[40:40 + H, :det.shape[1]] = det
    xa = det.shape[1] + 40
    row[40:40 + H, xa:xa + mv.shape[1]] = mv
    xb = xa + mv.shape[1] + 40
    row[40:40 + H, xb:xb + sl.shape[1]] = sl
    ms = {"einsiedeln": "Einsiedeln", "ms73": "MS73"}[notes[0]["manuscript"]]
    lig = ", ligated" if any(n["ligated"] == "1" for n in notes) else ""
    text(row, [f"{ms} {notes[0]['page']} · {name}, {len(notes)} notes{lig}"], (0, 6), 18)
    text(row, ["page detail"], (0, 40 + H + 6), 14, (110, 110, 110))
    text(row, ["model input (one crop per neume)"], (xa, 40 + H + 6), 14, NEUME_BOX)
    ok = None
    if prob is not None:
        cls = prob.argmax(1)
        target = [l - STEP_MIN for l in labels] + [NONE] * (MAX_NOTES - len(labels))
        ok = bool(np.all(cls == np.array(target)))
        text(row, ["whole neume: " + ("right" if ok else "wrong")], (xb, 40 + H + 6), 15, GREEN if ok else RED)
    return row, ok


def header(width, with_model):
    img = canvas(190, width)
    title = "One crop per neume: model input and output" + ("" if with_model else " (labels only: no trained model)")
    text(img, [title], (10, 10), 22)
    text(img, ["Input: the union of the notes' boxes (orange, standing in for IC's neume box), 10 units above and below its centre;",
               "          its width at true scale, left-aligned on a white 64-px (10-unit) canvas. No staff lines.",
               "Output: 8 slots in note order, each one of 13 positions (-3 to +9; 0 = bottom line, +1 per line or space up) or 'none'.",
               "          The number of slots before the first 'none' is the number of notes.",
               "Thin boxes in the page detail: each note's own MEI box; light blue lines: the MEI staff. The model sees neither."],
         (10, 50), 15, (70, 70, 70))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=FINAL)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = device()
    model = None
    if (CKPT / f"{args.model}.pt").exists():
        model = load_model(args.model, dev)
    neumes = test_neumes()
    rng = random.Random(SEED)
    rows = []
    for ms, contour, name in PICKS:
        got = pick(neumes, ms, contour, rng)
        if got is None:
            print(f"no {ms} neume with contour {contour}")
            continue
        (ms_, image), notes = got
        page_bgr = cv2.imread(str(DATA / "images" / ms_ / image))
        row, ok = neume_row(page_bgr, neumes[(ms_, image)][0], notes, name, model, args.model, dev)
        rows.append(row)
        print(f"{ms_} {notes[0]['page']} {name}: " + ("no model" if ok is None else "right" if ok else "wrong"))
    cw = max(r.shape[1] for r in rows) + 30  # two columns of neumes
    rows = [np.pad(r, ((0, 20), (10, cw - r.shape[1] - 10), (0, 0)), constant_values=255) for r in rows]
    if len(rows) % 2:
        rows.append(canvas(*rows[-1].shape[:2]))
    grid = [np.hstack(rows[i:i + 2]) for i in range(0, len(rows), 2)]
    rh = max(g.shape[0] for g in grid)
    grid = [np.pad(g, ((0, rh - g.shape[0]), (0, 0), (0, 0)), constant_values=255) for g in grid]
    cv2.imwrite(str(OUT / "neume_io.png"), np.vstack([header(2 * cw, model is not None)] + grid))
    print(f"wrote {OUT / 'neume_io.png'}")


if __name__ == "__main__":
    main()
