"""The figures in README.md (figures/readme/), in English:

  two_ways.png        one stacked neume (a podatus) from a test page, read the two
                      ways: one crop per note (step 1) and one crop per neume (step 2)
  neume_examples.png  the final neume model on test neumes, right and wrong
  salzinnes.png       its note-count errors on Salzinnes, the unseen manuscript
  by_type.png         test neumes entirely right by type: the final model, the model before
                      the rare-shape fixes, and step 1

    python make_readme_figures.py   # after eval_notes.py has written results/notes_predictions.npz
"""
import collections
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from paths import DATA, FIGURES, HERE, NOTES_CSV, RESULTS
from build_crops import BBOX_HALF_UNITS, CONTEXT, cut  # v2's crop rules
from neume_data import geom_of, load_neumes, model_input, predict_neumes
from neume_model import MAX_NOTES, NEUME_W
from train import device  # v2's train.py
from train_neume import CONFIGS, CROPS_CTX, FINAL, FINAL_SPACING, load_model as load_neume_model

sys.path.insert(0, str(HERE / "dataset"))
from build_neume_crops import cut_neume_ctx  # noqa: E402

OUT = FIGURES / "readme"
UP = 3
H = 128 * UP
RED, GREEN, BLUE, ORANGE, GREY = (40, 40, 210), (40, 150, 40), (200, 110, 30), (0, 120, 230), (110, 110, 110)
NOTE_COLORS = [(40, 40, 210), (30, 140, 30), (200, 110, 30), (160, 40, 160), (0, 150, 200)]
SINGLE, ALL_NOTES = "cls_aug_bbox128_shift15", "cls_aug_bbox128_shift15_all"
FONT = next((f for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
                         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if Path(f).exists()), None)


def text(img, lines, xy, size=17, color=(30, 30, 30)):
    """Lines of text on a BGR array (PIL underneath), as v2's make_readme_figures.py."""
    pil = Image.fromarray(img[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()
    for k, s in enumerate(lines):
        d.text((xy[0], xy[1] + k * int(size * 1.35)), s, fill=color[::-1], font=font)
    img[:] = np.array(pil)[:, :, ::-1]


def canvas(h, w):
    return np.full((h, w, 3), 255, np.uint8)


def paste(dst, src, x, y):
    dst[y:y + src.shape[0], x:x + src.shape[1]] = src


def frame(img, color=(200, 200, 200), t=1):
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, t)
    return img


def show(window, up=UP, color=(200, 200, 200)):
    g = (np.asarray(window) * 255).round().astype(np.uint8)
    img = cv2.cvtColor(cv2.resize(g, (g.shape[1] * up, g.shape[0] * up), interpolation=cv2.INTER_NEAREST),
                       cv2.COLOR_GRAY2BGR)
    return frame(img, color)


def steps_str(steps):
    return " ".join(f"{s:+d}" for s in steps)


def test_rows():
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(NOTES_CSV)):
        if r["split"] == "test" and r["kind"] == "nc" and not r["exclude"]:
            by[(r["manuscript"], r["page"])].append(r)
    return by


def box(r):
    return tuple(float(r[k]) for k in ("box_ulx", "box_uly", "box_lrx", "box_lry"))


def neume_ctx(gray, notes, unit):
    b = np.array([box(r) for r in notes])
    return cut_neume_ctx(gray, b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max(), unit)


def neume_window(ctx, g, dev):
    geom = tuple(torch.tensor([v], dtype=torch.float32, device=dev) for v in g)
    return model_input(CONFIGS[FINAL], torch.tensor(ctx[None], device=dev), geom, dev=dev)


def fig_two_ways(dev):
    """A podatus whose upper note v2's single-note model reads wrong, read right by the
    all-note model and by the neume model."""
    z = np.load(RESULTS / "notes_predictions.npz", allow_pickle=True)
    by_neume = collections.defaultdict(list)
    for i, key in enumerate(zip(z["manuscript"], z["page"].astype(str), z["neume_id"])):
        by_neume[key].append(i)
    rows = test_rows()
    neume_m = load_neume_model(FINAL, dev)
    rng = random.Random(1)
    cands = [k for k, idx in sorted(by_neume.items()) if k[0] == "einsiedeln" and len(idx) == 2
             and all(z["stacked"][idx]) and z["step"][idx[1]] - z["step"][idx[0]] == 1
             and (z[f"pred_{SINGLE}"][idx] != z["step"][idx]).any()
             and (z[f"pred_{ALL_NOTES}"][idx] == z["step"][idx]).all()]
    rng.shuffle(cands)
    for key in cands:
        ms, page, nid = key
        page_rows = rows[(ms, page)]
        notes = sorted([r for r in page_rows if r["neume_id"] == nid], key=lambda r: int(r["nc_index"]))
        gray = cv2.imread(str(DATA / "images" / ms / notes[0]["image"]), cv2.IMREAD_GRAYSCALE)
        unit = float(np.median([box(r)[3] - box(r)[1] for r in page_rows]))
        ctx, *g, _ = neume_ctx(gray, notes, unit)
        win = neume_window(ctx, g, dev)
        with torch.no_grad():
            cls = neume_m(win).argmax(2)[0].cpu().numpy()
        labels = [int(r["step"]) for r in notes]
        pred = [int(c) - 3 for c in cls[:2]] if (cls[2] == 13) else None
        if pred == labels:
            break
    else:
        raise RuntimeError("no suitable podatus")
    idx = by_neume[key]
    color = cv2.imread(str(DATA / "images" / ms / notes[0]["image"]))
    # page detail, one model window tall
    b = np.array([box(r) for r in notes])
    cy = (b[:, 1].min() + b[:, 3].max()) / 2
    X0, X1 = int(b[:, 0].min() - 4 * unit), int(b[:, 2].max() + 4 * unit)
    Y0, Y1 = int(cy - BBOX_HALF_UNITS * unit), int(cy + BBOX_HALF_UNITS * unit)
    det = color[max(0, Y0):Y1, max(0, X0):X1].copy()
    s = H / det.shape[0]
    det = cv2.resize(det, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    P = lambda x, y: (int(round((x - X0) * s)), int(round((y - Y0) * s)))
    for k, bb in enumerate(b):
        cv2.rectangle(det, P(bb[0], bb[1]), P(bb[2], bb[3]), NOTE_COLORS[k], 2)
    cv2.rectangle(det, P(b[:, 0].min() - 2, b[:, 1].min() - 2), P(b[:, 2].max() + 2, b[:, 3].max() + 2), ORANGE, 2)
    det = frame(det)
    # step 1: one crop per note, centred on its own box
    note_wins = []
    for r in notes:
        x0, y0, x1, y1 = box(r)
        c = (y0 + y1) / 2
        note_crop = cut(gray, x0, x1, c - BBOX_HALF_UNITS * unit, c + BBOX_HALF_UNITS * unit, round(128 * CONTEXT))
        mid = (note_crop.shape[0] - 128) // 2
        note_wins.append(show(note_crop[mid:mid + 128] / 255.0))
    for k, w in enumerate(note_wins):
        for x in (0, w.shape[1] - 12):
            cv2.line(w, (x, H // 2), (x + 11, H // 2), NOTE_COLORS[k], 3)
    neume_win = show(win[0, 0].cpu().numpy(), color=ORANGE)

    gap = 60
    W = det.shape[1] + 2 * 96 + neume_win.shape[1] + 4 * gap + 330
    img = canvas(H + 250, W)
    y = 80
    paste(img, det, 10, y)
    xa = det.shape[1] + 10 + gap
    for k, w in enumerate(note_wins):
        paste(img, w, xa + k * (96 + 30), y)
    xb = xa + 2 * 96 + 30 + gap + 20
    paste(img, neume_win, xb, y)
    text(img, [f"A podatus (Einsiedeln {page}): two notes stacked in one column, read two ways"], (10, 12), 21)
    text(img, ["page; boxes from the MEI"], (10, 50), 15, GREY)
    text(img, ["step 1: one crop per note,", "centred on its own box"], (xa, 36), 15, (30, 30, 30))
    text(img, ["step 2: one crop", "for the whole neume"], (xb, 36), 15, ORANGE)
    yt = y + H + 10
    for k, r in enumerate(notes):
        i = idx[k]
        ps, pa = int(z[f"pred_{SINGLE}"][i]), int(z[f"pred_{ALL_NOTES}"][i])
        text(img, [f"note {k + 1}, label {labels[k]:+d}"], (xa + k * 126, yt), 14, NOTE_COLORS[k])
        text(img, [f"v2: {ps:+d}"], (xa + k * 126, yt + 22), 14, GREEN if ps == labels[k] else RED)
        text(img, [f"all notes: {pa:+d}"], (xa + k * 126, yt + 44), 14, GREEN if pa == labels[k] else RED)
    xs = xb + neume_win.shape[1] + 24
    text(img, ["output slots"], (xs, y), 15, GREY)
    for k in range(4):
        lab = f"{labels[k]:+d}" if k < 2 else "none"
        text(img, [f"slot {k + 1}: {lab}" + ("" if k < 3 else "  ...")], (xs, y + 30 + 26 * k), 15,
             NOTE_COLORS[k] if k < 2 else GREY)
    text(img, ["all right"], (xs, y + 30 + 26 * 5), 15, GREEN)
    text(img, ["Step 1: the two crops are the same strip of page, shifted by one step -- only the box's height tells the notes",
               "apart. v2's model, trained on single notes with +-1.5-step box jitter, ignores it; trained on every note it learns to use it.",
               "Step 2: one crop, the notes in reading order (a podatus: lower note first), then 'none'. Staff lines are never given."],
         (10, yt + 80), 14, (60, 60, 60))
    return img


def fig_examples(dev, n_right=6, n_wrong=2):
    d = load_neumes("test", crops_dir=CROPS_CTX)
    m = load_neume_model(FINAL, dev)
    steps, n = predict_neumes(m, torch.tensor(d["crops"], device=dev), dev, CONFIGS[FINAL], geom_of(d, dev))
    ok = (n == d["n_notes"]) & np.all(steps == np.where(d["steps"] > -100, d["steps"], -100), 1)
    rng = np.random.default_rng(3)
    picks = []
    for k in (2, 3, 4, 5, 2, 3):  # right, by size
        picks += list(rng.choice(np.where(ok & (d["n_notes"] == k))[0], 1))
    picks = picks[:n_right]
    picks += list(rng.choice(np.where(~ok & (n != d["n_notes"]))[0], 1))  # a wrong note count
    picks += list(rng.choice(np.where(~ok & (n == d["n_notes"]))[0], n_wrong - 1))  # a wrong step
    up = 2
    tiles = []
    for i in picks:
        win = model_input(CONFIGS[FINAL], torch.tensor(d["crops"][[i]], device=dev),
                          tuple(torch.tensor(d[k][[i]], dtype=torch.float32, device=dev) for k in ("box_l", "box_r", "unit_px")),
                          dev=dev)[0, 0].cpu().numpy()
        t = canvas(128 * up + 95, NEUME_W * up + 120)
        paste(t, show(win, up, ORANGE), 10, 10)
        lab = [int(s) for s in d["steps"][i][:d["n_notes"][i]]]
        pr = [int(s) for s in steps[i][:n[i]]]
        ms = "Einsiedeln" if d["manuscript"][i] == "einsiedeln" else "MS73"
        text(t, [ms, f"{d['n_notes'][i]} notes"], (NEUME_W * up + 18, 14), 13, GREY)
        text(t, [f"label {steps_str(lab)}"], (10, 128 * up + 18), 14, (30, 30, 30))
        text(t, [f"model {steps_str(pr)}"], (10, 128 * up + 40), 14, GREEN if ok[i] else RED)
        text(t, ["right" if ok[i] else "wrong"], (10, 128 * up + 62), 14, GREEN if ok[i] else RED)
        tiles.append(t)
    row1, row2 = np.hstack(tiles[:4]), np.hstack(tiles[4:8])
    img = canvas(60, row1.shape[1])
    text(img, ["The neume model on test neumes: model input (the neume's box, +-10 units), label and output, bottom line = 0"],
         (10, 16), 17)
    return np.vstack([img, row1, row2])


def fig_salzinnes(dev, k=5):
    import eval_salzinnes as E
    neumes, _ = E.load_pages()
    d = {key: np.array([x[key] for x in neumes]) for key in ("steps", "n_notes", "box_l", "box_r", "unit_px")}
    m = load_neume_model(FINAL, dev)
    ctx = torch.tensor(np.stack([x["ctx"] for x in neumes]), device=dev)
    steps, n = predict_neumes(m, ctx, dev, CONFIGS[FINAL], geom_of(d, dev))
    wrong = np.where((n != d["n_notes"]) & (d["n_notes"] > 1))[0]
    picks = list(np.random.default_rng(0).choice(wrong, k, replace=False))
    up = 2
    tiles = []
    for i in picks:
        win = model_input(CONFIGS[FINAL], ctx[[i]], tuple(v[[i]] for v in geom_of(d, dev)), dev=dev)[0, 0].cpu().numpy()
        t = canvas(128 * up + 60, NEUME_W * up + 20)
        paste(t, show(win, up, ORANGE), 10, 10)
        text(t, [f"{d['n_notes'][i]} notes, model {n[i]}"], (10, 128 * up + 18), 14, RED)
        tiles.append(t)
    row = np.hstack(tiles)
    img = canvas(84, row.shape[1])
    text(img, [f"Salzinnes (unseen): wrong note counts of the model '{FINAL}'.",
               "Its notes sit further apart than in the training manuscripts."], (10, 12), 16)
    return np.vstack([img, row])


def fig_by_type(dev):
    """Matplotlib bar chart (saved directly); returns its path."""
    import matplotlib.pyplot as plt
    import show_accuracy_by_type as A
    from eval_neume import step1_notes_right
    d = load_neumes("test", crops_dir=CROPS_CTX)
    t = A.pool(A.types(d), d["n_notes"], A.MIN_N)
    labels, counts = A.order(t, d["n_notes"], 1)
    has = np.arange(MAX_NOTES)[None] < d["n_notes"][:, None]
    s1 = np.all(step1_notes_right(d)[A.STEP1] == has, 1)
    pct = lambda v: [100 * float(np.mean(v[t == l])) for l in labels]
    series = [("final model", pct(A.scores(d, *A.predict(FINAL, d, dev))[2]), A.C1),
              ("before the rare-shape fixes", pct(A.scores(d, *A.predict(FINAL_SPACING, d, dev))[2]), A.C3),
              ("one crop per note (step 1)", pct(s1), "#bfc3c9")]
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(labels) + 1.8))
    A.hbars(ax, labels, series, counts, (0, 106), "Test set: neumes entirely right, by type (n = test neumes)")
    fig.tight_layout()
    fig.savefig(OUT / "by_type.png", dpi=110)
    plt.close(fig)
    return OUT / "by_type.png"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = device()
    for name, fig in (("two_ways", fig_two_ways), ("neume_examples", fig_examples), ("salzinnes", fig_salzinnes)):
        img = fig(dev)
        cv2.imwrite(str(OUT / f"{name}.png"), img)
        print(f"figures/readme/{name}.png: {img.shape[1]} x {img.shape[0]}")
    print(f"figures/readme/{fig_by_type(dev).name}")


if __name__ == "__main__":
    main()
