"""Figure: what a 1-step error in the box centre does to the staff-free model.

Each row is one test note, each column the same note with its box centre moved
by -1 / -0.5 / 0 / +0.5 / +1 step (negative = up). Every tile is the model's
actual input (cls_aug_bbox128); ticks on its left mark the true staff lines,
the green tick the true notehead (the label), the red/green bar on the right
where the model's predicted step sits on that staff. Rows are picked from a
sample of test pages: notes the 1-step shift breaks, and notes it doesn't.

    python show_box_shift.py   # -> figures/box_shift_demo.png
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
from build_crops import BBOX_HALF_UNITS, CONTEXT, cut, staff_span  # noqa: E402

from data import batch_input  # noqa: E402
from train import CKPT, build_model, device, predict_steps  # noqa: E402

DATA = HERE.parents[1] / "data"
MODEL = "cls_aug_bbox128"
SHIFTS = (-1.0, -0.5, 0.0, 0.5, 1.0)
H, W, UP = 128, 32, 3
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
N_PAGES = {"einsiedeln": 12, "ms73": 8}


def crops_for(gray, rows, page_rows, shift):
    unit = float(np.median([float(r["box_lry"]) - float(r["box_uly"]) for r in page_rows]))
    out, geo = [], []
    for r in rows:
        hg = staff_span(r, 0)[2]
        cy = (float(r["box_uly"]) + float(r["box_lry"])) / 2 + shift * hg
        top, bottom = cy - BBOX_HALF_UNITS * unit, cy + BBOX_HALF_UNITS * unit
        out.append(cut(gray, float(r["box_ulx"]), float(r["box_lrx"]), top, bottom, round(H * CONTEXT)))
        x = (float(r["box_ulx"]) + float(r["box_lrx"])) / 2
        st, sb, _ = staff_span(r, x)
        def to_row(y, top=top, bottom=bottom):  # bind this note's crop, not the loop's last one
            return (y - top) / (bottom - top) * H
        geo.append(dict(lines=[to_row(st + k * (sb - st) / 3) for k in range(4)],
                        row_of_step=lambda s, sb=sb, hg=hg, to_row=to_row: to_row(sb - s * hg)))
    return np.stack(out), geo


def tile(img, geo, truth, pred):
    t = cv2.cvtColor(cv2.resize(img, (W * UP, H * UP), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    t = np.pad(t, ((0, 0), (14, 14), (0, 0)), constant_values=255)
    for y in geo["lines"]:
        y = int(y * UP)
        if 0 <= y < t.shape[0]:
            cv2.line(t, (0, y), (10, y), (0, 140, 255), 2)  # orange
    yt = int(geo["row_of_step"](truth) * UP)
    cv2.line(t, (0, yt), (13, yt), (0, 150, 0), 4)
    yp = int(geo["row_of_step"](pred) * UP)
    color = (0, 150, 0) if pred == truth else (0, 0, 220)
    if 0 <= yp < t.shape[0]:
        cv2.line(t, (t.shape[1] - 14, yp), (t.shape[1] - 1, yp), color, 4)
    cv2.line(t, (14, H * UP // 2), (14 + W * UP, H * UP // 2), (150, 150, 150), 2)  # window centre
    return t


def label(text, width, size=17, color=(30, 30, 30), height=None):
    font = ImageFont.truetype(FONT, size)
    lines = text.split("\n")
    h = height or 8 + len(lines) * int(size * 1.4)
    im = Image.new("RGB", (width, h), "white")
    d = ImageDraw.Draw(im)
    for k, s in enumerate(lines):
        d.text((4, 4 + k * int(size * 1.4)), s, fill=color, font=font)
    return np.array(im)[:, :, ::-1]


def main():
    dev = device()
    model = build_model(MODEL).to(dev)
    model.load_state_dict(torch.load(CKPT / f"{MODEL}.pt", map_location=dev))
    by_page = collections.defaultdict(list)
    for r in csv.DictReader(open(HERE / "dataset" / "notes.csv")):
        if r["split"] == "test" and r["kind"] == "nc" and not r["exclude"]:
            by_page[(r["manuscript"], r["image"])].append(r)
    rng = np.random.default_rng(3)
    pages = []
    for ms, n in N_PAGES.items():
        keys = sorted(k for k in by_page if k[0] == ms)
        pages += [keys[i] for i in rng.choice(len(keys), n, replace=False)]

    cands = []  # (broken?, ms, row, crops per shift, geo per shift, preds per shift)
    for ms, image in pages:
        page_rows = by_page[(ms, image)]
        rows = [r for r in page_rows if r["neume_size"] == "1"]
        gray = cv2.imread(str(DATA / "images" / ms / image), cv2.IMREAD_GRAYSCALE)
        per = {}
        for s in SHIFTS:
            crops, geo = crops_for(gray, rows, page_rows, s)
            pred = predict_steps(model, torch.tensor(crops, device=dev), True, dev, out_h=H)
            per[s] = (crops, geo, pred)
        for i, r in enumerate(rows):
            truth = int(r["step"])
            if per[0.0][2][i] != truth:
                continue  # only notes the model gets right with a correct box
            broken = per[-1.0][2][i] != truth or per[1.0][2][i] != truth
            cands.append((broken, ms, r, {s: (per[s][0][i], per[s][1][i], int(per[s][2][i])) for s in SHIFTS}))
    n_ok_clean = len(cands)
    n_broken = sum(c[0] for c in cands)
    print(f"{n_ok_clean} notes right with the true box on {len(pages)} pages; {n_broken} of them wrong at a 1-step shift")

    pick = []
    for ms, nb, nk in (("einsiedeln", 3, 2), ("ms73", 2, 1)):
        b = [c for c in cands if c[0] and c[1] == ms]
        k = [c for c in cands if not c[0] and c[1] == ms]
        pick += [b[i] for i in rng.choice(len(b), min(nb, len(b)), replace=False)]
        pick += [k[i] for i in rng.choice(len(k), min(nk, len(k)), replace=False)]

    head_titles = ["框往上偏 1 步", "往上偏 0.5 步", "框正确", "往下偏 0.5 步", "框往下偏 1 步"]
    rows_img = []
    for broken, ms, r, per in pick:
        truth = int(r["step"])
        tiles = []
        for s in SHIFTS:
            img, geo, pred = per[s]
            t = tile(img, geo, truth, pred)
            ok = pred == truth
            cap = label(f"预测 {pred:+d}  {'对' if ok else '错'}", t.shape[1], 16, (0, 120, 0) if ok else (200, 0, 0), 30)
            tiles.append(np.vstack([t, cap]))
        name = "Einsiedeln " + r["page"] if ms == "einsiedeln" else "MS73 第 " + r["page"] + " 页"
        left = label(f"{name}\n{r['pname']}{r['oct']}\n标签 {truth:+d}\n" + ("偏 1 步后\n出错" if broken else "偏 1 步后\n仍然正确"),
                     150, 16, height=tiles[0].shape[0])
        rows_img.append(np.hstack([left] + [np.pad(t, ((0, 0), (8, 8), (0, 0)), constant_values=255) for t in tiles]))
    width = rows_img[0].shape[1]
    header = np.hstack([label("", 150, height=34)] + [label(h, tiles[0].shape[1] + 16, 17, height=34) for h in head_titles])
    note = label("每张图都是模型的真实输入（不用谱线，128×32，放大 3 倍）。\n"
                 "灰线 = 图的正中间，模型默认目标符头就在这里。\n"
                 "左侧橙色刻度 = 真实的 4 条谱线；左侧绿色粗线 = 正确答案（标签）的位置；\n"
                 "右侧粗线 = 模型预测的位置（绿 = 对，红 = 错）。\n"
                 "整个 test 集上：框正确 98.7%，偏 0.5 步 98.3%，偏 1 步 93.8%（这个 128 版本）。", width, 16)
    out = HERE / "figures" / "box_shift_demo.png"
    cv2.imwrite(str(out), np.vstack([note, header] + [np.pad(x, ((6, 6), (0, 0), (0, 0)), constant_values=255) for x in rows_img]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
