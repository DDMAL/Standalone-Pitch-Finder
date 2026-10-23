"""Figures explaining the model input (labels in Chinese, for the write-up):

  figures/input_pipeline_<manuscript>.png  one note, from the page to the model input:
      (1) the page around it: MEI staff lines, the note's MEI box, the crop
          region (top line - 2 steps .. bottom line + 2 steps, the box's width)
          and the cached context region (1.5x taller)
      (2) the cached context crop, central model window marked
      (3) the model input with its staff lines and the label's row
      (4) six training-time augmentations of it
  figures/inputs_by_step.png  model inputs grouped by label (stave step)

    python show_inputs.py
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import data as D
from model import IMG_H, IMG_W

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
CTX_H = int(IMG_H * D.CONTEXT)
MARGIN = 2.0  # steps above the top line / below the bottom line
UP = 4        # display upscale of crops


def font(size):
    return ImageFont.truetype(FONT, size)


def step_row(step, h=IMG_H):
    return (6 + MARGIN - step) / (6 + 2 * MARGIN) * h


def text_block(lines, width, size=22, pad=8):
    img = Image.new("RGB", (width, pad * 2 + len(lines) * int(size * 1.45)), "white")
    d = ImageDraw.Draw(img)
    for k, t in enumerate(lines):
        d.text((pad, pad + k * int(size * 1.45)), t, fill=(30, 30, 30), font=font(size))
    return np.array(img)[:, :, ::-1]


def titled(panel, title, size=24):
    """Title above the panel; a panel narrower than its title is padded (white) to fit it."""
    need = int(font(size).getlength(title)) + 20
    if panel.shape[1] < need:
        panel = np.pad(panel, ((0, 0), (0, need - panel.shape[1]), (0, 0)), constant_values=255)
    head = text_block([title], panel.shape[1], size)
    return np.vstack([head, panel])


def hstack_pad(panels, gap=24):
    h = max(p.shape[0] for p in panels)
    out = []
    for p in panels:
        out.append(np.pad(p, ((0, h - p.shape[0]), (0, gap), (0, 0)), constant_values=255))
    return np.hstack(out)


def upscale(gray):
    return cv2.cvtColor(cv2.resize(gray, (gray.shape[1] * UP, gray.shape[0] * UP), interpolation=cv2.INTER_NEAREST),
                        cv2.COLOR_GRAY2BGR)


def page_panel(row, top, bottom, height=760):
    img = cv2.imread(str(D.ROOT / "data" / "images" / row["manuscript"] / row["image"]))
    x0, y0, x1, y1 = (float(row[k]) for k in ("box_ulx", "box_uly", "box_lrx", "box_lry"))
    ulx, lrx = float(row["staff_ulx"]), float(row["staff_lrx"])
    c, h = (top + bottom) / 2, (bottom - top) * D.CONTEXT
    w = x1 - x0
    X0, X1 = int(max(0, x0 - 7 * w)), int(min(img.shape[1], x1 + 7 * w))
    Y0, Y1 = int(max(0, c - h / 2 - 0.1 * h)), int(min(img.shape[0], c + h / 2 + 0.1 * h))
    s = height / (Y1 - Y0)
    reg = cv2.resize(img[Y0:Y1, X0:X1], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    P = lambda x, y: (int(round((x - X0) * s)), int(round((y - Y0) * s)))
    t = np.tan(np.radians(float(row["staff_rotate"])))
    H = (float(row["staff_lry"]) - float(row["staff_uly"])) - (lrx - ulx) * abs(t)
    xs0 = ulx if t >= 0 else lrx
    for k in range(4):  # MEI staff lines
        ya = float(row["staff_uly"]) + (X0 - xs0) * t + k * H / 3
        yb = float(row["staff_uly"]) + (X1 - xs0) * t + k * H / 3
        cv2.line(reg, P(X0, ya), P(X1, yb), (255, 90, 0), 2, cv2.LINE_AA)
    cv2.rectangle(reg, P(x0, c - h / 2), P(x1, c + h / 2), (0, 200, 230), 2)   # context region
    cv2.rectangle(reg, P(x0, top), P(x1, bottom), (0, 170, 0), 3)               # crop region
    cv2.rectangle(reg, P(x0, y0), P(x1, y1), (0, 0, 255), 2)                    # MEI note box
    return reg


def pipeline_figure(ms, page, note_id, d, idx, rows):
    row = rows[(ms, str(page), str(note_id))]
    top, bottom = float(d["top"][idx]), float(d["bottom"][idx])
    ctx = d["crops"][idx]
    p1 = page_panel(row, top, bottom)

    p2 = upscale(ctx)
    off = (CTX_H - IMG_H) // 2
    cv2.rectangle(p2, (0, off * UP), (IMG_W * UP - 1, (off + IMG_H) * UP - 1), (0, 170, 0), 3)

    win = D.batch_input(torch.tensor(ctx[None]))[0, 0].numpy()
    p3 = upscale((win * 255).astype(np.uint8))
    for s in (0, 2, 4, 6):
        y = int(step_row(s) * UP)
        cv2.line(p3, (0, y), (14, y), (255, 90, 0), 3)
        cv2.line(p3, (IMG_W * UP - 15, y), (IMG_W * UP - 1, y), (255, 90, 0), 3)
    y = int(step_row(int(d["step"][idx])) * UP)
    cv2.line(p3, (0, y), (24, y), (0, 0, 255), 4)

    aug = D.batch_input(torch.tensor(np.repeat(ctx[None], 6, 0)), rng=np.random.default_rng(3))[:, 0].numpy()
    p4 = np.hstack([np.pad(upscale((a * 255).astype(np.uint8)), ((0, 0), (0, 10), (0, 0)), constant_values=255)
                    for a in aug])

    step, pn = int(d["step"][idx]), f"{d['pname'][idx]}{d['oct'][idx]}"
    name = "Einsiedeln" if ms == "einsiedeln" else "MS73"
    fig = hstack_pad([titled(p1, "① 页面局部"), titled(p2, "② 缓存的上下文图"),
                      titled(p3, "③ 模型输入"), titled(p4, "④ 训练时的随机增强（6 个例子）")])
    caption = text_block([
        f"{name} 第 {page} 页，一个单音符 neume（{pn}），标签 step = {step:+d}（最下面那条线 = 0，往上每条线 +2，线和线之间的间隔是奇数）",
        "① 红框 = MEI 里这个音符的框；蓝线 = MEI 的 4 条谱线；绿框 = 切图范围：上下从最上线往上 2 步到最下线往下 2 步，左右就是音符框的宽度；"
        "黄框 = 实际缓存的范围（绿框的 1.5 倍高）",
        "② 黄框那一块缩放成 192×32 存下来；绿框是不做增强时模型看到的部分",
        "③ 模型输入 128×32：整个谱表 + 上下各 2 步。蓝色刻度 = 4 条谱线应在的位置（每张图都一样）；红色刻度 = 标签对应的位置，应正好落在符头上",
        "④ 训练时随机把切图范围拉高/压扁 ±20%、上下平移 ±15%、旋转 ±4°，再加亮度和噪声变化，模拟谱线检测不准的情况；标签不变",
    ], fig.shape[1], size=21)
    out = OUT / f"input_pipeline_{ms}.png"
    cv2.imwrite(str(out), np.vstack([fig, caption]))
    return out


def by_step_figure(d, per_row=14):
    rng = np.random.default_rng(0)
    rows_img = []
    for step in range(-2, 9):
        idx = np.flatnonzero((d["step"] == step) & (d["refined"] == 0))
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, min(per_row, len(idx)), replace=False)
        wins = D.batch_input(torch.tensor(d["crops"][pick]))[:, 0].numpy()
        tiles = []
        for a, i in zip(wins, pick):
            t = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            t = cv2.resize(t, (IMG_W * 2, IMG_H * 2), interpolation=cv2.INTER_NEAREST)
            y = int(step_row(step) * 2)
            cv2.line(t, (0, y), (10, y), (0, 0, 255), 3)
            color = (0, 130, 0) if d["manuscript"][i] == "einsiedeln" else (150, 0, 150)
            cv2.rectangle(t, (0, 0), (t.shape[1] - 1, t.shape[0] - 1), color, 2)
            tiles.append(np.pad(t, ((3, 3), (3, 3), (0, 0)), constant_values=255))
        tiles += [np.full_like(tiles[0], 255)] * (per_row - len(tiles))
        label = text_block([f"step {step:+d}", "线" if step % 2 == 0 else "间"], 110, size=22)
        label = np.pad(label, ((0, tiles[0].shape[0] - label.shape[0]), (0, 0), (0, 0)), constant_values=255)
        rows_img.append(np.hstack([label] + tiles))
    grid = np.vstack(rows_img)
    head = text_block(["模型输入样例，按标签（step）分组，每行同一个标签",
                       "红色刻度 = 标签位置；绿边 = Einsiedeln，紫边 = MS73",
                       "偶数 step 在线上，奇数 step 在两线之间"], grid.shape[1], size=22)
    out = OUT / "inputs_by_step.png"
    cv2.imwrite(str(out), np.vstack([head, grid]))
    return out


def main():
    import csv
    OUT.mkdir(exist_ok=True)
    rows = {(r["manuscript"], r["page"], r["id"]): r for r in csv.DictReader(open(HERE / "dataset" / "notes.csv"))
            if r["split"] == "val" and r["neume_size"] == "1"}
    d = D.load(("einsiedeln", "ms73"), "val")
    rng = np.random.default_rng(7)
    for ms in ("einsiedeln", "ms73"):
        cand = np.flatnonzero((d["manuscript"] == ms) & (d["refined"] == 0) & (d["step"] == 3) & (d["tilt"] == "s"))
        i = int(rng.choice(cand))
        print(pipeline_figure(ms, d["page"][i], d["id"][i], d, i, rows))
    print(by_step_figure(d))


if __name__ == "__main__":
    main()
