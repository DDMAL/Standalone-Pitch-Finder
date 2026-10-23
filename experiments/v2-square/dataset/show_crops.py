"""Render a grid of cached crops (build_crops.py) to eyeball them before training:
the model-input window of each crop, its staff lines where the crop geometry
puts them (blue), and the row its label points at (red tick). On an
un-refined crop the tick should sit on the notehead.

    python show_crops.py [--split train] [--n 48] [--seed 0]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from build_crops import CTX_H, IMG_H, IMG_W, MARGIN_STEPS, OUT

HERE = Path(__file__).resolve().parent
SCALE = 3  # display upscale


def step_row(step):
    """Row of a stave step inside the IMG_H-tall model window (top line = step 6)."""
    steps_tall = 6 + 2 * MARGIN_STEPS
    return (6 + MARGIN_STEPS - step) / steps_tall * IMG_H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tiles = []
    for ms in ("einsiedeln", "ms73"):
        d = np.load(OUT / f"{ms}_{args.split}.npz")
        idx = rng.choice(np.flatnonzero(d["refined"] == 0), args.n // 2, replace=False)
        for i in idx:
            off = (CTX_H - IMG_H) // 2
            win = cv2.cvtColor(d["crops"][i][off:off + IMG_H], cv2.COLOR_GRAY2BGR)
            win = cv2.resize(win, (IMG_W * SCALE, IMG_H * SCALE), interpolation=cv2.INTER_NEAREST)
            for s in (0, 2, 4, 6):
                y = int(step_row(s) * SCALE)
                cv2.line(win, (0, y), (8, y), (255, 120, 0), 2)
                cv2.line(win, (IMG_W * SCALE - 9, y), (IMG_W * SCALE - 1, y), (255, 120, 0), 2)
            y = int(step_row(int(d["step"][i])) * SCALE)
            cv2.line(win, (0, y), (14, y), (0, 0, 255), 3)
            lab = np.full((34, IMG_W * SCALE, 3), 255, np.uint8)
            cv2.putText(lab, f"{d['step'][i]:+d} {d['pname'][i]}{d['oct'][i]}", (2, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(lab, "E" if ms == "einsiedeln" else "M", (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (120, 0, 120) if ms == "ms73" else (0, 110, 0), 1, cv2.LINE_AA)
            tiles.append(np.vstack([win, lab]))
    cols = 16
    rows = [np.hstack(tiles[i:i + cols] + [np.full_like(tiles[0], 255)] * (cols - len(tiles[i:i + cols])))
            for i in range(0, len(tiles), cols)]
    grid = np.vstack([np.pad(r, ((4, 4), (0, 0), (0, 0)), constant_values=255) for r in rows])
    out = HERE / "figures" / f"crop_samples_{args.split}.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
