"""Render a grid of real crops next to a few augmented variants each, so the
calibrated jitter ranges in augment.py can be sanity-checked by eye instead
of just by their numbers.

    python visualize_augmentation.py
"""
import numpy as np

from data import load_real_labeled_page, load_shapes, crop_from_bounds, ROOT
from augment import jitter_bounds, augment_image

import sys
sys.path.insert(0, str(ROOT / "scripts"))
import cv2

N_VARIANTS = 4
EXAMPLE_PAGES = ["McGill_MS234-064", "CantusMA1537_p22", "MS025a-02", "MS025b-01"]
SEED = 7


def main():
    shapes = load_shapes()
    rng = np.random.default_rng(SEED)

    rows_per_page = {}
    for page in EXAMPLE_PAGES:
        rows = load_real_labeled_page(page, shapes)
        # a couple of examples per page, picked for a mid-sized box (skip
        # degenerate tiny/huge ones so the grid is easy to read)
        rows = sorted(rows, key=lambda r: r["box_bottom"] - r["box_top"])
        mid = len(rows) // 2
        rows_per_page[page] = rows[mid:mid + 2]

    cell_h, cell_w = 128, 32
    pad = 6
    cols = 1 + N_VARIANTS
    example_rows = [(p, r) for p, rs in rows_per_page.items() for r in rs]

    grid = np.ones((len(example_rows) * (cell_h + pad) + pad,
                     cols * (cell_w + pad) + pad), dtype=np.float32)
    labels = []
    image_by_page = {}
    from page_inputs import resolve_page_inputs
    for page in EXAMPLE_PAGES:
        inputs = resolve_page_inputs(ROOT / page)
        image_by_page[page] = cv2.imread(str(inputs.image), cv2.IMREAD_GRAYSCALE)

    for i, (page, row) in enumerate(example_rows):
        y0 = pad + i * (cell_h + pad)
        grid[y0:y0 + cell_h, pad:pad + cell_w] = row["crop"]
        labels.append((page, row["class_name"], row["box_bottom"] - row["box_top"]))
        image = image_by_page[page]
        for j in range(N_VARIANTS):
            top, bottom = jitter_bounds(row["box_top"], row["box_bottom"], rng)
            crop = crop_from_bounds(image, row["ulx"], row["ncols"], top, bottom)
            if crop is None:
                continue
            crop = augment_image(crop, rng)
            x0 = pad + (j + 1) * (cell_w + pad)
            grid[y0:y0 + cell_h, x0:x0 + cell_w] = crop

    grid_u8 = (grid * 255).clip(0, 255).astype(np.uint8)
    scale = 3
    grid_big = cv2.resize(grid_u8, (grid_u8.shape[1] * scale, grid_u8.shape[0] * scale),
                           interpolation=cv2.INTER_NEAREST)
    grid_bgr = cv2.cvtColor(grid_big, cv2.COLOR_GRAY2BGR)

    for i, (page, cls, height) in enumerate(labels):
        y0 = (pad + i * (cell_h + pad)) * scale
        text = f"{page} / {cls} (h={height:.0f}px)"
        cv2.putText(grid_bgr, text, (pad * scale, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 140, 255), 1, cv2.LINE_AA)
    header_labels = ["original"] + [f"aug {j+1}" for j in range(N_VARIANTS)]
    for j, text in enumerate(header_labels):
        x0 = (pad + j * (cell_w + pad)) * scale
        cv2.putText(grid_bgr, text, (x0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 200, 0), 1, cv2.LINE_AA)

    out_path = ROOT / "experiments" / "v1-hfngl" / "figures" / "augmentation_examples.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid_bgr)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
