"""Exploratory: render a large, randomly-sampled set of real crops next to
several augmented variants each, paginated into multiple images, for
eyeballing whether anything about the augmented inputs looks unreasonable
(e.g. the notehead vanishing, an empty crop). Not part of the main
experiment pipeline -- a visual sanity check, run on demand.

    python visualize_many_augmentations.py
"""
from pathlib import Path

import numpy as np

from data import REAL_LABELED_PAGES, load_real_labeled_page, load_shapes, crop_from_bounds, ROOT
from augment import jitter_bounds, augment_image
from page_inputs import resolve_page_inputs
import cv2

HERE = Path(__file__).resolve().parent
N_VARIANTS = 3
N_SAMPLES = 45
ROWS_PER_PAGE = 15
SEED = 3


def main():
    shapes = load_shapes()
    rng = np.random.default_rng(SEED)

    all_rows = []
    for page in REAL_LABELED_PAGES:
        all_rows += load_real_labeled_page(page, shapes)

    idx = rng.choice(len(all_rows), size=min(N_SAMPLES, len(all_rows)), replace=False)
    sample = [all_rows[i] for i in idx]
    print(f"sampled {len(sample)} rows out of {len(all_rows)} total")

    image_by_page = {}
    for page in REAL_LABELED_PAGES:
        inputs = resolve_page_inputs(ROOT / page)
        image_by_page[page] = cv2.imread(str(inputs.image), cv2.IMREAD_GRAYSCALE)

    cell_h, cell_w = 128, 32
    pad = 6
    cols = 1 + N_VARIANTS

    n_pages = (len(sample) + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE
    for page_i in range(n_pages):
        chunk = sample[page_i * ROWS_PER_PAGE:(page_i + 1) * ROWS_PER_PAGE]
        grid = np.ones((len(chunk) * (cell_h + pad) + pad,
                         cols * (cell_w + pad) + pad), dtype=np.float32)
        labels = []
        for i, row in enumerate(chunk):
            y0 = pad + i * (cell_h + pad)
            grid[y0:y0 + cell_h, pad:pad + cell_w] = row["crop"]
            labels.append((row["page"], row["class_name"], row["box_bottom"] - row["box_top"], row["truth"]))
            image = image_by_page[row["page"]]
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

        for i, (page, cls, height, truth) in enumerate(labels):
            y0 = (pad + i * (cell_h + pad)) * scale
            text = f"{page} / {cls} / step={truth:.0f} (h={height:.0f}px)"
            cv2.putText(grid_bgr, text, (pad * scale, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 140, 255), 1, cv2.LINE_AA)
        header_labels = ["original"] + [f"aug {j+1}" for j in range(N_VARIANTS)]
        for j, text in enumerate(header_labels):
            x0 = (pad + j * (cell_w + pad)) * scale
            cv2.putText(grid_bgr, text, (x0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 200, 0), 1, cv2.LINE_AA)

        out_path = HERE / "figures" / f"many_aug_page{page_i+1}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), grid_bgr)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
