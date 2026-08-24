"""How wrong does bad staff-finding actually make a glyph's crop box?

For each of the 5 hand-corrected pages, compares expanded_boxes() under the
corrected vs the TRUE pre-correction staff geometry, for every glyph that
gets a valid box under both. This is what augment.py's jitter ranges should
be calibrated against, instead of guessed constants.

    python measure_staff_error.py
"""
import numpy as np

from data import ROOT, MANUAL_STAFF_PAGES, staves_for
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from ic_io import parse_ic_xml
from page_inputs import resolve_page_inputs
from annotate_notecenters import expanded_boxes


def main():
    scale_ratios, shifts, per_page = [], [], {}
    for page in sorted(MANUAL_STAFF_PAGES):
        inputs = resolve_page_inputs(ROOT / page)
        glyphs = [g for g in parse_ic_xml(inputs.ic_xml) if g.state != "UNCLASSIFIED"]
        boxes_c = expanded_boxes(glyphs, staves_for(page, corrected=True))
        boxes_u = expanded_boxes(glyphs, staves_for(page, corrected=False))

        page_scale, page_shift = [], []
        for g in glyphs:
            if g.index not in boxes_c or g.index not in boxes_u:
                continue
            top_c, bottom_c, _ = boxes_c[g.index]
            top_u, bottom_u, _ = boxes_u[g.index]
            height_c = bottom_c - top_c
            if height_c <= 0:
                continue
            page_scale.append((bottom_u - top_u) / height_c - 1.0)
            page_shift.append(((top_u + bottom_u) / 2 - (top_c + bottom_c) / 2) / height_c)

        print(f"{page}: n={len(page_scale)}  "
              f"scale ratio mean={np.mean(page_scale):+.3f} std={np.std(page_scale):.3f}  "
              f"shift mean={np.mean(page_shift):+.3f} std={np.std(page_shift):.3f}")
        per_page[page] = (page_scale, page_shift)
        scale_ratios += page_scale
        shifts += page_shift

    scale_ratios, shifts = np.array(scale_ratios), np.array(shifts)
    print(f"\noverall (n={len(scale_ratios)}):")
    for name, arr in [("scale ratio", scale_ratios), ("shift (frac of height)", shifts)]:
        print(f"  {name}: mean={arr.mean():+.3f} std={arr.std():.3f}  "
              f"p10={np.percentile(arr,10):+.3f} p90={np.percentile(arr,90):+.3f}  "
              f"abs-p90={np.percentile(np.abs(arr),90):.3f}")


if __name__ == "__main__":
    main()
