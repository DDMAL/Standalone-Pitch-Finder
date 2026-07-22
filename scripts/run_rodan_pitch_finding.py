"""
CLI driver for the Rodan-style pitch finder (rodan_pitch_finder.py).

Mirrors run_pitch_finding.py's interface so the two algorithms' outputs and
debug visualizations can be produced the same way and viewed side by side.

Usage:
    python run_rodan_pitch_finding.py \\
        --image page.jpg --ic-xml ic_output/ic-session-page.xml \\
        --staff-json page_stafflines.json --output page_rodan_pitch_finding.json \\
        [--debug-viz page_rodan_pitch_finding_debug.jpg]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from ic_io import parse_ic_xml
from staff_io import load_staves
from rodan_pitch_finder import find_pitches_rodan
from viz_utils import load_scaled_image, draw_labeled_box

_COLOR_PITCH_OK = (60, 170, 60)      # green: a note with pitch computed
_COLOR_CLEF = (200, 130, 0)          # blue: clef (the pitch reference, not a note)
_COLOR_PITCHLESS = (150, 150, 150)   # grey: pitchless_symbol / not music
_COLOR_PROBLEM = (40, 40, 220)       # red: missing_clef / missing_staff


def run(image_path: Path, ic_xml_path: Path, staff_json_path: Path, output_path: Path,
        debug_viz_path: Path = None, debug_scale: float = 2.5):
    glyphs = parse_ic_xml(ic_xml_path)
    staves = load_staves(staff_json_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    results = find_pitches_rodan(glyphs, staves, image)

    page = {
        "image": str(image_path),
        "ic_xml": str(ic_xml_path),
        "staff_json": str(staff_json_path),
        "algorithm": "rodan_style",
        "glyphs": [r.to_dict() for r in results],
    }
    output_path.write_text(json.dumps(page, indent=2))

    reasons = Counter(r.reason or "pitch_ok" for r in results)
    print(f"{len(results)} glyphs processed. Breakdown: {dict(reasons)}")
    print(f"Wrote {output_path}")

    if debug_viz_path:
        _render_debug_viz(image_path, results, debug_viz_path, debug_scale)
        print(f"Wrote debug viz {debug_viz_path} (scale={debug_scale}x)")


def _render_debug_viz(image_path: Path, results, out_path: Path, scale: float):
    img = load_scaled_image(image_path, scale)
    if img is None:
        print(f"  Could not load {image_path} for debug viz; skipping.")
        return

    for r in results:
        if r.ic["class_name"].startswith("clef."):
            color = _COLOR_CLEF
        elif r.reason == "pitchless_symbol":
            color = _COLOR_PITCHLESS
        elif r.reason in ("missing_clef", "missing_staff"):
            color = _COLOR_PROBLEM
        else:
            color = _COLOR_PITCH_OK

        label = None
        if r.pitch:
            label = f"{r.pitch['pname']}{r.pitch['oct']}"
        elif r.reason and r.reason != "pitchless_symbol":
            label = r.reason

        draw_labeled_box(img, r.ic["ulx"], r.ic["uly"], r.ic["ncols"], r.ic["nrows"], label, color, scale)

    cv2.imwrite(str(out_path), img)


def main():
    parser = argparse.ArgumentParser(description="Rodan-style pitch finder")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--ic-xml", required=True, type=Path)
    parser.add_argument("--staff-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--debug-viz", type=Path, default=None)
    parser.add_argument("--debug-scale", type=float, default=2.5)
    args = parser.parse_args()

    run(args.image, args.ic_xml, args.staff_json, args.output, args.debug_viz, args.debug_scale)


if __name__ == "__main__":
    main()
