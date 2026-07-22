"""
CLI driver for the Mothra pitch-finding prototype.

Ties together ic_io / staff_io / neume_shapes / pitch_finder to turn one
page's (IC XML, staff-finding JSON) pair into an intermediate JSON recording
each glyph's stave assignment, staff position, and pitch (or the reason it
couldn't be computed). Optionally renders a debug overlay on the manuscript
image, in the same spirit as staff-finding's *_stave_grouping.png artifacts.

Usage:
    python run_pitch_finding.py \\
        --image page.jpg --ic-xml ic_output/ic-session-page.xml \\
        --staff-json page_stafflines.json --output page_pitch_finding.json \\
        [--neume-csv neumes-cheatsheet/csv-square_notation_neume_level_newest.csv] \\
        [--debug-viz page_pitch_finding_debug.jpg]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from ic_io import parse_ic_xml
from staff_io import load_staves
from neume_shapes import load_neume_shapes
from pitch_finder import find_pitches
from viz_utils import load_scaled_image, draw_labeled_box

DEFAULT_NEUME_CSV = Path(__file__).parent.parent / "neumes-cheatsheet" / "csv-square_notation_neume_level_newest.csv"

# BGR colors for the debug overlay.
_COLOR_PITCH_OK = (60, 170, 60)      # green: a note with pitch computed, CSV-backed
_COLOR_APPROX = (10, 165, 230)       # amber: pitch computed via the single-note fallback
                                      #        for a class missing from the neume CSV
_COLOR_CLEF = (200, 130, 0)          # blue: clef (the pitch *reference*, not a note itself)
_COLOR_PITCHLESS = (150, 150, 150)   # grey: pitchless_symbol / not music
_COLOR_PROBLEM = (40, 40, 220)       # red: missing_clef / missing_staff / no_line_coverage


def run(image_path: Path, ic_xml_path: Path, staff_json_path: Path, output_path: Path,
        neume_csv_path: Path, debug_viz_path: Path = None, debug_scale: float = 2.5):
    glyphs = parse_ic_xml(ic_xml_path)
    staves = load_staves(staff_json_path)
    shapes = load_neume_shapes(neume_csv_path)

    results = find_pitches(glyphs, staves, shapes)

    page = {
        "image": str(image_path),
        "ic_xml": str(ic_xml_path),
        "staff_json": str(staff_json_path),
        "glyphs": [r.to_dict() for r in results],
    }
    output_path.write_text(json.dumps(page, indent=2))

    reasons = Counter(r.reason or "pitch_ok" for r in results)
    print(f"{len(results)} glyphs processed. Breakdown: {dict(reasons)}")
    print(f"Wrote {output_path}")

    if debug_viz_path:
        _render_debug_viz(image_path, results, debug_viz_path, debug_scale)
        print(f"Wrote debug viz {debug_viz_path} (scale={debug_scale}x)")


def _render_debug_viz(image_path: Path, results, out_path: Path, scale: float = 2.5):
    img = load_scaled_image(image_path, scale)
    if img is None:
        print(f"  Could not load {image_path} for debug viz; skipping.")
        return

    for r in results:
        is_approx = bool(r.flags and "approximate_unknown_shape" in r.flags)

        if r.ic["class_name"].startswith("clef."):
            # The clef is the pitch *reference*, not a note -- color it
            # distinctly so it doesn't read as "just another computed pitch".
            color = _COLOR_CLEF
        elif r.reason == "pitchless_symbol":
            color = _COLOR_PITCHLESS
        elif r.reason in ("missing_clef", "missing_staff", "no_line_coverage"):
            color = _COLOR_PROBLEM
        elif is_approx:
            color = _COLOR_APPROX
        else:
            color = _COLOR_PITCH_OK

        label = None
        if r.note_components and r.note_components[0].pitch:
            p = r.note_components[0].pitch
            label = f"{p['pname']}{p['oct']}"
            if is_approx:
                # Flag that this pitch is a guess (single-note fallback, not
                # a CSV-backed decomposition) and show what IC called it.
                label = f"~{label} ({r.ic['class_name']})"
        elif r.reason and r.reason != "pitchless_symbol":
            label = r.reason

        draw_labeled_box(img, r.ic["ulx"], r.ic["uly"], r.ic["ncols"], r.ic["nrows"], label, color, scale)

    cv2.imwrite(str(out_path), img)


def main():
    parser = argparse.ArgumentParser(description="Mothra pitch-finding prototype")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--ic-xml", required=True, type=Path)
    parser.add_argument("--staff-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--neume-csv", type=Path, default=DEFAULT_NEUME_CSV)
    parser.add_argument("--debug-viz", type=Path, default=None)
    parser.add_argument("--debug-scale", type=float, default=2.5,
                         help="Upscale factor for the debug viz canvas (bigger = more legible labels).")
    args = parser.parse_args()

    run(args.image, args.ic_xml, args.staff_json, args.output, args.neume_csv, args.debug_viz, args.debug_scale)


if __name__ == "__main__":
    main()
