"""
CLI driver for the Rodan-style pitch finder (rodan_pitch_finder.py).

Produces the same output shape and debug visualization as run_pitch_finding.py
so the two algorithms can be viewed side by side. It still takes its inputs one
flag at a time; run_pitch_finding.py additionally accepts a page folder and
discovers them (page_inputs.py).

Usage:
    python run_rodan_pitch_finding.py \\
        --image page.jpg --ic-xml ic_output/ic-session-page.xml \\
        --staff-json page_stafflines.json --output page_rodan_pitch_finding.json \\
        [--debug-viz page_rodan_pitch_finding_debug.jpg]

As in run_pitch_finding.py, --output also accepts a directory, --debug-viz can
be passed bare to name the overlay after --output, and the overlay is written
twice -- once labelled, once as a text-free '_nolabels' copy.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from ic_io import parse_ic_xml
from staff_io import load_staves_with_report
from rodan_pitch_finder import find_pitches_rodan
from viz_utils import (load_scaled_image, draw_labeled_box, draw_note_center,
                       draw_stafflines, resolve_output_path, resolve_debug_viz_path,
                       unlabeled_variant_path, write_image)

_COLOR_PITCH_OK = (60, 170, 60)      # green: a note with pitch computed
_COLOR_CLEF = (200, 130, 0)          # blue: clef (the pitch reference, not a note)
_COLOR_PITCHLESS = (150, 150, 150)   # grey: pitchless_symbol / not music
_COLOR_PROBLEM = (40, 40, 220)       # red: missing_clef / missing_staff


def run(image_path: Path, ic_xml_path: Path, staff_json_path: Path, output_path: Path,
        debug_viz: str = None, debug_scale: float = 2.5, regroup_staves: bool = True):
    # Resolve both artifact paths before doing any work: an unwritable
    # --debug-viz should be reported now, not after the debug render.
    output_path = resolve_output_path(output_path, image_path, "_rodan_pitch_finding.json")
    debug_viz_path = resolve_debug_viz_path(
        debug_viz, output_path.with_name(f"{output_path.stem}_debug.jpg"))

    glyphs = parse_ic_xml(ic_xml_path)
    staves, regroup_report = load_staves_with_report(staff_json_path,
                                                     regroup=regroup_staves)
    if regroup_report:
        print(regroup_report.summary())
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(page, indent=2))

    reasons = Counter(r.reason or "pitch_ok" for r in results)
    print(f"{len(results)} glyphs processed. Breakdown: {dict(reasons)}")
    print(f"Wrote {output_path}")

    if debug_viz_path:
        _render_debug_viz(image_path, results, staves, debug_viz_path, debug_scale)
        print(f"Wrote debug viz {debug_viz_path} (scale={debug_scale}x)")
        # Same overlay, no text. Rendered from a second load of the page rather
        # than by peeling labels off the first canvas: the labelled pass draws
        # white-backed text over the boxes it has already drawn, so there is
        # nothing left to remove afterwards.
        plain_path = unlabeled_variant_path(debug_viz_path)
        _render_debug_viz(image_path, results, staves, plain_path, debug_scale,
                          labels=False)
        print(f"Wrote debug viz {plain_path} (scale={debug_scale}x, boxes and "
              "note centers only)")


def _render_debug_viz(image_path: Path, results, staves, out_path: Path, scale: float,
                      labels: bool = True):
    """Draw the overlay for one page and write it to out_path.

    labels=False suppresses the pitch labels and staff-line tags, leaving only
    staff lines, glyph boxes and centroid crosshairs. On a full page the labels
    are wider than the glyphs they belong to, so where the notation is dense
    they overlap each other and cover the ink and markers being checked; the
    text-free copy is where that geometry stays visible. Both are rendered at
    the same scale, so the same page pixel is the same pixel in both files.
    """
    img = load_scaled_image(image_path, scale)
    if img is None:
        print(f"  Could not load {image_path} for debug viz; skipping.")
        return

    # Staff lines first, so glyph boxes and labels stay legible on top of them.
    draw_stafflines(img, staves, scale, label_lines=labels)

    colored = []
    for r in results:
        if r.ic["class_name"].startswith("clef."):
            color = _COLOR_CLEF
        elif r.reason == "pitchless_symbol":
            color = _COLOR_PITCHLESS
        elif r.reason in ("missing_clef", "missing_staff"):
            color = _COLOR_PROBLEM
        else:
            color = _COLOR_PITCH_OK
        colored.append((r, color))

        label = None
        if labels:
            if r.pitch:
                label = f"{r.pitch['pname']}{r.pitch['oct']}"
            elif r.reason and r.reason != "pitchless_symbol":
                label = r.reason

        draw_labeled_box(img, r.ic["ulx"], r.ic["uly"], r.ic["ncols"], r.ic["nrows"], label, color, scale)

    # Notehead centers in a second pass, on top of every box and label. One per
    # glyph here -- this algorithm is one-pitch-per-glyph by design, so unlike
    # run_pitch_finding there is nothing to number. The marker is the measured
    # ink centroid *before* the line/space snap, so its offset from the nearest
    # staff line is how close the snap came to going the other way.
    for r, color in colored:
        if r.center_x is None or r.center_y is None:
            continue
        draw_note_center(img, r.center_x, r.center_y, scale, color)

    write_image(out_path, img)


def main():
    parser = argparse.ArgumentParser(description="Rodan-style pitch finder")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--ic-xml", required=True, type=Path)
    parser.add_argument("--staff-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                         help="JSON file to write, or a directory to write "
                              "<image stem>_rodan_pitch_finding.json into.")
    parser.add_argument("--debug-viz", nargs="?", const="auto", default=None,
                         help="Render the debug overlay, twice: as given, plus a "
                              "text-free '_nolabels' copy with only boxes, note "
                              "centers and staff lines. Give an image filename, or "
                              "pass the flag bare to name it after --output.")
    parser.add_argument("--debug-scale", type=float, default=2.5)
    parser.add_argument("--no-regroup", dest="regroup_staves", action="store_false",
                         help="Trust staff-finding's own stave_id / "
                              "within_stave_index instead of re-deriving the "
                              "grouping from line geometry. Its grouping is by y "
                              "alone, so on a two-column page it merges the two "
                              "columns' lines into single eight-line staves.")
    args = parser.parse_args()

    try:
        run(args.image, args.ic_xml, args.staff_json, args.output, args.debug_viz,
            args.debug_scale, args.regroup_staves)
    except ValueError as exc:  # bad --debug-viz path: a usage error, not a crash
        parser.error(str(exc))


if __name__ == "__main__":
    main()
