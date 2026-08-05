"""
CLI driver for the Mothra pitch-finding prototype.

Ties together ic_io / staff_io / neume_shapes / pitch_finder to turn one
page's (IC XML, staff-finding JSON) pair into an intermediate JSON recording
each glyph's stave assignment, staff position, and pitch (or the reason it
couldn't be computed). Optionally renders a debug overlay on the manuscript
image, in the same spirit as staff-finding's *_stave_grouping.png artifacts.

Usage:
    python run_pitch_finding.py ../McGill_MS234-064 --debug-viz

One argument: the page folder. page_inputs finds the image, IC XML and staff
JSON in its input/ subfolder, and the artifacts are written to its output/
subfolder (<image stem>_pitch_finding.json, plus _debug.jpg and
_debug_nolabels.jpg with --debug-viz -- the same overlay with and without
captions, since the captions are what makes a densely notated page unreadable).
Keeping the two apart is what lets a page be re-run without its own artifacts
becoming candidate inputs; output/ is created on the first write. Any of those
paths can still be given explicitly to override discovery:

    python run_pitch_finding.py page_dir/ --ic-xml other/ic.xml \\
        --output out_dir/ --debug-viz overlay.jpg

--anchor-mode picks how each glyph's notehead position is found: "pixel"
(default) reads rodan-style per-class ink centroids off the image, "bbox"
uses bbox geometry only. See pitch_finder's module docstring.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from ic_io import parse_ic_xml
from staff_io import load_staves_with_report
from neume_shapes import load_neume_shapes
from page_inputs import PageInputs, resolve_page_inputs
from pitch_finder import find_pitches
from viz_utils import (load_scaled_image, draw_labeled_box, draw_note_center,
                       draw_stafflines, resolve_output_path, resolve_debug_viz_path,
                       unlabeled_variant_path, write_image)

DEFAULT_NEUME_CSV = Path(__file__).parent.parent / "neumes-cheatsheet" / "csv-square_notation_neume_level_newest.csv"

# BGR colors for the debug overlay.
_COLOR_PITCH_OK = (60, 170, 60)      # green: a note with pitch computed, CSV-backed
_COLOR_APPROX = (10, 165, 230)       # amber: the neume's shape is not CSV-backed --
                                      #        either the single-note fallback for an
                                      #        undecodable class, or intervals decoded
                                      #        from the class name (neume.distropha)
_COLOR_CLEF = (200, 130, 0)          # blue: clef (the pitch *reference*, not a note itself)
_COLOR_PITCHLESS = (150, 150, 150)   # grey: pitchless_symbol / not music
_COLOR_PROBLEM = (40, 40, 220)       # red: missing_clef / missing_staff / no_line_coverage


def run(inputs: PageInputs, output_path: Path = None, *,
        neume_csv_path: Path = DEFAULT_NEUME_CSV, debug_viz: str = None,
        debug_scale: float = 2.5, anchor_mode: str = "pixel",
        regroup_staves: bool = True):
    # Resolve both artifact paths before doing any work: an unwritable
    # --debug-viz should be reported now, not after the debug render. With no
    # --output, both land in the page folder's output/ (created at write time).
    output_path = resolve_output_path(output_path or inputs.output_dir, inputs.image,
                                     "_pitch_finding.json")
    debug_viz_path = resolve_debug_viz_path(
        debug_viz, output_path.with_name(f"{output_path.stem}_debug.jpg"))

    glyphs = parse_ic_xml(inputs.ic_xml)
    staves, regroup_report = load_staves_with_report(inputs.staff_json,
                                                     regroup=regroup_staves)
    if regroup_report:
        print(regroup_report.summary())
    shapes = load_neume_shapes(neume_csv_path)

    image = None
    if anchor_mode == "pixel":
        image = cv2.imread(str(inputs.image))
        if image is None:
            raise FileNotFoundError(
                f"Could not load image: {inputs.image} (needed for "
                "--anchor-mode pixel; --anchor-mode bbox needs no pixels)")

    results = find_pitches(glyphs, staves, shapes, image)

    page = {
        "image": str(inputs.image),
        "ic_xml": str(inputs.ic_xml),
        "staff_json": str(inputs.staff_json),
        "anchor_mode": anchor_mode,
        "glyphs": [r.to_dict() for r in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(page, indent=2))

    reasons = Counter(r.reason or "pitch_ok" for r in results)
    print(f"{len(results)} glyphs processed (anchor mode: {anchor_mode}). "
          f"Breakdown: {dict(reasons)}")
    _print_anchor_summary(results)
    print(f"Wrote {output_path}")

    if debug_viz_path:
        _render_debug_viz(inputs.image, results, staves, debug_viz_path, debug_scale)
        print(f"Wrote debug viz {debug_viz_path} (scale={debug_scale}x)")
        # Same overlay, no text. Rendered from a second load of the page rather
        # than by peeling labels off the first canvas: the labelled pass draws
        # white-backed text over the boxes it has already drawn, so there is
        # nothing left to remove afterwards.
        plain_path = unlabeled_variant_path(debug_viz_path)
        _render_debug_viz(inputs.image, results, staves, plain_path, debug_scale,
                          labels=False)
        print(f"Wrote debug viz {plain_path} (scale={debug_scale}x, boxes and "
              "note centers only)")


def _print_anchor_summary(results):
    """Report how many glyphs actually got a measured anchor, per class.

    A silent fall back to geometry on the classes the pixel crops exist for
    (virga's stem, podatus's second head) would look exactly like a working
    run, so count them out loud.
    """
    fell_back = Counter(r.ic["class_name"] for r in results
                        if "pixel_anchor_unavailable" in r.flags)
    if fell_back:
        print(f"  {sum(fell_back.values())} glyph(s) fell back to bbox geometry "
              f"(no measurable ink): {dict(fell_back)}")
    off_range = sum(1 for r in results if "anchor_x_fell_back_to_center" in r.flags)
    if off_range:
        print(f"  {off_range} glyph(s) had their anchor x clamped to the bbox center "
              "(crop band outside the detected staff lines)")


def _pitch_str(pitch: dict) -> str:
    return f"{pitch['pname']}{pitch['oct']}"


def _color_for(r) -> tuple:
    if r.ic["class_name"].startswith("clef."):
        # The clef is the pitch *reference*, not a note -- color it distinctly
        # so it doesn't read as "just another computed pitch".
        return _COLOR_CLEF
    if r.reason == "pitchless_symbol":
        return _COLOR_PITCHLESS
    if r.reason in ("missing_clef", "missing_staff", "no_line_coverage"):
        return _COLOR_PROBLEM
    if {"approximate_unknown_shape", "shape_from_class_name"} & set(r.flags):
        return _COLOR_APPROX
    return _COLOR_PITCH_OK


def _box_label(r) -> str:
    """The label above the glyph box: every note's pitch in note order.

    A multi-note neume reads as "d4-f4-e4", so the whole decomposition is
    visible without having to open the JSON -- labelling only the first
    component (as this used to) makes a decomposed torculus indistinguishable
    from a punctum. Which pitch sits where is left to the per-component
    crosshair labels; this is the reading-order summary.
    """
    pitches = [nc.pitch for nc in r.note_components]
    if pitches and all(pitches):
        label = "-".join(_pitch_str(p) for p in pitches)
        if "approximate_unknown_shape" in r.flags:
            # Flag that this pitch is a guess (single-note fallback, not a
            # CSV-backed decomposition) and show what IC called it.
            label = f"~{label} ({r.ic['class_name']})"
        elif "shape_from_class_name" in r.flags:
            # Properly decomposed, but on intervals read off the class name
            # rather than the CSV -- neume.distropha's unison is still an
            # unconfirmed reading (see the plan doc).
            label = f"?{label} ({r.ic['class_name']})"
        return label
    if r.reason and r.reason != "pitchless_symbol":
        return r.reason
    return None


def _draw_note_centers(img, r, color: tuple, scale: float, labels: bool = True):
    """One crosshair per computed notehead center, labelled "<note no>:<pitch>".

    Components are numbered in note order (1 = the neume's first note), which
    is not always top-to-bottom -- a torculus goes up then down -- so the
    number is what ties a marker back to the box label's reading order.

    Components that resolved to the same step share one marker: a neume can
    legitimately repeat a pitch (torculus33 = [0, 2, 0]), and two crosshairs
    drawn on the same pixel would just look like one marker with an
    overstruck label. "1,3:d4" says the same thing honestly.

    Single-note glyphs get an unlabelled marker -- the box label already names
    the only pitch there is, so a second copy of it is noise. With labels=False
    every marker is unlabelled, for the text-free copy of the overlay.
    """
    groups = {}  # rounded page-pixel (x, y) -> (x, y, [note numbers], pitch)
    for note_no, nc in enumerate(r.note_components, start=1):
        if nc.center_x is None or nc.center_y is None:
            continue
        key = (round(nc.center_x), round(nc.center_y))
        if key in groups:
            groups[key][2].append(note_no)
        else:
            groups[key] = (nc.center_x, nc.center_y, [note_no], nc.pitch)

    multi_note = len(r.note_components) > 1
    for x, y, note_nos, pitch in groups.values():
        label = None
        if labels and multi_note and pitch:
            label = f"{','.join(str(n) for n in note_nos)}:{_pitch_str(pitch)}"
        # Labels start at the box's right edge so they never cover the glyph's
        # own ink -- with up to four of them stacked inside one bbox, that ink
        # is the thing you're checking the markers against.
        draw_note_center(img, x, y, scale, color, label,
                         label_x=r.ic["ulx"] + r.ic["ncols"])


def _render_debug_viz(image_path: Path, results, staves, out_path: Path,
                      scale: float = 2.5, labels: bool = True):
    """Draw the overlay for one page and write it to out_path.

    labels=False suppresses every caption -- pitch labels, per-note marker
    labels and staff-line tags -- leaving only staff lines, glyph boxes and
    note-center crosshairs. On a full page the labelled version is unreadable
    where the notation is dense (labels are wider than the glyphs they belong
    to, so they collide), and this is the copy where the geometry actually being
    debugged, box vs marker vs staff line, is visible. Both are rendered at the
    same scale, so the same page pixel is the same pixel in both files.
    """
    img = load_scaled_image(image_path, scale)
    if img is None:
        print(f"  Could not load {image_path} for debug viz; skipping.")
        return

    # Staff lines first, so glyph boxes and labels stay legible on top of them.
    draw_stafflines(img, staves, scale, label_lines=labels)

    colored = [(r, _color_for(r)) for r in results]

    for r, color in colored:
        draw_labeled_box(img, r.ic["ulx"], r.ic["uly"], r.ic["ncols"], r.ic["nrows"],
                         _box_label(r) if labels else None, color, scale)

    # Markers in a second pass, after every box and label: a marker sits inside
    # its own box, but glyphs here are close enough together that a neighbour's
    # white-backed label would paint over it if they were drawn glyph by glyph.
    for r, color in colored:
        _draw_note_centers(img, r, color, scale, labels)

    write_image(out_path, img)


def main():
    parser = argparse.ArgumentParser(description="Mothra pitch-finding prototype")
    parser.add_argument("page", type=Path,
                         help="Page folder whose input/ subfolder holds the image, "
                              "IC XML and staff JSON (or the page image itself, to "
                              "pick one of two pages sharing a folder).")
    parser.add_argument("--output", type=Path, default=None,
                         help="JSON file to write, or a directory to write "
                              "<image stem>_pitch_finding.json into. "
                              "Defaults to the page folder's output/ subfolder.")
    parser.add_argument("--debug-viz", nargs="?", const="auto", default=None,
                         help="Render the debug overlay, twice: as given, plus a "
                              "text-free '_nolabels' copy with only boxes, note "
                              "centers and staff lines. Give an image filename, or "
                              "pass the flag bare to name it after --output.")
    parser.add_argument("--debug-scale", type=float, default=2.5,
                         help="Upscale factor for the debug viz canvas. Label text "
                              "grows with its square root, so a bigger scale means "
                              "labels that are both more legible and less crowded.")
    parser.add_argument("--anchor-mode", choices=("pixel", "bbox"), default="pixel",
                         help="How to find each glyph's notehead: 'pixel' (default) "
                              "uses rodan-style per-class ink centroids from the image; "
                              "'bbox' uses bbox top/bottom geometry only.")
    parser.add_argument("--no-regroup", dest="regroup_staves", action="store_false",
                         help="Trust staff-finding's own stave_id / "
                              "within_stave_index instead of re-deriving the "
                              "grouping from line geometry. Its grouping is by y "
                              "alone, so on a two-column page it merges the two "
                              "columns' lines into single eight-line staves.")
    # Overrides for anything discovery gets wrong or can't see, e.g. an IC XML
    # kept outside the page folder.
    override_help = ("Use this file instead of discovering one in the page's "
                     "input/ folder.")
    parser.add_argument("--image", type=Path, default=None, help=override_help)
    parser.add_argument("--ic-xml", type=Path, default=None, help=override_help)
    parser.add_argument("--staff-json", type=Path, default=None, help=override_help)
    parser.add_argument("--neume-csv", type=Path, default=DEFAULT_NEUME_CSV)
    args = parser.parse_args()

    try:
        inputs = resolve_page_inputs(args.page, args.image, args.ic_xml, args.staff_json)
        print(f"Page {inputs.page_dir}: image={inputs.image.name}, "
              f"ic_xml={inputs.ic_xml.name}, staff_json={inputs.staff_json.name}")
        run(inputs, args.output, neume_csv_path=args.neume_csv,
            debug_viz=args.debug_viz, debug_scale=args.debug_scale,
            anchor_mode=args.anchor_mode, regroup_staves=args.regroup_staves)
    except ValueError as exc:  # unresolvable input / bad --debug-viz path
        parser.error(str(exc))


if __name__ == "__main__":
    main()
