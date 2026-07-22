"""
Render IC's raw classification results on the manuscript image.

This is deliberately decoupled from staff-finding/pitch_finder -- it draws
exactly what IC said about each glyph (class_name, state, confidence), so
you can visually judge IC's own classification accuracy without any of the
stave-assignment/pitch logic in between.

Usage:
    python render_ic_debug.py --image page.jpg --ic-xml ic-session-page.xml \\
        --output page_ic_debug.jpg [--scale 2.5] [--show-confidence]
"""

import argparse
from pathlib import Path

import cv2

from ic_io import parse_ic_xml
from viz_utils import load_scaled_image, draw_labeled_box

# BGR colors, grouped by category so misclassifications between categories
# (e.g. a neume mistaken for skip.*) jump out by color alone.
_COLOR_NEUME = (60, 170, 60)        # green: neume.* / custos -- pitch-bearing
_COLOR_CLEF = (200, 130, 0)         # blue: clef.*
_COLOR_MODIFIER = (170, 90, 170)    # purple: divisio.* / accidental.*
_COLOR_JUNK = (0, 140, 255)         # orange: skip.* (Gamera's junk catch-all)
_COLOR_TEXT = (150, 150, 150)       # grey: text bbox, never sent through the neume classifier
_COLOR_OTHER = (40, 40, 220)        # red: anything else / unrecognized category


def _color_for(class_name: str, state: str) -> tuple:
    if state == "UNCLASSIFIED" or class_name == "text":
        return _COLOR_TEXT
    if class_name.startswith("neume.") or class_name == "custos":
        return _COLOR_NEUME
    if class_name.startswith("clef."):
        return _COLOR_CLEF
    if class_name.startswith("divisio.") or class_name.startswith("accidental."):
        return _COLOR_MODIFIER
    if class_name.startswith("skip."):
        return _COLOR_JUNK
    return _COLOR_OTHER


def run(image_path: Path, ic_xml_path: Path, output_path: Path,
        scale: float = 2.5, show_confidence: bool = False, show_text: bool = False):
    glyphs = parse_ic_xml(ic_xml_path)
    n_total = len(glyphs)
    if not show_text:
        glyphs = [g for g in glyphs if not (g.state == "UNCLASSIFIED" or g.class_name == "text")]

    img = load_scaled_image(image_path, scale)
    if img is None:
        print(f"Could not load {image_path}")
        return

    for g in glyphs:
        color = _color_for(g.class_name, g.state)
        label = g.class_name
        if show_confidence and g.state == "AUTOMATIC":
            label = f"{label} ({g.confidence:.2f})"
        draw_labeled_box(img, g.ulx, g.uly, g.ncols, g.nrows, label, color, scale)

    cv2.imwrite(str(output_path), img)
    hidden_note = "" if show_text else f" ({n_total - len(glyphs)} text/unclassified bboxes hidden, use --show-text to include)"
    print(f"{len(glyphs)} glyphs drawn{hidden_note}. Wrote {output_path} (scale={scale}x)")
    print("Colors: green=neume/custos, blue=clef, purple=divisio/accidental, "
          "orange=skip.* (junk), grey=text (unclassified), red=other")


def main():
    parser = argparse.ArgumentParser(description="Render IC's raw classification output on the manuscript image")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--ic-xml", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scale", type=float, default=2.5)
    parser.add_argument("--show-confidence", action="store_true",
                         help="Append IC's confidence score to the label for AUTOMATIC glyphs.")
    parser.add_argument("--show-text", action="store_true",
                         help="Also draw text/UNCLASSIFIED bboxes (hidden by default -- they were "
                              "never sent through the neume classifier, so they just add clutter "
                              "when judging IC's classification accuracy).")
    args = parser.parse_args()

    run(args.image, args.ic_xml, args.output, args.scale, args.show_confidence, args.show_text)


if __name__ == "__main__":
    main()
