"""Score algorithm #1's stave steps against the hand labels.

Compares annotate_notecenters.heuristic_steps() -- pitch_finder in pixel mode,
each note's step rounded, i.e. the same guess the annotator was shown -- with
<page>/labels/human_annotated_stave_steps.json, page by page. This is where the
accuracy numbers in ANNOTATION_PROGRESS.md and PROGRESS.md come from.

Two metrics, kept in separate columns on purpose (ANNOTATION_PROGRESS.md once
held both in one column, which made its pages incomparable):

  per note   labeled notes whose step equals the heuristic's note at the same
             position, over all labeled notes. A glyph split into the wrong
             number of notes still scores on the positions both lists have.
  per glyph  glyphs whose whole step list equals the label, note count included.

Glyphs the annotator skipped are left out of both. "wrong count" is the number
of glyphs whose note count differs from the label's -- on MS025b-01 that, not
anchor position, is most of the error.

Two things are flagged rather than scored silently:

  - Staff grouping. A page whose staff JSON carries fix_stafflines.py's
    manually_grouped flag is loaded with regroup=False, since regrouping would
    rebuild the hand fix away; the "regroup" column says which way each page
    went.
  - Label/XML drift. Labels are keyed by glyph index -- the order of <glyph>
    elements in the IC XML -- and record the class_name each glyph had when it
    was labeled. "class changed" counts glyphs whose class differs now. A few
    mean some glyphs were reclassified, and their labels still hold; most of a
    page means the XML was replaced, and its labels no longer describe these
    glyphs at all.

Usage:
    python score_stave_steps.py                      # every labeled page in the repo
    python score_stave_steps.py ../MS025b-01 ../NZ-Wt_MSR-03_065r
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from annotate_notecenters import heuristic_steps, labels_path
from ic_io import parse_ic_xml
from neume_shapes import load_neume_shapes
from page_inputs import resolve_page_inputs
from run_pitch_finding import DEFAULT_NEUME_CSV
from staff_io import load_staves

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_GLOB = "*/labels/human_annotated_stave_steps.json"


@dataclass
class PageScore:
    page: str
    regroup: bool
    glyphs: int = 0
    notes: int = 0
    notes_right: int = 0
    glyphs_right: int = 0
    wrong_count: int = 0    # glyphs whose note count differs from the label's
    class_changed: int = 0  # glyphs whose class_name differs from the label's


def score_glyph(predicted: list[int], labeled: list[int]) -> tuple[int, bool]:
    """(labeled notes the prediction gets right, whether the whole glyph is right).

    Notes are matched by position, the n-th predicted note against the n-th
    labeled one: a surplus predicted note earns nothing, a missing one loses its
    label's point.
    """
    return sum(p == t for p, t in zip(predicted, labeled)), predicted == labeled


def is_manually_grouped(staff_json: Path) -> bool:
    """Whether fix_stafflines.py hand-grouped this file's lines."""
    entries = json.loads(Path(staff_json).read_text())
    return any("manually_grouped" in (e.get("quality") or {}).get("flags", [])
               for e in entries)


def score_page(page: Path, shapes) -> PageScore:
    inputs = resolve_page_inputs(page)
    labels_file = labels_path(inputs)
    if not labels_file.exists():
        raise FileNotFoundError(f"{inputs.page_dir} has no labels at {labels_file}")
    regroup = not is_manually_grouped(inputs.staff_json)

    glyphs = parse_ic_xml(inputs.ic_xml)
    by_index = {g.index: g for g in glyphs}
    staves = load_staves(inputs.staff_json, regroup=regroup)
    image = cv2.imread(str(inputs.image))
    if image is None:
        raise FileNotFoundError(f"could not read image {inputs.image}")
    predicted = heuristic_steps(glyphs, staves, shapes, image)

    score = PageScore(inputs.page_dir.name, regroup)
    for key, entry in json.loads(labels_file.read_text()).items():
        if entry["skipped"]:
            continue
        index = int(key)
        glyph = by_index.get(index)
        if glyph is None or glyph.class_name != entry["class_name"]:
            score.class_changed += 1
        pred, truth = predicted.get(index, []), entry["steps"]
        notes_right, glyph_right = score_glyph(pred, truth)
        score.glyphs += 1
        score.notes += len(truth)
        score.notes_right += notes_right
        score.glyphs_right += glyph_right
        score.wrong_count += len(pred) != len(truth)
    return score


def _ratio(right: int, total: int) -> str:
    return f"{right / total:6.1%} ({right}/{total})" if total else "-"


def _row(s: PageScore, regroup: str) -> str:
    return (f"{s.page:40s} {regroup:8s} {s.glyphs:6d}  {_ratio(s.notes_right, s.notes):20s} "
            f"{_ratio(s.glyphs_right, s.glyphs):20s} {s.wrong_count:11d} {s.class_changed:13d}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pages", type=Path, nargs="*",
                        help="page folders to score (default: every page folder with labels/)")
    args = parser.parse_args()

    pages = args.pages or sorted(p.parent.parent for p in REPO_ROOT.glob(LABELS_GLOB))
    if not pages:
        parser.error(f"no labeled page folders found under {REPO_ROOT}")
    shapes = load_neume_shapes(DEFAULT_NEUME_CSV)
    scores = [score_page(page, shapes) for page in pages]

    print(f"{'page':40s} {'regroup':8s} {'glyphs':>6s}  {'per note':20s} {'per glyph':20s} "
          f"{'wrong count':>11s} {'class changed':>13s}")
    for s in scores:
        print(_row(s, "yes" if s.regroup else "no"))
    if len(scores) > 1:
        total = PageScore(f"total ({len(scores)} pages)", regroup=True)
        for s in scores:
            total.glyphs += s.glyphs
            total.notes += s.notes
            total.notes_right += s.notes_right
            total.glyphs_right += s.glyphs_right
            total.wrong_count += s.wrong_count
            total.class_changed += s.class_changed
        print(_row(total, ""))

    hand_grouped = [s.page for s in scores if not s.regroup]
    if hand_grouped:
        print(f"\nregroup=no: staff JSON is hand-grouped (manually_grouped flag): "
              f"{', '.join(hand_grouped)}")
    for s in scores:
        if s.class_changed:
            print(f"\nWARNING: {s.page}: {s.class_changed} of {s.glyphs} labeled glyphs have a "
                  f"different class than when they were labeled. A few means reclassified "
                  f"glyphs; most of the page means the IC XML was replaced and its labels "
                  f"no longer match the glyphs -- don't trust this page's numbers.")


if __name__ == "__main__":
    main()
