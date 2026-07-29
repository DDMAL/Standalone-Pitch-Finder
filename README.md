# Mothra Pitch-Finding Prototype

Given IC (Interactive Classifier) output and staff-finding output for a
manuscript page, assigns each glyph to a stave/line/space and computes its
pitch when a clef is available.

## Layout

```
scripts/
  ic_io.py, staff_io.py           parse IC XML / staff-finding JSON
  neume_shapes.py, clef_rules.py  neume interval table, clef -> pitch conversion
  pitch_finder.py                 algorithm #1 (ours; decomposes multi-note neumes)
  rodan_pitch_finder.py           algorithm #2 (Rodan heuristic pitch finder, reimplemented; pixel-based, one pitch/glyph)
  glyph_pixels.py                 pixel cropping/centroid helpers -- notehead finding, used by both
  run_pitch_finding.py / run_rodan_pitch_finding.py   CLIs + debug viz for each algorithm
  render_ic_debug.py              visualize IC's raw classification only
  viz_utils.py                    shared drawing helpers
  script_tests/                   pytest unit tests
neumes-cheatsheet/                 neume class -> MEI/interval lookup CSV
GentAnt1475_0017_AC_rightcrop/, McGill_MS234-064/   sample pages (image + IC XML + staff JSON)
CH-E-611_Einsiedeln/                raw scans, not annotated yet
text_music_detector_fulldata.pt     Phase-1 YOLO weights
render_mei.py                       unrelated standalone tool, needs its own MEI file
```

## Setup

```bash
conda create -n pitch-finding python=3.11 -y
conda activate pitch-finding
pip install numpy opencv-python pytest
```

## Quick start

Both sample folders already have image + IC XML + staff JSON:

```bash
cd scripts

python run_pitch_finding.py \
  --image ../McGill_MS234-064/McGill_MS234-064.jpg \
  --ic-xml ../McGill_MS234-064/ic-session-McGill_MS234-064-manual-neumes.xml \
  --staff-json ../McGill_MS234-064/McGill_MS234-064_stafflines.json \
  --output ../McGill_MS234-064/out.json --debug-viz ../McGill_MS234-064/out_debug.jpg

python run_rodan_pitch_finding.py \
  --image ../McGill_MS234-064/McGill_MS234-064.jpg \
  --ic-xml ../McGill_MS234-064/ic-session-McGill_MS234-064-manual-neumes.xml \
  --staff-json ../McGill_MS234-064/McGill_MS234-064_stafflines.json \
  --output ../McGill_MS234-064/out_rodan.json --debug-viz ../McGill_MS234-064/out_rodan_debug.jpg

python render_ic_debug.py \
  --image ../McGill_MS234-064/McGill_MS234-064.jpg \
  --ic-xml ../McGill_MS234-064/ic-session-McGill_MS234-064-manual-neumes.xml \
  --output ../McGill_MS234-064/ic_debug.jpg
```

Debug-viz color legend: green = pitch found, amber = fell back to a
single-note approximation (class missing from the CSV, algorithm #1 only),
blue = clef, grey = pitchless, red = no stave/clef found.

Every pitched glyph also gets a crosshair at each **computed notehead
center** — the exact point the staff position was read from. The label above
a box lists every note of the neume in note order (`A3-C4-F3`); in algorithm
#1 each crosshair is labelled `<note no>:<pitch>` so you can see which note
of a multi-note neume landed where (numbers, not top-to-bottom order: a
torculus goes up then down, and notes sharing a step share one marker).
Algorithm #2 is one pitch per glyph, so its single crosshair is unlabelled —
it marks the raw ink centroid *before* the line/space snap, at that
algorithm's own reference x (the bbox's left edge, not its center).

## Which algorithm

**`pitch_finder.py`** decomposes multi-note neumes. The neumes-cheatsheet CSV
tells us how many notes a neume class has and each note's interval offset
from the first one (e.g. `neume.clivis2` = 2 notes, second one a step below),
so **one anchor point on the glyph determines every note**: read that point's
line/space position, then step off it by the known intervals. Stave
assignment picks the closest stave within a margin derived from that stave's
own local line spacing; clef lookup only looks within the same stave (nearest
one to the left), reporting `missing_clef` rather than borrowing a
neighboring stave's.

`--anchor-mode` picks how that anchor is found:

- **`pixel`** (default) borrows algorithm #2's notehead finding: the anchor is
  the ink centroid of a per-class crop of the bbox (`glyph_pixels.py`), whose
  whole purpose is to exclude a virga's stem or a podatus's upper head. Which
  crop fired also says *which* note the point is — a bottom-left crop is the
  neume's lowest note, a top crop its highest, a full-bbox crop the middle of
  the span — and that is what ties the measured point to the interval table.
- **`bbox`** is the original geometry-only path: no pixel access, the bbox's
  top and bottom edges are mapped onto the interval span. It assumes the
  outermost noteheads' centers sit on the ink extremes, which any stem or
  tail breaks. Kept for comparison and for running without the image.

On `McGill_MS234-064` the switch moves 106 of 218 notes by a whole pitch, and
collapses the per-class step disagreement with algorithm #2 (median absolute
0.49 → 0.14 steps; `neume.virga` −0.70 → −0.03, `neume.podatus2b` −1.10 →
−0.09). On `GentAnt1475_0017_AC_rightcrop`, where the same classes are drawn
smaller, `bbox` was off by up to 3.3 steps and `pixel` is within 0.34.

Every glyph's output records the `anchor` it was placed from — source, crop
region, pixel position, step, and which note of the neume the point was taken
to be — so a neume whose pitches are uniformly off can be traced to a bad
anchor position vs. a bad anchor *role*.

**`rodan_pitch_finder.py`** reimplements the old Rodan design instead:
pixel-centroid based (crops the real ink, with per-class special cases
like "just the top of a virga" to exclude the stem) to get one reference
point -> one pitch per glyph, no decomposition. Stave assignment has three
fallback tiers (bbox overlap -> y-margin -> optional forced-nearest). It
snaps each position to the nearest line/space, and clefs to the nearest line.
The clef propagates across the whole page in reading order, so a stave with no
clef of its own silently inherits the previous one instead of being flagged.

So in default `pixel` mode the two share their notehead finding and differ
only in what they do with it: algorithm #2 reports that one point as the
glyph's one pitch, algorithm #1 uses it to place every note of the neume,
against its own stave assignment and per-stave clef resolution.

Both are predictions, not checked against ground truth — run both and compare debug images rather than treating either as "correct".

## Tests

```bash
cd scripts && python -m pytest script_tests/ -v   # 52 passing
```

## Known limitations

- `CLEF_OCTAVE_REFERENCE` in `clef_rules.py` (C=4/F=3/G=4) is an unvalidated placeholder.
- Pitch depends on staff-finding's `within_stave_index` being correctly ordered.
- No human-verified pitch/MEI ground truth exists yet.
