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
  glyph_pixels.py                 pixel cropping/centroid helpers for algorithm #2
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

## Which algorithm

**`pitch_finder.py`** is pure geometry, no pixel access. The
neumes-cheatsheet CSV tells us how many notes a neume class has and each
note's interval offset from the first one (e.g. `neume.clivis2` = 2 notes,
second one a step below); the glyph's bbox top/bottom is mapped onto that
known interval span to get every note's line/space position at once — so
one glyph can produce several pitches. Stave assignment picks the closest
stave within a margin derived from that stave's own local line spacing;
clef lookup only looks within the same stave (nearest one to the left),
reporting `missing_clef` rather than borrowing a neighboring stave's.

**`rodan_pitch_finder.py`** reimplements the old Rodan design instead:
pixel-centroid based (crops the real ink, with per-class special cases
like "just the top of a virga" to exclude the stem) to get one reference
point -> one pitch per glyph, no decomposition. Stave assignment has three
fallback tiers (bbox overlap -> y-margin -> optional forced-nearest). The
clef propagates across the whole page in reading order, so a stave with no
clef of its own silently inherits the previous one instead of being flagged.

Both are predictions, not checked against ground truth — run both and compare debug images rather than treating either as "correct".

## Tests

```bash
cd scripts && python -m pytest script_tests/ -v   # 52 passing
```

## Known limitations

- `CLEF_OCTAVE_REFERENCE` in `clef_rules.py` (C=4/F=3/G=4) is an unvalidated placeholder.
- Pitch depends on staff-finding's `within_stave_index` being correctly ordered.
- No human-verified pitch/MEI ground truth exists yet.
