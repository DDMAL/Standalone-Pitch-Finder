# Mothra Pitch-Finding Prototype

Given IC (Interactive Classifier) output and staff-finding output for a
manuscript page, assigns each glyph to a stave/line/space and computes its
pitch when a clef is available.

## Layout

```
scripts/
  ic_io.py, staff_io.py           parse IC XML / staff-finding JSON
  staff_regroup.py                re-derive stave grouping (columns, fragments) from line geometry
  neume_shapes.py, clef_rules.py  neume interval table, clef -> pitch conversion
  pitch_finder.py                 algorithm #1 (ours; decomposes multi-note neumes)
  rodan_pitch_finder.py           algorithm #2 (Rodan heuristic pitch finder, reimplemented; pixel-based, one pitch/glyph)
  glyph_pixels.py                 pixel cropping/centroid helpers -- notehead finding, used by both
  run_pitch_finding.py / run_rodan_pitch_finding.py   CLIs + debug viz for each algorithm
  render_ic_debug.py              visualize IC's raw classification only
  page_inputs.py                  find a page folder's image / IC XML / staff JSON
  viz_utils.py                    shared drawing helpers
  script_tests/                   pytest unit tests
neumes-cheatsheet/                 neume class -> MEI/interval lookup CSV
GentAnt1475_0017_AC_rightcrop/, McGill_MS234-064/, Breviarium_ad_usum/   sample pages (image + IC XML + staff JSON)
CH-E-611_Einsiedeln/                raw scans, not annotated yet
text_music_detector_fulldata.pt     Phase-1 YOLO weights
render_mei.py                       unrelated standalone tool, needs its own MEI file
```

## Data provenance

Not everything under a page folder is equally trustworthy. IC XML (glyph
bboxes + `class_name`/`confidence`) and staff-finding JSON (stave-line
geometry) are both **model output**, not ground truth: IC's own bboxes are
documented as noisy/imprecise, and staff-finding is wrong often enough that
5 of the 13 hand-labeled pages needed their stave grouping manually fixed
(`scripts/fix_stafflines.py`). The only **human-verified ground truth**
anywhere in this repo is the per-glyph stave **step** recorded in
`<page>/labels/human_annotated_stave_steps.json`, for the 13 pages listed
as `REAL_LABELED_PAGES` in `experiments/v1-hfngl/data.py` -- produced by a
human clicking through each glyph via `scripts/annotate_notecenters.py`
(see `experiments/v1-hfngl/README.md` for the train/test split built from
it). `staff_finding_rerun/` at the repo root holds a from-scratch rerun of
the staff-finding model against all 13 labeled pages' images, kept
separate from each page's own `input/` so later experiments can reuse a
known-fresh detection run without re-invoking the external model.

## Setup

```bash
conda create -n pitch-finding python=3.11 -y
conda activate pitch-finding
pip install numpy opencv-python pytest
```

## Quick start

Each page folder holds its image + IC XML + staff JSON in `input/`, so
pitch-finding takes just the folder — it finds the three inputs in `input/` and
writes `<image stem>_pitch_finding.json` (+ `_debug.jpg` and
`_debug_nolabels.jpg`) to `output/`, creating it if needed:

```bash
cd scripts

python run_pitch_finding.py ../McGill_MS234-064
# -> ../McGill_MS234-064/output/McGill_MS234-064_pitch_finding.json
python run_pitch_finding.py ../McGill_MS234-064 --debug-viz
```

Keeping inputs and artifacts in separate folders is what lets a page be re-run
without a previous run's renders becoming candidate inputs.

Pass the page image instead of the folder to pick one of two pages sharing a
folder, and `--image` / `--ic-xml` / `--staff-json` / `--output` to override
any path discovery gets wrong:

```bash
python run_pitch_finding.py ../McGill_MS234-064/input/McGill_MS234-064.jpg \
  --ic-xml ../elsewhere/ic-session-manual-neumes.xml \
  --output ../McGill_MS234-064/out.json --debug-viz ../McGill_MS234-064/out_debug.jpg
```

The other two CLIs still take their inputs one flag at a time, and have no
notion of the `input/` / `output/` split — name the paths on both sides:

```bash
python run_rodan_pitch_finding.py \
  --image ../McGill_MS234-064/input/McGill_MS234-064.jpg \
  --ic-xml ../McGill_MS234-064/input/ic-session-McGill_MS234-064-page.xml \
  --staff-json ../McGill_MS234-064/input/McGill_MS234-064_stafflines.json \
  --output ../McGill_MS234-064/output/ --debug-viz

python render_ic_debug.py \
  --image ../McGill_MS234-064/input/McGill_MS234-064.jpg \
  --ic-xml ../McGill_MS234-064/input/ic-session-McGill_MS234-064-page.xml \
  --output ../McGill_MS234-064/output/ic_debug.jpg
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

Those labels are wider than the ~20–30px glyphs they belong to, so on a
densely notated page they overlap each other and cover the ink and markers
being checked. Both pitch-finding CLIs therefore write the overlay twice:

- `..._debug.jpg` — boxes, crosshairs, staff lines **and** all the labels
  above: what pitch was assigned and why.
- `..._debug_nolabels.jpg` — the same render with every caption suppressed,
  leaving only bbox, notehead crosshair and staff lines: for checking whether
  a marker sits on the right notehead and the right staff position.

Same scale for both, so a given page pixel is the same pixel in both files —
open them in two tabs and flip between them. `--debug-viz overlay.jpg` names
the pair `overlay.jpg` / `overlay_nolabels.jpg`.

`--debug-scale` (default 2.5) is also the knob for how crowded the labelled
copy is: label text grows with the *square root* of the scale, so raising it
gives more text pixels in absolute terms while each label covers less of the
page. `--debug-scale 5` roughly halves the page area a caption takes up
compared to `1`. The same applies to `render_ic_debug.py --scale`, whose
class-name labels are the longest of the three.

## Stave grouping

Staff-finding tags each fitted line with a `stave_id`, but it groups by y alone.
On a two-column page (`Breviarium_ad_usum`) the left column's four lines and the
right column's four sit at the same height, so they arrive as one stave of eight
lines — which `staff_io`'s `2 * (max_index - index)` numbering reads as a 14-step
staff. Every pitch on such a page comes out wrong, and both columns then compete
for one clef.

So `staff_regroup.py` rebuilds the grouping from the line geometry the same file
carries: column blocks from a coverage projection of the x spans, staves from
vertical gaps within a column, and `within_stave_index` from `round((y - y_top) /
line_gap)`. That last one also merges the two fragments a decorated initial
splits a line into, and leaves a hole at the index of a line that was never
detected instead of renumbering the lines below it.

On `Breviarium_ad_usum` this turns 13 mixed-column staves (one of 10 lines) into
20 single-column ones, 16 with all four lines and 3 correctly reporting 3 of 4.
Left-column pitches go from spanning f3–b4 to a3–f4, and each column resolves its
own clef. Single-column pages are unaffected: `McGill_MS234-064` output is
byte-identical either way. `--no-regroup` restores staff-finding's own grouping.

## Which algorithm

**`pitch_finder.py`** decomposes multi-note neumes. The neumes-cheatsheet CSV
tells us how many notes a neume class has and each note's melodic interval
(e.g. `neume.clivis2` = 2 notes, second one a step below). MEI's `@intm` is
measured from the *preceding* note, so those are accumulated into offsets from
the neume's first note: `neume.torculus22`'s `1S, -1S` is `[0, 1, 0]`, up a
second and back down to where it started. From there
**one anchor point on the glyph determines every note**: read that point's
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
  neume's first note, a top crop its highest, a full-bbox crop the middle of
  the span — and that is what ties the measured point to the interval table.
  Torculus gets one crop rule of its own that algorithm #2 does not have (its
  first notehead, found from the left band's own ink extent); everything else
  is shared, so algorithm #2 stays an independent baseline.
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
cd scripts && python -m pytest script_tests/ -v   # 94 passing
```

## Known limitations

- `CLEF_OCTAVE_REFERENCE` in `clef_rules.py` (C=4/F=3/G=4) is an unvalidated placeholder.
- Regrouping recovers the index of an undetected line from the line spacing, but
  if the *bottom* line of a stave is missing there is nothing to measure it
  against, and the step-0 anchor is off by whole steps for that stave.
- Column detection assumes some stave crosses the gap a decorated initial leaves
  inside a stave; a crop holding one interrupted stave and nothing else would
  have that stave split as if it were two columns.
- No human-verified pitch/MEI ground truth exists yet.
