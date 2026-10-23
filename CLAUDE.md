# CLAUDE.md

Standalone pitch-finding prototype for square-notation chant manuscripts. Input
is one page's **IC (Interactive Classifier) XML** + **staff-finding JSON** +
**page image**; output is an intermediate JSON giving every glyph a stave, a
staff position, and a pitch (or the reason there isn't one), plus debug overlays.

Two independent algorithms live here: `pitch_finder.py` (**algorithm #1**, ours,
decomposes multi-note neumes) and `rodan_pitch_finder.py` (**algorithm #2**, a
reimplementation of Rodan's heuristic pitch finder, one pitch per glyph). The only
ground truth is hand-labeled **stave steps** on 13 Hufnagel pages (2607 notes, in
`<page>/labels/human_annotated_stave_steps.json`). Algorithm #1 scores 81.6% per
note against it. Algorithm #2 is unscored, and no pitch-name or MEI ground truth
exists. See `README.md` for the narrative version, `PLAN-neume-anchors.md` for
design rationale, `PROGRESS.md` for status, open questions and who did what, and
`ANNOTATION_PROGRESS.md` for per-page numbers.

## Commands

```bash
conda activate pitch-finding          # needs numpy, opencv-python, pytest
cd scripts

python run_pitch_finding.py ../McGill_MS234-064 --debug-viz     # algorithm #1
python -m pytest script_tests/ -v
```

Verify deps before running anything (`python -c "import cv2, numpy"`) — the
`pitch-finding` env has been found missing `cv2`/`pytest`. On this Intel Mac a
plain `pip install opencv-python` tries to build OpenCV 5 from source and fails,
because conda's Python reports macOS 10.16 and pip skips the newer wheels.
`pip install "numpy<2" "opencv-python==4.9.0.80" pytest` works. Use the
non-headless package, because the labeling GUIs need `cv2.imshow`.

Page folders: the 13 labeled Hufnagel pages (listed as `REAL_LABELED_PAGES` in
`experiments/v1-hfngl/data.py`; `McGill_MS234-064` is one of them), plus three
unlabeled samples: `Breviarium_ad_usum` and `Breviarium_ad_usum_crop` (two
columns, the regrouping test case) and `GentAnt1475_0017_AC_rightcrop` (smaller
notation).

## Layout convention

A page folder holds `input/` (image, IC XML, `*stafflines*.json`) and takes
artifacts in `output/`. Labeled pages also have `labels/`. Discovery never looks
inside `output/`, which is what lets a page be re-run without a previous run's
renders becoming candidate inputs. Older folders keep all three files flat at the
top level; discovery still handles those.

Labels are keyed by glyph **index**, meaning the order of `<glyph>` elements in
the IC XML. If a labeled page's IC XML is replaced or reclassified, its labels
stop matching the glyphs, and nothing raises an error. Check that `class_name`
still matches before trusting any score.

---

## The `run_pitch_finding.py` pipeline (algorithm #1)

This is the current main entry point. `main()` resolves inputs, `run()` does the
work. Every module it touches, in call order:

### 1. `page_inputs.py` — resolve the three inputs

`resolve_page_inputs(page, image, ic_xml, staff_json)` → `PageInputs(image,
ic_xml, staff_json, page_dir, input_dir, output_dir)`.

`page` may be the page folder, its `input/` folder, or any file inside either
(passing the image is how you disambiguate a folder holding two pages — it also
narrows the XML/JSON candidates by the image's stem). Discovery is deliberately
**strict**: each kind must resolve to exactly one candidate or it raises, naming
what it found and the flag that overrides it. `--image` / `--ic-xml` /
`--staff-json` bypass discovery entirely and may point outside the page folder.

Only `run_pitch_finding.py` uses this module; the other two CLIs take every path
by flag.

### 2. `viz_utils.py` — resolve artifact paths *before* any work

`resolve_output_path()` lets `--output` name either a file or a directory
(a directory gets `<image stem>_pitch_finding.json`); `resolve_debug_viz_path()`
maps `--debug-viz` (bare flag, `auto`, or a filename) to a path and validates the
extension. Both run up front on purpose: a bad `--debug-viz` extension must be
reported now, not after the render has already allocated hundreds of MB.

### 3. `ic_io.py` — parse the IC XML

`parse_ic_xml(path)` → `list[Glyph]`, one per `<glyph>` element:
`index, ulx, uly, nrows, ncols, class_name, confidence, state`, plus
`center_x`/`center_y`/`lry` properties. Malformed elements are skipped with a
warning rather than raising. `state == "UNCLASSIFIED"` marks a text bbox that
never went through the neume classifier; `index` is the stable id every
downstream artifact refers to.

### 4. `staff_io.py` + `staff_regroup.py` — parse and regroup staff lines

`load_staves_with_report(path, regroup=True)` → `(list[Stave], RegroupReport)`.
The runner prints `report.summary()`.

**`staff_regroup.py` runs first** (unless `--no-regroup`). Staff-finding's own
`stave_id` groups by y alone, so on a two-column page the left column's four
lines and the right column's four merge into one eight-line "stave" — which
`staff_io`'s `2 * (max_index - index)` numbering then reads as a 14-step staff,
making every pitch on the page wrong. `regroup_entries()` rebuilds the grouping
from the line geometry the same file carries:

1. **Columns** — coverage projection of x spans; a gutter is a wide interior band
   almost nothing covers. Counting *coverage* (not just uncovered x) is what
   keeps one full-width line from bridging the columns, and what distinguishes a
   gutter from the gap a decorated initial leaves inside a stave.
2. **Staves** — within a column, cut where the vertical step exceeds
   `STAVE_CUT_GAPS` (2.0) line gaps. The gap comes from `estimate_line_gap()`
   measured on the page, not from `scale_unit`, which does not track it.
3. **Line index** — `round((y - y_top) / gap)`, not rank. This collapses the two
   fragments an initial splits a line into, and **leaves a hole** at the index of
   an undetected line instead of renumbering the lines below it (which would
   shift every pitch on that stave by a step).

Before grouping, entries with no `y_values` are dropped (zero-width detections
used to crash it).

**Hand-fixed pages need `--no-regroup`.** On five pages (`CantusMA1537_p22`,
`MS025a-01`, `MS025a-02`, `MS025b-01`, `MS234_p005`), `fix_stafflines.py` wrote
explicit `stave_id`/`within_stave_index` values and a `manually_grouped` flag into
the staff JSON. Nothing reads that flag. With `regroup=True` (the default), those
values are rebuilt from the line geometry and the fix is lost without any warning.

**`staff_io.py`** then builds `Stave` objects. Step convention: bottom-most
*detected* line is step 0, each line index 2 steps apart. `pitch_finder` queries
staves only through these methods:

| Method | Used for |
|---|---|
| `step_at_x(x)` | `(step, y)` pairs of every line covering x; collapses same-index fragments |
| `continuous_step_at_y(x, y)` | fractional step from a pixel row — the core lookup; flags `sparse_stave_lines` when only one line is available |
| `y_at_step(x, step)` | exact inverse, converts a computed step back to pixels for the overlay |
| `y_span_at_x` / `half_gap_at_x` / `nearest_line_distance` | stave assignment |

### 5. `neume_shapes.py` — the interval table

`load_neume_shapes(csv_path)` → `NeumeShapeTable`, parsed from
`neumes-cheatsheet/csv-square_notation_neume_level_newest.csv` (the
`DEFAULT_NEUME_CSV` constant in the runner; `--neume-csv` overrides).

The CSV's MEI `@intm="1S"` values are intervals **from the preceding note**, so
`_extract_intervals` accumulates them into offsets from the neume's first note:
`torculus22`'s `1S, -1S` → `[0, 1, 0]`. Reading them as absolute offsets inverts
the contour of every neume that changes direction — `test_neume_shapes.py` pins
this against an independent name-derived oracle.

`intervals_with_source(class_name)` returns `(intervals, SOURCE_*)` after up to
four lookups, weakest evidence last: exact CSV row → curated
`_BUILTIN_INTERVALS` (the repeated-note neumes, which resolve to **one** note,
not N) → CSV with a trailing variant letter dropped (`clivis2a` → `clivis2`) →
decoded from the class name. Only the exact hit is documented ground truth, so
the source comes back with it and `pitch_finder` flags anything below it.
`is_pitchless()` / `is_clef()` / `clef_pname()` classify the rest.

### 6. `glyph_pixels.py` — find the notehead (`--anchor-mode pixel`, the default)

The runner loads the image with `cv2.imread` only in `pixel` mode, and raises a
clear `FileNotFoundError` if it can't. Note the plumbing: `find_pitches` has no
`anchor_mode` parameter — **passing the image *is* pixel mode**, passing `None`
is `bbox` mode.

`pitch_finder` calls three functions here:

- `average_punctum(glyphs)` — Rodan's proxy for one notehead's **width**, sizing
  every crop. Measured once per page. Zero (no punctum/virga on the page) falls
  back to geometry rather than anchoring on a 1px sliver.
- `notehead_height(glyphs)` — median height of punctum/inclinatum, i.e. how
  **deep** one notehead is. Rodan uses `average_punctum` for both dimensions; on
  `McGill_MS234-064` that is 35px against a 25px-tall notehead, so a band "one
  notehead deep" reached 40% too far into the ligature stroke above the head.
- `reference_point(image, glyph, avg_punctum, notehead_h=...)` →
  `ReferencePoint(x, y, region)` or `None`. Otsu-binarizes a per-class crop of
  the bbox and takes its ink-weighted row centroid.

The **crop rules** (`reference_region`) are the substance. Their whole purpose is
to exclude ink that isn't the reference notehead — a virga's stem, a podatus's
upper head, an F-clef's left dots:

| Class | Region | Notes |
|---|---|---|
| ascending ligatures (`podatus`, `pescephalicus`, `scandicus`, `torculus`) | `REGION_BOTTOM_LEFT` | `_head_ink_region(from_bottom=True)` — extended rules only |
| `clivis*` | `REGION_TOP` | `from_bottom=False`, depth × `TOP_HEAD_DEPTH_FRACTION` (0.75) |
| `neume.virga` | `REGION_TOP` | fixed band off the bbox top (Rodan's rule) |
| `BOTTOM_LEFT_CLASSES` | `REGION_BOTTOM_LEFT` | fixed band off the bbox bottom (Rodan's rule) |
| `clef.f*` | `REGION_F_CLEF_RIGHT` | right half, trimmed to its own ink |
| everything else | `REGION_FULL` | full height, width capped at `avg_punctum` |

`_head_ink_region` locates the head by **ink**, not by a bbox edge: it segments
the left column band's row profile and keeps the outermost run heavy enough to be
a notehead (`_notehead_runs`, `HEAD_MASS_FRACTION`), then clamps it to one
notehead's depth. Two failures motivate it — the bbox bottom is set by whatever
hangs lowest (a podatus's ligature stroke, a torculus's *third* note), and staff
lines crossing the band are ink too and drag a fixed-window centroid toward
themselves.

**`extended_rules` is the seam between the two algorithms.** The first-head rules
(`FIRST_HEAD_INK_CLASSES`, `TOP_HEAD_INK_CLASSES`) are on unconditionally in
`reference_point` and off in `reference_row`. Rodan gets one pitch per glyph and
can settle for a centroid belonging to no particular notehead; a caller placing
three notes off one point cannot. Keeping them off `reference_row` is what leaves
algorithm #2 an independent baseline instead of a copy of this module's opinions
— **do not "unify" the two paths.**

`reference_point` also differs from `reference_row` in returning `None` rather
than a silent `0.0` when nothing was measurable, and in reporting x at the crop
band's *center* (Rodan reads at the bbox's left edge).

### 7. `pitch_finder.py` — the algorithm

`find_pitches(glyphs, staves, shapes, image=None)` → `list[GlyphResult]`, in two
passes.

**Pass 1 — stave assignment, anchoring, decomposition** (clef-independent):

1. `state == "UNCLASSIFIED"` or `shapes.is_pitchless()` → `pitchless_symbol`, no
   stave attempted.
2. `assign_stave()` — closest stave (by `nearest_line_distance`) whose line span
   at the glyph's center x, padded by `STAVE_MARGIN_STEPS` (2) × that stave's own
   local `half_gap_at_x`, covers the center y. None qualifying →
   `missing_staff`. The margin is per-stave local spacing, not a page constant.
3. `reference_point()` if in pixel mode; `None` back → flag
   `pixel_anchor_unavailable` and fall through to geometry.
4. `_decompose()` — **one anchor point determines every note.** Get the interval
   list from `shapes.intervals_with_source()`, then:
   - `_anchor_from_pixels` — read `continuous_step_at_y` at the measured centroid.
     If the crop's own x band falls off the end of the detected lines (a clef left
     of where line-fitting starts), retry at the bbox center x and flag
     `anchor_x_fell_back_to_center` — lines are near-horizontal, so the row is
     the part that matters.
   - `_anchor_interval(region, intervals)` says **which note** the point is:
     `REGION_BOTTOM_LEFT` → interval 0 (the first note — correct even for a
     torculus that ends below where it started, where `min(intervals)` is not);
     `REGION_TOP` → `max(intervals)`; `REGION_FULL` → the span midpoint, a
     fractional interval, honest about belonging to no single note.
   - each note's step is `anchor.stave_step + (interval - anchor.interval)`, and
     its `center_y` comes back from `Stave.y_at_step` for the overlay.
   - `_anchor_from_bbox_span` is the `bbox`-mode path: map the bbox's top and
     bottom edges onto the interval span. No pixels needed, but it takes ink
     extremes for notehead centers, so any stem or tail biases every note.

   Classes with no interval list at all fall back to a single note flagged
   `approximate_unknown_shape` (drawn amber); a name-decoded list is flagged
   `shape_from_class_name`. `_decompose` returns `None` only when the stave has
   no line coverage at that x (`no_line_coverage`).
5. Clefs additionally get `clef_rules.clef_octave_for()` and are registered in
   `clefs_by_stave`.

**Pass 2 — clef resolution:** for each pending glyph, take the nearest clef **on
its own stave**, preferring one to the left (already in effect reading
left-to-right; otherwise flag `clef_after_glyph`). A stave with no clef reports
`missing_clef` rather than borrowing a neighbour's — this is a deliberate
difference from algorithm #2. Then `clef_rules.step_to_pitch(nc.stave_step -
clef_step, clef_pname, clef_octave)` per note component.

Every result records its `anchor` (source, region, x, y, stave_step, interval),
so a neume whose pitches are uniformly off can be traced to a bad anchor
*position* vs. a bad anchor *role*.

### 8. `clef_rules.py` — step → pitch

`step_to_pitch(step_delta_from_clef, clef_pname, clef_octave)` rounds the delta
to a whole step and walks `NOTE_LETTERS`. `clef_octave_for(pname)` reads
`CLEF_OCTAVE_REFERENCE` (C=4/F=3/G=4) — an **unvalidated placeholder**, flagging
`clef_octave_unconfigured` for an unlisted letter. Letter names and step
distances are the trustworthy part of the output; absolute octave numbers ride on
this assumption.

### 9. Output + `viz_utils.py` — the debug overlays

The JSON records the input paths, `anchor_mode`, and every `GlyphResult`. The
runner then prints a reason breakdown and `_print_anchor_summary()`, which counts
`pixel_anchor_unavailable` and `anchor_x_fell_back_to_center` **out loud** — a
silent fall back to geometry on exactly the classes the crops exist for would
look identical to a working run.

With `--debug-viz`, `_render_debug_viz` runs **twice** — labelled, then
`labels=False` via `unlabeled_variant_path()` (`_nolabels`), from a second image
load (the labelled pass paints white-backed text over its own boxes, so nothing
can be peeled off afterwards). Same scale in both, so a given page pixel is the
same pixel in both files. Draw order matters: staff lines → boxes+labels →
markers in a separate pass (a neighbour's label would otherwise paint over a
marker).

`viz_utils` supplies `load_scaled_image`, `draw_stafflines` (traces the fitted
y_values polyline, so curvature is visible; tags lines `s<id>/<step>`),
`draw_labeled_box`, `draw_note_center` (crosshair, halo-under-color so it
survives landing on ink; multi-note labels start at the bbox's right edge to stay
off the glyph's ink), `write_image` (fails loudly — `cv2.imwrite` returns `False`
instead of raising), and `label_font_scale` (text grows as `scale ** 0.5`, which
is what makes `--debug-scale` a *crowding* knob rather than just a file-size one).

Colors: green = pitch found, amber = shape not CSV-backed, blue = clef, grey =
pitchless, red = `missing_clef`/`missing_staff`/`no_line_coverage`.

### Flags on `run_pitch_finding.py`

| Flag | Effect |
|---|---|
| `--anchor-mode {pixel,bbox}` | `pixel` (default) reads ink centroids; `bbox` is geometry-only and needs no image |
| `--no-regroup` | trust staff-finding's own grouping (wrong on two-column pages) |
| `--debug-viz [path]` | render the overlay pair; bare = named after `--output` |
| `--debug-scale` | default 2.5; also the label-crowding knob |
| `--output` | file or directory |
| `--image` / `--ic-xml` / `--staff-json` | override discovery |
| `--neume-csv` | override the cheatsheet CSV |

---

## Other scripts

### `rodan_pitch_finder.py` + `run_rodan_pitch_finding.py` (algorithm #2)

`find_pitches_rodan(glyphs, staves, image)` — one pitch per glyph, no
decomposition. Shares `ic_io`, `staff_io`, `clef_rules`, `viz_utils`, and
`glyph_pixels` (via `reference_row`, i.e. **without** the extended crop rules).
Deliberately different from algorithm #1 in three ways worth keeping:

- **Stave assignment**: three tiers — bbox intersection → y-margin fallback
  (`GET_STAFF_MARGIN` × `avg_punctum`, nearest by x) → optional forced-nearest
  (off, matching Rodan's `always_find_staff_no=False`).
- **Discretization**: `_stave_position` rounds the continuous step, which is
  mathematically equivalent to Rodan's `space_proportion=0.5`; clefs snap to the
  nearest *line* only. It clamps x into the stave's line-covered range, and
  extrapolates past the outermost real line rather than clamping (our
  staff-finding synthesizes no ledger lines, so a literal port would flatten
  every out-of-staff note onto the edge line).
- **Clef propagation**: one "current clef" walks the **whole page** in
  `(stave_id, ulx)` reading order, so a stave with no clef silently inherits the
  previous one — where algorithm #1 reports `missing_clef`.

The CLI takes every path by flag (no `page_inputs`, no `input/`/`output/`
notion), and writes `_rodan_pitch_finding.json` plus the same overlay pair.

### `render_ic_debug.py`

Draws IC's raw classification only — no staff or pitch logic — so IC's accuracy
can be judged on its own. Colors by category so a cross-category
misclassification shows up by color alone. Hides text/UNCLASSIFIED bboxes unless
`--show-text`; `--show-confidence` appends the score.

### `render_mei.py` (repo root)

Unrelated standalone tool: overlays `pname` letters from an existing MEI file
under each neume. Needs PIL, its own MEI input, and is not part of the pipeline.

### `tools/iiif_dl.py`

Downloads page images from a IIIF manifest (Presentation API v2/v3 and IIP).
`--manifest URL_OR_PATH`, plus `--output-dir`, `--format {jpg,tiff}`,
`--pages 1-3,7`, `--max-dim`, `--timeout`, `--user-agent`. Sends a browser UA by
default because some hosts (Gallica/BnF) 403 the `python-requests` default. Needs
`requests`, `tqdm`, `Pillow`, `cv2`. Raw downloads land in `data/`
(`data/Hufnagel/`, `data/Square/`). `tools/` is committed, but `data/*` is
gitignored.

### Ground-truth tooling (GUIs: need a display, cannot run headless)

- **`annotate_notecenters.py <page> [--no-regroup]`** steps through each glyph.
  It pre-fills one integer stave step per expected note from algorithm #1
  (`heuristic_steps()`), and the human nudges them with Up/Down (k/j) and adds or
  removes notes with `+`/`-`. Results are saved to
  `labels/human_annotated_stave_steps.json` after each glyph. The crop it shows
  comes from `expanded_boxes()`, which is the same crop the CNN is fed.
  `heuristic_steps()` is also the function every heuristic-accuracy number in the
  repo is computed from.
- **`fix_stafflines.py <page>`** lets you drag, add and delete staff lines and
  assign each one to a stave group. It saves explicit grouping plus the
  `manually_grouped` flag (see the `--no-regroup` note under section 4).
- **`check_stafflines.py <page>`** is not a GUI. It flags line pairs that cross or
  sit closer than a fraction of `scale_unit`. Run it after `fix_stafflines.py`.
- **`score_stave_steps.py [page ...]`** is not a GUI. It scores `heuristic_steps()`
  against the labels for each page and prints a total. With no arguments it
  scores every folder that has `labels/`. It reports **per note** (position-wise
  match over labeled notes) and **per glyph** (whole step list equal) as separate
  columns. It also reports "wrong count", the glyphs whose note count differs
  from the label. It reads the staff JSON's `manually_grouped` flag to choose
  `regroup=False`. It counts glyphs whose `class_name` differs from the label's,
  and warns about them instead of scoring silently. Its two helpers,
  `score_glyph` and `is_manually_grouped`, are unit-tested.

The three GUIs have no unit tests.

### `experiments/v1-hfngl/` — stave-step CNN

Has its own README. It is a per-glyph CNN (regression and 10-way classifier) for
**single-note** glyphs, trained on the 13 labeled pages. It reuses
`annotate_notecenters.expanded_boxes`/`heuristic_steps` and `staff_io` from
`scripts/`. The comparison it is built for is accuracy under uncorrected staff
lines: `staff_originals/` holds the true pre-correction staff geometry of the 5
hand-fixed pages. Checkpoints are committed (`*.pt` is gitignored except
`experiments/*/checkpoints/*.pt`). It needs `torch`, which the `pitch-finding`
env does not have.

### Data folders

- `bbox-ground-truths/`: Gen's neume-splitting bbox annotations (18 folios, 12
  Hufnagel and 6 square). The labeled page folders were built from them. Square
  folios have no page folders yet.
- `staff_finding_rerun/`: a fresh mothra staff-finding run on the 13 labeled
  pages (`detected_stafflines/` JSON + `_viz/` PNGs). Raw per-page pipeline
  directories are gitignored.

### `script_tests/`

Unit tests for the nine library modules plus `score_stave_steps`'s metric
(not the other CLIs or `viz_utils`), run with pytest from `scripts/`; 157 pass. `test_neume_shapes.py` is the one worth
knowing about: it cross-checks the whole CSV
interval table against intervals derived independently from the class names,
which is what pins the accumulate-don't-read-absolute convention permanently.

---

## Working notes

- **Measure stave steps against the labels. Pitch names are still predictions.**
  Score any change to algorithm #1 with `python score_stave_steps.py` against the
  2607 labeled steps, before and after. Quote per-note and per-glyph numbers
  separately (ANNOTATION_PROGRESS.md once mixed them). Pitch names and octaves
  have no ground truth, so compare those using the debug images.
- **Never silently degrade.** The established pattern is to flag and count:
  `pixel_anchor_unavailable`, `anchor_x_fell_back_to_center`,
  `approximate_unknown_shape`, `shape_from_class_name`, `sparse_stave_lines`,
  `clef_after_glyph`, `clef_octave_unconfigured`. A fallback that looks like a
  working run is the failure mode this codebase is built against.
- **The comments carry measurements.** Constants like
  `TOP_HEAD_DEPTH_FRACTION = 0.75`, `HEAD_MASS_FRACTION = 0.50`,
  `STAVE_CUT_GAPS = 2.0` are documented with the page data that chose them and
  the plateau they sit on. Re-measure before changing one.
- Known limitations are listed at the end of `README.md` — placeholder clef
  octaves, a missing *bottom* stave line having nothing to measure against,
  column detection assuming some stave bridges an initial's gap.
