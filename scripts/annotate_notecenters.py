"""
Interactive tool for hand-labeling ground-truth notehead *stave steps*.

Why a step and not a pixel: which clef governs a note, and the step-to-pitch
arithmetic once you have a step, are both fully deterministic (see
pitch_finder.py) -- there is nothing to learn there and no ambiguity for a
human to resolve. What the current heuristic gets wrong is finding the
notehead itself under a noisy/imprecise bbox, i.e. which line or space it
sits on -- exactly the thing a learned model (or a better heuristic) needs
real labels for.

Step convention (see staff_io.py, unchanged here): within a stave, the
bottom-most *detected* line is step 0, each line up is +2 steps, and the
space between two adjacent lines is the odd step between them -- so a line
is always an even step, a space always odd, matching "adjacent lines are a
third apart = 2 diatonic degrees". Steps extend past the detected lines too
(y_at_step extrapolates), which is exactly why a note sitting above/below
the stave is still labelable.

No clicking: each glyph opens with one integer step per expected note,
pre-filled from the current heuristic's own guess (rounded, from a real
find_pitches() pixel-anchor run) -- confirming a correct guess is then just
"don't touch anything, save". Digit keys 1-9 pick which note's step is
"active"; Up/Down nudge the active note's step by one, with the crosshair
moving live so the correction is entirely by eye. cv2's arrow-key codes are
inconsistent enough across platforms/builds that a fixed set of known codes
(see UP_KEYS/DOWN_KEYS) is checked via waitKeyEx rather than assumed; k/j do
the same thing as a fallback in case a given build's arrow codes aren't in
that set.

Each glyph's expected point count comes from neume_shapes' interval table.
Since the classifier's own guess at *what this glyph is* can be wrong too,
+/- add or remove a note if the count itself looks wrong for what's on the
page, rather than trusting the class name unconditionally.

The displayed crop is the vertically-expanded box worked out in
noise-exploration/ (full-stave span + MARGIN_STEPS half_gaps, a page-median-
height fallback for degenerate/sparse staves, unioned with the glyph's own
bbox as a last resort), drawn as an explicit rectangle -- orange if no
refinement fired for this glyph, red if it did -- with a bit of extra margin
around it so its edges are visible rather than being the crop's own border.
This is deliberately the same box a future model would be fed, so the human
label and the model's eventual input describe the same crop.

Usage (run directly in a terminal with a display -- this opens a real GUI
window, so it cannot run headless or inside a tool sandbox):

    python annotate_notecenters.py ../McGill_MS234-064

Controls (shown in the window's own footer too, one per line):
    Up / Down   nudge the active note's step by one (k/j also work)
    1-9         make that note (1st, 2nd, ...) the active one
    +           add a new note (step 0), making it active
    -           remove the active note
    c           reset this glyph's steps back to the heuristic guess
    s / Enter   save the current steps and move on
    n           skip this glyph (couldn't tell / bad image), no points saved
    b           go back to the previous glyph
    q           save progress and quit

Progress is written after every glyph to
<page>/labels/human_annotated_stave_steps.json, and reloaded on the next run
-- already-labeled or skipped glyphs are not shown again unless you page
back to them with 'b'. The filename says "human_annotated" (not just
"stave_steps") so it's never mistaken for a heuristic's own guess further
down the pipeline. This is a different file from the older
labels/notecenters.json (raw pixel positions, superseded by this step-based
scheme) -- that file is left alone, not migrated.
"""
import argparse
import json
import statistics
from pathlib import Path

import cv2
import numpy as np

from ic_io import parse_ic_xml
from staff_io import load_staves
from neume_shapes import load_neume_shapes
from page_inputs import resolve_page_inputs
from pitch_finder import find_pitches, assign_stave
from run_pitch_finding import DEFAULT_NEUME_CSV

ACTIVE_COLOR = (60, 200, 60)        # bright green: the note k/j currently adjusts
INACTIVE_COLOR = (0, 140, 255)      # orange: this glyph's other notes
REFERENCE_COLOR = (170, 170, 170)   # gray: the original heuristic guess, only
                                    # drawn where it differs from the current value
ORIGINAL_BOX_COLOR = (255, 60, 20)  # blue: the glyph's own IC bbox
EXPANDED_BOX_COLOR = (0, 140, 255)  # orange: expanded box, no refinement needed
REFINED_BOX_COLOR = (0, 0, 255)     # vivid red: expanded box needed refining
LINE_COLOR = (255, 0, 255)

# cv2.waitKeyEx()'s arrow-key codes vary by platform/build, so a known set is
# matched rather than assumed; k/j remain a fallback if a given build's codes
# aren't in this set. Covers macOS (Cocoa), Linux (GTK), and Windows builds.
UP_KEYS = {ord('k'), 63232, 65362, 2490368}
DOWN_KEYS = {ord('j'), 63233, 65364, 2621440}

MAX_DISPLAY_H, MAX_DISPLAY_W = 1400, 900   # popup window bound, before zoom caps
MIN_ZOOM, MAX_ZOOM = 1.0, 14.0
H_PAD_FRAC = 1.2        # horizontal crop padding, as a fraction of the glyph's own width
BOX_MARGIN_FRAC = 0.15  # extra crop margin beyond the expanded box, so its
                        # edges are visible rather than being the crop border

# Same constants and two-pass (median-height-fallback + bbox-union) logic as
# noise-exploration/visualize_vertical_expansion_fullpage.py -- see there for
# the reasoning; kept in sync by hand for now (nothing here depends on that
# script, this just duplicates its box math).
MARGIN_STEPS = 2.0
MEDIAN_FALLBACK_FRAC = 0.6


def full_line_span(stave, cx):
    """(min_y, max_y) across EVERY line of the stave, not just the ones whose
    own detected x-range covers cx -- a line that doesn't reach cx still
    contributes its y at the nearest x it does cover."""
    ys = []
    for line in stave.lines:
        x = min(max(cx, line.x_start), line.x_end)
        y = line.y_at_x(x)
        if y is not None:
            ys.append(y)
    return (min(ys), max(ys)) if ys else (None, None)


def expanded_boxes(glyphs, staves):
    """glyph_index -> (top, bottom, was_refined) for every glyph with an
    assigned stave. See module docstring."""
    local = {}
    for g in glyphs:
        stave, _flags = assign_stave(g, staves)
        if stave is None:
            continue
        cx = g.center_x
        half_gap = stave.half_gap_at_x(cx)
        stave_top, stave_bottom = full_line_span(stave, cx)
        local[g.index] = (g, stave_top - MARGIN_STEPS * half_gap, stave_bottom + MARGIN_STEPS * half_gap)

    heights = [b - t for _, t, b in local.values()]
    median_height = statistics.median(heights) if heights else 0.0
    threshold = MEDIAN_FALLBACK_FRAC * median_height

    out = {}
    for index, (g, top, bottom) in local.items():
        refined = (bottom - top) < threshold
        if refined:
            mid = (top + bottom) / 2
            top, bottom = mid - median_height / 2, mid + median_height / 2
        if top > g.uly or bottom < g.uly + g.nrows:
            refined = True
            top, bottom = min(top, g.uly), max(bottom, g.uly + g.nrows)
        out[index] = (top, bottom, refined)
    return out


def stave_for_glyphs(glyphs, staves):
    return {g.index: assign_stave(g, staves)[0] for g in glyphs}


def heuristic_steps(glyphs, staves, shapes, image):
    """glyph_index -> [rounded_step, ...] from a real pixel-anchor find_pitches run."""
    results = find_pitches(glyphs, staves, shapes, image)
    return {r.glyph_index: [round(nc.stave_step) for nc in r.note_components
                            if nc.stave_step is not None]
           for r in results}


def expected_count(shapes, class_name: str) -> "int | None":
    intervals = shapes.intervals_for(class_name)
    return len(intervals) if intervals is not None else None


def step_description(step: int) -> str:
    return f"{step} ({'line' if step % 2 == 0 else 'space'})"


def labels_path(inputs) -> Path:
    return inputs.page_dir / "labels" / "human_annotated_stave_steps.json"


def load_progress(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_progress(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


class Annotator:
    def __init__(self, inputs, glyphs, staves, shapes, image, progress):
        self.inputs = inputs
        self.glyphs = [g for g in glyphs if g.state != "UNCLASSIFIED"
                      and not shapes.is_pitchless(g.class_name) and not shapes.is_clef(g.class_name)]
        self.staves = staves
        self.shapes = shapes
        self.image = image
        self.progress = progress
        self.guesses = heuristic_steps(glyphs, staves, shapes, image)
        self.boxes = expanded_boxes(glyphs, staves)
        self.stave_of = stave_for_glyphs(glyphs, staves)
        self.i = 0
        self.initial_steps = []
        self.current_steps = []
        self.active = 0
        self.path = labels_path(inputs)

    def _entry_key(self, g):
        return str(g.index)

    def _pending(self):
        """Indices into self.glyphs not yet labeled or skipped."""
        return [i for i, g in enumerate(self.glyphs)
                if self._entry_key(g) not in self.progress]

    def current(self):
        return self.glyphs[self.i]

    def _initial_steps_for(self, g) -> list:
        """Heuristic guess, padded/truncated to the class's expected note
        count (or the guess's own length if the class is unrecognized) --
        missing entries default to 0 rather than being left unlabeled, since
        there's no click to skip via anymore."""
        guess = self.guesses.get(g.index, [])
        n = expected_count(self.shapes, g.class_name)
        if n is None:
            n = max(1, len(guess))
        return (guess + [0] * n)[:n]

    def _load_glyph_state(self):
        g = self.current()
        self.initial_steps = self._initial_steps_for(g)
        self.current_steps = list(self.initial_steps)
        self.active = 0

    def render(self):
        g = self.current()
        stave = self.stave_of.get(g.index)
        box_top, box_bottom, refined = self.boxes.get(g.index, (g.uly, g.uly + g.nrows, False))
        box_height = box_bottom - box_top

        v_margin = box_height * BOX_MARGIN_FRAC
        h_pad = round(g.ncols * H_PAD_FRAC) + 15
        x0 = max(0, g.ulx - h_pad)
        x1 = g.ulx + g.ncols + h_pad
        y0 = max(0, round(box_top - v_margin))
        y1 = round(box_bottom + v_margin)

        crop = self.image[y0:y1, x0:x1].copy()
        ch, cw = crop.shape[:2]
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, MAX_DISPLAY_H / max(ch, 1), MAX_DISPLAY_W / max(cw, 1)))
        canvas = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)

        def to_canvas(px, py):
            return round((px - x0) * zoom), round((py - y0) * zoom)

        for st in self.staves:
            for line in st.lines:
                pts = []
                for cx in range(x0, x1):
                    y = line.y_at_x(cx)
                    if y is not None and y0 <= y <= y1:
                        pts.append(to_canvas(cx, y))
                if len(pts) > 1:
                    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, LINE_COLOR, 1, cv2.LINE_AA)

        def draw_rect(top, bottom, color, thickness):
            p1, p2 = to_canvas(g.ulx, top), to_canvas(g.ulx + g.ncols, bottom)
            cv2.rectangle(canvas, p1, p2, color, thickness, cv2.LINE_AA)

        draw_rect(box_top, box_bottom, REFINED_BOX_COLOR if refined else EXPANDED_BOX_COLOR, 2)
        draw_rect(g.uly, g.uly + g.nrows, ORIGINAL_BOX_COLOR, 1)

        cx_query = g.center_x

        def draw_step_marker(step, color, size, thickness, label=None):
            if stave is None:
                return
            y = stave.y_at_step(cx_query, step)
            if y is None:
                return
            cx, cy = to_canvas(cx_query, y)
            cv2.drawMarker(canvas, (cx, cy), color, cv2.MARKER_CROSS, size, thickness)
            if label:
                cv2.putText(canvas, label, (cx + 8, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        for i, orig in enumerate(self.initial_steps):
            if i >= len(self.current_steps) or orig != self.current_steps[i]:
                draw_step_marker(orig, REFERENCE_COLOR, 10, 1)

        for i, step in enumerate(self.current_steps):
            is_active = i == self.active
            draw_step_marker(step, ACTIVE_COLOR if is_active else INACTIVE_COLOR,
                            20 if is_active else 16, 3 if is_active else 2,
                            label=f"{i + 1}:{step}")

        n_exp = expected_count(self.shapes, g.class_name)
        done = len(self.glyphs) - len(self._pending())
        steps_desc = "  ".join(
            f"[{i + 1}]{step_description(s)}" + ("*" if i == self.active else "")
            for i, s in enumerate(self.current_steps)
        )
        footer_lines = [
            f"[{done}/{len(self.glyphs)}] glyph {g.index}  class={g.class_name}  "
            f"expected points={n_exp if n_exp is not None else '?'}",
            f"notes: {steps_desc}   (* = active)",
            "Up / Down : nudge the active note's step by one (k/j also work)",
            "1-9       : make that note (1st, 2nd, ...) the active one",
            "+         : add a new note (step 0), making it active",
            "-         : remove the active note",
            "c         : reset this glyph's steps back to the heuristic guess",
            "s / Enter : save the current steps and move on",
            "n         : skip this glyph (couldn't tell / bad image)",
            "b         : go back to the previous glyph",
            "q         : save progress and quit",
        ]
        footer = np.full((18 + 18 * len(footer_lines), canvas.shape[1], 3), 255, dtype=np.uint8)
        for i, line in enumerate(footer_lines):
            cv2.putText(footer, line, (6, 14 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        full = np.vstack([canvas, footer])
        return full

    def save_current(self, steps, skipped: bool):
        g = self.current()
        self.progress[self._entry_key(g)] = {
            "class_name": g.class_name,
            "steps": list(steps),
            "accepted_heuristic": list(steps) == self.initial_steps,
            "skipped": skipped,
        }
        save_progress(self.path, self.progress)

    def advance(self, direction: int):
        pending = self._pending()
        if direction > 0:
            upcoming = [i for i in pending if i > self.i]
            self.i = upcoming[0] if upcoming else (pending[0] if pending else min(self.i + 1, len(self.glyphs) - 1))
        else:
            self.i = max(0, self.i - 1)
        self._load_glyph_state()

    def run(self):
        pending = self._pending()
        if not pending:
            print("Nothing left to label.")
            return
        self.i = pending[0]
        self._load_glyph_state()

        win = "annotate_notecenters"
        cv2.namedWindow(win)

        while True:
            cv2.imshow(win, self.render())
            key = cv2.waitKeyEx(20)

            if key == ord('q'):
                break
            elif key in (ord('s'), 13):
                self.save_current(self.current_steps, skipped=False)
                self.advance(1)
            elif key == ord('n'):
                self.save_current([], skipped=True)
                self.advance(1)
            elif key == ord('c'):
                self.current_steps = list(self.initial_steps)
            elif key in UP_KEYS:
                self.current_steps[self.active] += 1
            elif key in DOWN_KEYS:
                self.current_steps[self.active] -= 1
            elif key == ord('+'):
                self.current_steps.append(0)
                self.active = len(self.current_steps) - 1
            elif key == ord('-') and len(self.current_steps) > 1:
                self.current_steps.pop(self.active)
                self.active = min(self.active, len(self.current_steps) - 1)
            elif ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if idx < len(self.current_steps):
                    self.active = idx
            elif key == ord('b'):
                self.advance(-1)
            if not self._pending() and key != ord('b'):
                print("All glyphs labeled or skipped.")
                break

        cv2.destroyAllWindows()
        print(f"Progress saved to {self.path} ({len(self.progress)}/{len(self.glyphs)} done)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("page", type=Path)
    parser.add_argument("--no-regroup", dest="regroup_staves", action="store_false",
                         help="Trust the staff JSON's own stave_id / within_stave_index instead of "
                             "re-deriving them from line geometry -- needed for a page whose grouping "
                             "was fixed by hand in fix_stafflines.py (see its manually_grouped flag). "
                             "See run_pitch_finding.py's own --no-regroup for the same option.")
    args = parser.parse_args()

    inputs = resolve_page_inputs(args.page)
    glyphs = parse_ic_xml(inputs.ic_xml)
    staves = load_staves(inputs.staff_json, regroup=args.regroup_staves)
    shapes = load_neume_shapes(DEFAULT_NEUME_CSV)
    image = cv2.imread(str(inputs.image))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {inputs.image}")

    progress = load_progress(labels_path(inputs))
    Annotator(inputs, glyphs, staves, shapes, image, progress).run()


if __name__ == "__main__":
    main()
