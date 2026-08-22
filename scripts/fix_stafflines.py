"""
Interactive GUI for manually correcting a page's raw staff-line list before
staff_regroup groups it into staves.

Why this exists: for a handful of manuscripts (CantusMA1537_p22 and the
MS025a/MS025b/MS234 folios) the automatic staff-line detection is bad enough
-- either no real annotation existed and staff-finding was self-run and came
out fragmented/duplicate-heavy, or Gen's own boxes converted into lines still
regroup into garbage (dozens of near-empty staves) -- that annotate_notecenters
.py's heuristic guess is wrong on most glyphs. Fixing the underlying lines
once, here, is cheaper than correcting every single glyph's step by hand
afterward.

Each line is edited as a straight segment (x_start, y_start)-(x_end, y_end),
regardless of whatever curvature its original fit had -- these are already
crude/wrong lines being replaced by hand, not lines worth preserving
subtlety in. Saving flattens each line back into the centerline_page{x_start,
x_end, y_values} shape staff_regroup.regroup_entries() / staff_io.load_staves()
expect, linearly interpolating y between the two endpoints.

Grouping lines into staves: correcting line *positions* isn't enough on its
own -- staff_regroup's automatic grouping cuts a new stave wherever the
vertical gap between consecutive lines exceeds a fixed multiple of the
page's own line gap, and on a manuscript whose staves sit close together
that threshold can land wrong (measured, not hypothetical: one page had two
real inter-stave gaps just a few px under the cutoff, silently merging 3
physical staves' worth of lines into one). So this tool computes its own
starting guess at the grouping the same way, then gives every line an
explicit, always-visible group number (drawn right on the line, not just on
hover) that a digit key reassigns directly. Saving writes the resulting
stave_id / within_stave_index directly into the file (not null) plus a
manually_grouped quality flag -- but that only sticks if the caller loads
with regroup=False (run_pitch_finding.py's --no-regroup, or
annotate_notecenters.py's --no-regroup); regroup=True (the default)
re-derives grouping from geometry every time regardless of what's saved
here.

Controls (also shown in the window's own footer):
    left-drag near an endpoint     move that endpoint (free 2D movement)
    left-drag near a line's body   move the whole line vertically
    right-click near a line        delete that line
    a, then left-drag               add a new line (click-drag defines it)
    0-9, while hovering a line       assign that line directly to group 0-9
    g, while hovering a line        give that line a fresh, unused group
                                     number (for a group beyond 0-9, or to
                                     split it off before re-typing others
                                     into it)
    hover a line or endpoint        shows its id and group near the cursor
    +/-                             zoom in/out (centered on cursor)
    arrow keys                      pan (once zoomed in)
    s                                save
    r                                reset to the file's original lines
    q                                save and quit

--highlight draws specific line ids (as printed by check_stafflines.py) in
vivid red so a flagged pair is easy to spot instead of having to hover over
every line hunting for it by id.

Usage:
    python fix_stafflines.py ../CantusMA1537_p22
    python fix_stafflines.py ../CantusMA1537_p22 --highlight line0024,line0097
    python fix_stafflines.py ../MS025a-02 --image ../MS025a-02/input/foo.jpg \\
        --staff-json ../MS025a-02/input/foo_stafflines.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import page_inputs as pi
from staff_regroup import estimate_line_gap, STAVE_CUT_GAPS, _line_indices

MAX_DISPLAY_H, MAX_DISPLAY_W = 1400, 1400  # window bound at zoom=1 (fit-to-window)
MIN_ZOOM, MAX_ZOOM = 1.0, 12.0
ZOOM_STEP = 1.25
PAN_FRACTION = 0.15  # of the current viewport, per arrow-key press

HANDLE_RADIUS_SCREEN_PX = 10
BODY_HIT_SCREEN_PX = 6

LINE_COLOR = (255, 0, 255)
HANDLE_COLOR = (0, 220, 0)
NEW_LINE_PREVIEW_COLOR = (0, 220, 220)
HOVER_COLOR = (0, 140, 255)
HIGHLIGHT_COLOR = (0, 0, 255)  # vivid red: lines passed via --highlight

# Cyclic palette for coloring lines by their current stave-group -- adjacent
# groups get visibly different colors so a wrongly-merged or wrongly-split
# boundary jumps out without having to read the (also always-drawn) number.
GROUP_COLORS = [
    (255, 0, 255), (255, 160, 0), (0, 200, 200), (60, 220, 60),
    (255, 90, 180), (140, 90, 255), (0, 120, 255), (200, 200, 0),
]

UP_KEYS = {63232, 65362, 2490368}
DOWN_KEYS = {63233, 65364, 2621440}
LEFT_KEYS = {ord('h'), 63234, 65361, 2424832}
RIGHT_KEYS = {ord('l'), 63235, 65363, 2555904}


def find_image(input_dir: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    candidates = pi._image_candidates(input_dir)
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one image in {input_dir}, found {len(candidates)}: {candidates}. "
                         "Use --image to disambiguate.")
    return candidates[0]


def find_staff_json(input_dir: Path, image_path: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    candidates = pi._prefer_stem(pi._find(input_dir, "*stafflines*.json"), image_path)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Multiple stafflines candidates in {input_dir}: {candidates}. Use --staff-json.")
    # None found: derive a path to create fresh, next to the image.
    return input_dir / f"{image_path.stem}_stafflines.json"


MANUAL_MARKER = "_manually_corrected"


def manually_corrected_path(path: Path) -> Path:
    """<name>_stafflines.json -> <name>_stafflines_manually_corrected.json --
    once a file's been through this tool, its own name should say so rather
    than looking identical to an untouched automatic detection. Idempotent:
    a path that already carries the marker is returned as-is."""
    if MANUAL_MARKER in path.stem:
        return path
    return path.with_name(f"{path.stem}{MANUAL_MARKER}{path.suffix}")


def y_mid(ln: dict) -> float:
    return (ln["y_start"] + ln["y_end"]) / 2


def as_regroup_entry(ln: dict) -> dict:
    """Just enough of staff_regroup's entry schema for estimate_line_gap /
    _x_span / _y_mean to work on one of this tool's simplified line dicts."""
    return {"centerline_page": {"x_start": ln["x_start"], "x_end": ln["x_end"],
                                "y_values": [ln["y_start"], ln["y_end"]]}}


def auto_assign_groups(lines: list[dict]) -> dict:
    """{line id: group number} from the same gap-threshold heuristic
    staff_regroup uses (its own estimate_line_gap and STAVE_CUT_GAPS), so
    this starting guess matches what regroup_entries would actually do. A
    starting guess only; digit keys / 'g' in the editor override individual
    lines by hand for exactly the cases where that automatic threshold gets
    it wrong."""
    gap_unit = estimate_line_gap([as_regroup_entry(ln) for ln in lines]) or 20.0
    threshold = gap_unit * STAVE_CUT_GAPS
    ordered = sorted(lines, key=y_mid)
    groups, current, prev_y = {}, -1, None
    for ln in ordered:
        y = y_mid(ln)
        if prev_y is None or (y - prev_y) > threshold:
            current += 1
        groups[ln["id"]] = current
        prev_y = y
    return groups


def load_lines(staff_json: Path) -> list[dict]:
    """Raw file entries -> editable {id, x_start, y_start, x_end, y_end,
    scale_unit, group} dicts.

    group is taken straight from the file's own stave_id if it was saved
    with the manually_grouped flag (so reopening a page keeps prior
    grouping work exactly, including any gaps in the numbering left by
    'g'); otherwise it's freshly auto-computed.
    """
    if not staff_json.exists():
        return []
    raw = json.loads(staff_json.read_text())
    lines = []
    had_manual_groups = any(
        e.get("stave_id") is not None and "manually_grouped" in e.get("quality", {}).get("flags", [])
        for e in raw
    )
    for entry in raw:
        cp = entry.get("centerline_page")
        if not cp or not cp.get("y_values"):
            continue
        lines.append({
            "id": entry.get("id", f"line{len(lines)}"),
            "x_start": cp["x_start"], "x_end": cp["x_end"],
            "y_start": cp["y_values"][0], "y_end": cp["y_values"][-1],
            "scale_unit": entry.get("scale_unit") or 20.0,
            "group": entry.get("stave_id") if had_manual_groups else None,
        })
    if not had_manual_groups:
        groups = auto_assign_groups(lines)
        for ln in lines:
            ln["group"] = groups[ln["id"]]
    return lines


def compute_group_ids(lines: list[dict]) -> dict:
    """{id(line dict): group number} straight from each line's own group field --
    no derivation needed now that group is assigned directly (by the digit
    keys / 'g' / initial auto-assignment), not inferred from a boundary flag.

    Keyed by object identity, not the line's own "id" string: that string is
    meant to be unique but isn't guaranteed to be (e.g. a stale duplicate from
    before the manual-id-collision fix, or an upstream detector quirk) -- and
    if two distinct line dicts shared a string id, keying by it here would
    silently coalesce their groups/indices into whichever happened to be
    written last, moving both when only one was meant to move."""
    return {id(ln): ln.get("group", 0) for ln in lines}


def next_fresh_group(lines: list[dict]) -> int:
    used = {ln.get("group", 0) for ln in lines}
    return max(used, default=-1) + 1


def save_lines(staff_json: Path, lines: list[dict]):
    """Writes explicit stave_id / within_stave_index for every line.

    within_stave_index is delegated to staff_regroup._line_indices rather
    than a naive top-to-bottom count within each group: a physical line
    split into left/right fragments by a decorated initial is two entries
    here, and a naive count would give them two different indices instead
    of the one shared index the real pitch-finding pipeline expects (see
    its own fragment tests) -- silently doubling every step number below a
    split line. _line_indices is the same fragment-aware logic
    regroup_entries already uses on the automatic path, just applied to
    the stave boundaries this tool's 'g' key decided instead of the ones
    the gap threshold would have picked on its own.
    """
    group_ids = compute_group_ids(lines)
    gap = estimate_line_gap([as_regroup_entry(ln) for ln in lines]) or 20.0
    within_index = {}
    for gid in set(group_ids.values()):
        members = [ln for ln in lines if group_ids[id(ln)] == gid]
        entries = [as_regroup_entry(ln) for ln in members]
        indices = _line_indices(entries, gap)  # [(within_stave_index, residual), ...], same order as members
        for ln, (idx, _residual) in zip(members, indices):
            within_index[id(ln)] = idx

    out = []
    for i, ln in enumerate(lines):
        x0, x1 = round(ln["x_start"]), round(ln["x_end"])
        if x1 < x0:
            x0, x1 = x1, x0
        n = x1 - x0 + 1
        y0, y1 = ln["y_start"], ln["y_end"]
        y_values = [y0 + (y1 - y0) * t / max(1, n - 1) for t in range(n)]
        out.append({
            "id": ln.get("id", f"line{i}"),
            "source": "detected",
            "centerline_page": {"x_start": x0, "x_end": x1, "y_values": y_values},
            "fit": {"residual_mean": 1.0},
            "quality": {"flags": ["manually_corrected", "manually_grouped"]},
            "scale_unit": ln["scale_unit"],
            "column_id": None,
            "stave_id": group_ids[id(ln)],
            "within_stave_index": within_index[id(ln)],
        })
    target = manually_corrected_path(staff_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2))
    if target != staff_json and staff_json.exists():
        staff_json.unlink()  # don't leave the old plain-named file sitting next to it -- page_inputs.py's
                             # discovery requires exactly one *stafflines*.json candidate per page
    n_groups = len(set(group_ids.values()))
    print(f"saved {len(out)} lines in {n_groups} groups to {target}")
    print("(the saved stave_id/within_stave_index only take effect if the caller loads with "
          "regroup=False -- pass --no-regroup to run_pitch_finding.py / annotate_notecenters.py)")
    return target


def dist_point_segment(px, py, x0, y0, x1, y1) -> float:
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        return ((px - x0) ** 2 + (py - y0) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length_sq))
    proj_x, proj_y = x0 + t * dx, y0 + t * dy
    return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5


class StafflineEditor:
    def __init__(self, image, lines, default_scale_unit, highlight_ids=None):
        self.image = image
        self.original_lines = [dict(ln) for ln in lines]
        self.lines = [dict(ln) for ln in lines]
        self.default_scale_unit = default_scale_unit
        self.highlight_ids = set(highlight_ids or [])

        self.img_h, self.img_w = image.shape[:2]
        self.fit_scale = min(MAX_DISPLAY_W / self.img_w, MAX_DISPLAY_H / self.img_h, 1.0)
        self.zoom = 1.0
        self.pan_x, self.pan_y = 0.0, 0.0  # top-left of viewport, in image coords

        self.drag = None  # dict describing the in-progress mouse action
        self.add_mode = False
        self.mouse_screen = (0, 0)
        self.dirty = False

        # Seed past any existing "manualN" ids so newly-added lines never collide with
        # ones from an earlier session (len(self.lines) is not collision-safe: deleting
        # a line and adding a new one can repeat a previously-used id).
        existing_manual_nums = [
            int(ln["id"][len("manual"):]) for ln in self.lines
            if ln["id"].startswith("manual") and ln["id"][len("manual"):].isdigit()
        ]
        self.next_manual_id = max(existing_manual_nums, default=-1) + 1

    @property
    def scale(self):
        return self.fit_scale * self.zoom

    def viewport_size(self):
        s = self.scale
        return MAX_DISPLAY_W / s, MAX_DISPLAY_H / s

    def clamp_pan(self):
        vp_w, vp_h = self.viewport_size()
        self.pan_x = max(0.0, min(self.pan_x, max(0.0, self.img_w - vp_w)))
        self.pan_y = max(0.0, min(self.pan_y, max(0.0, self.img_h - vp_h)))

    def to_screen(self, x, y):
        s = self.scale
        return (x - self.pan_x) * s, (y - self.pan_y) * s

    def to_image(self, sx, sy):
        s = self.scale
        return self.pan_x + sx / s, self.pan_y + sy / s

    def zoom_at(self, sx, sy, factor):
        img_x, img_y = self.to_image(sx, sy)
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom * factor))
        s = self.scale
        self.pan_x = img_x - sx / s
        self.pan_y = img_y - sy / s
        self.clamp_pan()

    def nearest_endpoint(self, sx, sy):
        best = None
        best_d = HANDLE_RADIUS_SCREEN_PX
        for i, ln in enumerate(self.lines):
            for which in ("start", "end"):
                x, y = ln[f"x_{which}"], ln[f"y_{which}"]
                ex, ey = self.to_screen(x, y)
                d = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
                if d < best_d:
                    best_d, best = d, (i, which)
        return best

    def nearest_line(self, sx, sy):
        best = None
        best_d = BODY_HIT_SCREEN_PX
        for i, ln in enumerate(self.lines):
            x0, y0 = self.to_screen(ln["x_start"], ln["y_start"])
            x1, y1 = self.to_screen(ln["x_end"], ln["y_end"])
            d = dist_point_segment(sx, sy, x0, y0, x1, y1)
            if d < best_d:
                best_d, best = d, i
        return best

    def on_mouse(self, event, sx, sy, flags, param):
        self.mouse_screen = (sx, sy)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.add_mode:
                ix, iy = self.to_image(sx, sy)
                self.drag = {"type": "new", "x0": ix, "y0": iy, "x1": ix, "y1": iy}
                return
            hit = self.nearest_endpoint(sx, sy)
            if hit is not None:
                i, which = hit
                self.drag = {"type": "endpoint", "line": i, "which": which}
                return
            hit = self.nearest_line(sx, sy)
            if hit is not None:
                ln = self.lines[hit]
                ix, iy = self.to_image(sx, sy)
                self.drag = {"type": "body", "line": hit, "anchor_y": iy,
                            "orig_y_start": ln["y_start"], "orig_y_end": ln["y_end"]}
                return
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag is None:
                return
            ix, iy = self.to_image(sx, sy)
            if self.drag["type"] == "endpoint":
                ln = self.lines[self.drag["line"]]
                ln[f"x_{self.drag['which']}"] = ix
                ln[f"y_{self.drag['which']}"] = iy
                self.dirty = True
            elif self.drag["type"] == "body":
                dy = iy - self.drag["anchor_y"]
                ln = self.lines[self.drag["line"]]
                ln["y_start"] = self.drag["orig_y_start"] + dy
                ln["y_end"] = self.drag["orig_y_end"] + dy
                self.dirty = True
            elif self.drag["type"] == "new":
                self.drag["x1"], self.drag["y1"] = ix, iy
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag and self.drag["type"] == "new":
                x0, y0, x1, y1 = self.drag["x0"], self.drag["y0"], self.drag["x1"], self.drag["y1"]
                if x1 < x0:
                    x0, x1, y0, y1 = x1, x0, y1, y0
                if x1 - x0 > 2:
                    new_y_mid = (y0 + y1) / 2
                    nearest_group = (min(self.lines, key=lambda ln: abs(y_mid(ln) - new_y_mid)).get("group", 0)
                                    if self.lines else 0)
                    self.lines.append({
                        "id": f"manual{self.next_manual_id}", "x_start": x0, "x_end": x1,
                        "y_start": y0, "y_end": y1, "scale_unit": self.default_scale_unit,
                        "group": nearest_group,  # assume it joins its nearest neighbour; digit key to override
                    })
                    self.next_manual_id += 1
                    self.dirty = True
                self.add_mode = False
            self.drag = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            hit = self.nearest_line(sx, sy)
            if hit is not None:
                self.lines.pop(hit)
                self.dirty = True

    def render(self):
        vp_w, vp_h = self.viewport_size()
        x0, y0 = int(self.pan_x), int(self.pan_y)
        x1, y1 = min(self.img_w, x0 + int(np.ceil(vp_w)) + 1), min(self.img_h, y0 + int(np.ceil(vp_h)) + 1)
        crop = self.image[y0:y1, x0:x1]
        s = self.scale
        disp_w, disp_h = max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))
        canvas = cv2.resize(crop, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        # pad up to the fixed window size so it doesn't jitter near page edges
        pad_h, pad_w = MAX_DISPLAY_H - canvas.shape[0], MAX_DISPLAY_W - canvas.shape[1]
        if pad_h > 0 or pad_w > 0:
            canvas = cv2.copyMakeBorder(canvas, 0, max(0, pad_h), 0, max(0, pad_w),
                                        cv2.BORDER_CONSTANT, value=(40, 40, 40))

        hover_line = self.nearest_line(*self.mouse_screen) if self.drag is None and not self.add_mode else None
        hover_handle = self.nearest_endpoint(*self.mouse_screen) if self.drag is None and not self.add_mode else None
        group_ids = compute_group_ids(self.lines)
        n_groups = len(set(group_ids.values())) if self.lines else 0

        for i, ln in enumerate(self.lines):
            p0 = tuple(round(v) for v in self.to_screen(ln["x_start"], ln["y_start"]))
            p1 = tuple(round(v) for v in self.to_screen(ln["x_end"], ln["y_end"]))
            is_highlighted = ln.get("id") in self.highlight_ids
            group_color = GROUP_COLORS[group_ids[id(ln)] % len(GROUP_COLORS)]
            if i == hover_line:
                color, thickness = HOVER_COLOR, 3
            elif is_highlighted:
                color, thickness = HIGHLIGHT_COLOR, 3
            else:
                color, thickness = group_color, 2
            cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)
            for which, p in (("start", p0), ("end", p1)):
                hc = HOVER_COLOR if hover_handle == (i, which) else HANDLE_COLOR
                cv2.circle(canvas, p, 4, hc, -1, cv2.LINE_AA)
            # group number always visible at the line's left end, not just on hover
            label = str(group_ids[id(ln)])
            lx, ly = p0[0] - 6, p0[1] - 8
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), (255, 255, 255), -1)
            cv2.putText(canvas, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        hover_idx = hover_line if hover_line is not None else (hover_handle[0] if hover_handle else None)
        if hover_idx is not None:
            ln = self.lines[hover_idx]
            hover_text = f"{ln.get('id', '?')}  group {group_ids[id(ln)]}"
            tx, ty = self.mouse_screen[0] + 12, max(20, self.mouse_screen[1] - 10)
            (tw, th), _ = cv2.getTextSize(hover_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), (255, 255, 255), -1)
            cv2.putText(canvas, hover_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if self.drag and self.drag["type"] == "new":
            p0 = tuple(round(v) for v in self.to_screen(self.drag["x0"], self.drag["y0"]))
            p1 = tuple(round(v) for v in self.to_screen(self.drag["x1"], self.drag["y1"]))
            cv2.line(canvas, p0, p1, NEW_LINE_PREVIEW_COLOR, 2, cv2.LINE_AA)

        footer_lines = [
            f"lines: {len(self.lines)}   groups: {n_groups}   zoom: {self.zoom:.2f}x"
            + (f"   {len(self.highlight_ids)} highlighted (red)" if self.highlight_ids else "")
            + ("   [ADD MODE: click-drag to place a line]" if self.add_mode else ""),
            "hover a line/endpoint       : shows its id and group near the cursor",
            "left-drag near an endpoint : move that endpoint",
            "left-drag near a line body : move the whole line vertically",
            "right-click near a line    : delete that line",
            "a, then left-drag           : add a new line",
            "0-9, while hovering a line  : assign that line to group 0-9",
            "g, while hovering a line    : give that line a fresh group number",
            "+ / -                       : zoom in / out",
            "arrow keys                  : pan",
            "s : save     r : reset to original     q : save and quit",
        ]
        footer = np.full((18 + 18 * len(footer_lines), canvas.shape[1], 3), 255, dtype=np.uint8)
        for i, line in enumerate(footer_lines):
            cv2.putText(footer, line, (6, 16 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        return np.vstack([canvas, footer])

    def run(self, staff_json: Path):
        win = "fix_stafflines"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self.on_mouse)

        while True:
            cv2.imshow(win, self.render())
            key = cv2.waitKeyEx(20)

            if key == ord('q'):
                staff_json = save_lines(staff_json, self.lines)
                break
            elif key == ord('s'):
                staff_json = save_lines(staff_json, self.lines)
            elif key == ord('r'):
                self.lines = [dict(ln) for ln in self.original_lines]
                self.dirty = False
                print("reset to original lines")
            elif key == ord('a'):
                self.add_mode = not self.add_mode
            elif key == ord('g'):
                hit = self.nearest_line(*self.mouse_screen)
                if hit is not None:
                    self.lines[hit]["group"] = next_fresh_group(self.lines)
                    self.dirty = True
            elif ord('0') <= key <= ord('9'):
                hit = self.nearest_line(*self.mouse_screen)
                if hit is not None:
                    self.lines[hit]["group"] = key - ord('0')
                    self.dirty = True
            elif key in (ord('+'), ord('=')):
                self.zoom_at(self.mouse_screen[0], self.mouse_screen[1], ZOOM_STEP)
            elif key == ord('-'):
                self.zoom_at(self.mouse_screen[0], self.mouse_screen[1], 1 / ZOOM_STEP)
            elif key in UP_KEYS:
                self.pan_y -= self.viewport_size()[1] * PAN_FRACTION
                self.clamp_pan()
            elif key in DOWN_KEYS:
                self.pan_y += self.viewport_size()[1] * PAN_FRACTION
                self.clamp_pan()
            elif key in LEFT_KEYS:
                self.pan_x -= self.viewport_size()[0] * PAN_FRACTION
                self.clamp_pan()
            elif key in RIGHT_KEYS:
                self.pan_x += self.viewport_size()[0] * PAN_FRACTION
                self.clamp_pan()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("page", type=Path)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--staff-json", type=Path, default=None)
    parser.add_argument("--highlight", type=str, default=None,
                         help="Comma-separated line ids (as printed by check_stafflines.py) "
                              "to draw in red so they're easy to spot, e.g. "
                              "--highlight line0024,line0097")
    args = parser.parse_args()
    highlight_ids = args.highlight.split(",") if args.highlight else []

    page_dir = args.page
    input_dir = page_dir / pi.INPUT_DIR_NAME
    if not input_dir.is_dir():
        input_dir = page_dir

    image_path = find_image(input_dir, args.image)
    staff_json = find_staff_json(input_dir, image_path, args.staff_json)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    lines = load_lines(staff_json)
    default_scale_unit = (sum(ln["scale_unit"] for ln in lines) / len(lines)) if lines else 20.0
    print(f"image: {image_path}")
    print(f"staff json: {staff_json} ({'exists' if staff_json.exists() else 'will be created'}, {len(lines)} lines loaded)")

    StafflineEditor(image, lines, default_scale_unit, highlight_ids=highlight_ids).run(staff_json)


if __name__ == "__main__":
    main()
