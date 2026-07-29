"""
Shared debug-overlay drawing helpers, plus the artifact-path handling the
three CLIs (run_pitch_finding / run_rodan_pitch_finding / render_ic_debug)
have in common.

Glyph bboxes on these manuscript scans are ~20-30px, too small to caption
legibly at native resolution. load_scaled_image upscales the canvas first
so labels get enough real pixels to be readable without shrinking relative
to the glyphs; draw_labeled_box then draws a box plus a white-backed label
so text doesn't get lost against parchment texture / dark ink.

draw_stafflines draws the fitted staff centerlines that pitch-finding
actually queried, so a wrong pitch can be read as either "the staff fit is
off here" or "the fit is fine, the step lookup is wrong". Call it before the
draw_labeled_box loop: lines go down first and glyph boxes/labels on top, so
labels stay readable where they cross a line.

draw_note_center completes that chain: it marks the exact point a pitch was
read from, which is the one quantity neither the box nor the staff lines
show. A bbox says where the glyph is, not which pixel row inside it the
algorithm treated as the notehead -- and for multi-note neumes there is more
than one such row per box.
"""

from pathlib import Path

import cv2
import numpy as np

# BGR magenta: deliberately outside the palettes the callers use for glyph
# boxes (green/blue/purple/orange/grey/red), so staff lines never read as
# another glyph category.
STAFFLINE_COLOR = (255, 0, 255)

# Halo drawn under note-center crosshairs. White, matching the label
# backgrounds, so a marker sitting on a notehead's own black ink stays visible
# without needing a seventh category color.
MARKER_HALO_COLOR = (255, 255, 255)

# Extensions cv2.imwrite can pick an encoder for. Anything else fails deep
# inside imwrite_ with "could not find a writer for the specified extension",
# which says nothing about which argument was wrong -- so check it ourselves.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})

# --debug-viz reads like an on/off flag, so it gets passed like one. Treat a
# bare flag or any of these as "render it, you pick the filename".
_AUTO_TOKENS = frozenset({"1", "true", "yes", "on", "auto"})


def check_image_suffix(path: Path):
    """Raise ValueError unless cv2.imwrite can infer a format from path."""
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        want = ", ".join(sorted(IMAGE_SUFFIXES))
        raise ValueError(
            f"'{path}' has no image extension cv2 can write (want one of: {want}). "
            "Pass a filename like page_debug.jpg, or pass the flag bare to have "
            "one named after --output.")


def resolve_output_path(output_path: Path, image_path: Path, suffix: str) -> Path:
    """Let --output name either a file or a directory to drop the file into.

    A path that already exists as a directory, or that has no extension yet
    (i.e. 'some/dir/' -- argparse's Path() conversion eats the trailing
    separator, so that's the only trace left of the intent), gets
    '<image stem><suffix>' appended instead of being written to as a file.

    Pure: directories are created at write time, so a later usage error
    doesn't leave an empty one behind.
    """
    if output_path.is_dir() or (not output_path.exists() and not output_path.suffix):
        output_path = output_path / f"{image_path.stem}{suffix}"
    return output_path


def resolve_debug_viz_path(value, auto_path: Path):
    """Map a --debug-viz argument to the image path to write, or None if unset.

    Validating here means a bad extension is reported before the debug render,
    which upscales the whole page and allocates hundreds of MB -- not after.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _AUTO_TOKENS:
        return auto_path
    path = Path(text)
    check_image_suffix(path)
    return path


def write_image(path: Path, img):
    """cv2.imwrite that fails loudly.

    imwrite returns False rather than raising when the parent directory is
    missing, so create it first and check the result -- otherwise a debug
    render silently produces no file.
    """
    check_image_suffix(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"cv2 could not write {path} ({img.shape[1]}x{img.shape[0]}px); "
                      "if this is a large --debug-scale, try a smaller one or a .png.")


def load_scaled_image(image_path: Path, scale: float):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _draw_label(img, label: str, org: tuple, color: tuple, scale: float,
                center_v: bool = False):
    """White-backed text at org (bottom-left of the text), clamped into frame.

    With center_v, org's y is treated as the vertical *center* of the text
    instead of its baseline -- for labelling a point marker, where the text
    should read as level with the point it belongs to.
    """
    font_scale = 0.45 * scale
    font_thickness = max(1, round(scale * 0.7))
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    if center_v:
        org = (org[0], org[1] + round((th - baseline) / 2))
    org = (max(0, org[0]), max(th + baseline, org[1]))
    bg_p1 = (org[0] - 1, org[1] - th - baseline)
    bg_p2 = (org[0] + tw + 1, org[1] + baseline)
    cv2.rectangle(img, bg_p1, bg_p2, (255, 255, 255), -1)
    cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                font_thickness, cv2.LINE_AA)


def draw_labeled_box(img, ulx: float, uly: float, ncols: float, nrows: float,
                      label: str, color: tuple, scale: float):
    p1 = (round(ulx * scale), round(uly * scale))
    p2 = (round((ulx + ncols) * scale), round((uly + nrows) * scale))

    box_thickness = max(1, round(scale))
    cv2.rectangle(img, p1, p2, color, box_thickness)

    if not label:
        return

    _draw_label(img, label, (p1[0], p1[1] - round(4 * scale)), color, scale)


def draw_note_center(img, x: float, y: float, scale: float, color: tuple,
                     label: str = None, label_x: float = None, arm: float = 5.0):
    """Mark one computed notehead center at page-pixel (x, y).

    A crosshair rather than a dot: the horizontal arm makes the measured row
    readable against the staff lines it has to be compared with (that y is the
    whole pitch decision), and the vertical arm shows which x the stave was
    queried at -- the two algorithms use different reference x values (bbox
    center vs bbox left edge), which is otherwise invisible.

    Drawn color-over-white so the marker survives landing on black ink, which
    is exactly where noteheads are. arm is in page pixels, so markers keep
    their size relative to the glyphs at any --debug-scale.

    label goes to the right of the marker, vertically centered on it.
    label_x (page pixels) overrides where it starts -- pass the glyph's right
    edge to keep multi-note labels off the glyph's own ink.
    """
    px, py = round(x * scale), round(y * scale)
    arm_px = max(2, round(arm * scale))
    thickness = max(1, round(scale * 0.6))
    halo_thickness = thickness + max(2, round(scale * 0.8))

    for line_color, line_thickness in ((MARKER_HALO_COLOR, halo_thickness), (color, thickness)):
        cv2.line(img, (px - arm_px, py), (px + arm_px, py), line_color, line_thickness, cv2.LINE_AA)
        cv2.line(img, (px, py - arm_px), (px, py + arm_px), line_color, line_thickness, cv2.LINE_AA)

    if not label:
        return

    text_x = round(label_x * scale) if label_x is not None else px + arm_px
    _draw_label(img, label, (text_x + max(1, round(2 * scale)), py), color, scale, center_v=True)


def draw_stafflines(img, staves, scale: float, color: tuple = STAFFLINE_COLOR,
                    label_lines: bool = False):
    """Draw every stave's fitted centerlines on an already-scaled canvas.

    staves is a list of staff_io.Stave. Each StaffLine's y_values are one
    sample per page pixel starting at x_start, so the polyline traces the fit
    exactly (including its curvature) rather than a straight approximation.
    Lines are drawn thinner than glyph boxes so they don't dominate the
    overlay. With label_lines, each line is tagged "s<stave_id>/<step>" at its
    left end, where step is the same bottom-line-is-0 value staff_io hands to
    pitch-finding -- useful for checking an off-by-a-third pitch against the
    line it was measured from.
    """
    thickness = max(1, round(scale * 0.5))
    for stave in staves:
        max_idx = max(ln.within_stave_index for ln in stave.lines)
        for line in stave.lines:
            if not line.y_values:
                continue
            xs = line.x_start + np.arange(len(line.y_values))
            pts = np.stack([xs * scale, np.asarray(line.y_values) * scale], axis=1)
            pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)

            if label_lines:
                step = 2 * (max_idx - line.within_stave_index)
                x0, y0 = int(pts[0][0][0]), int(pts[0][0][1])
                _draw_label(img, f"s{stave.stave_id}/{step}",
                            (x0 + round(2 * scale), y0 - round(2 * scale)), color, scale)
