"""Sanity checks for Stave step/interpolation math on synthetic lines."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from staff_io import StaffLine, Stave


def make_flat_line(y: float, within_idx: int, x_start=0, x_end=99, scale_unit=10.0) -> StaffLine:
    return StaffLine(
        line_id=f"line{within_idx}", stave_id=0, within_stave_index=within_idx,
        x_start=x_start, x_end=x_end, y_values=[y] * (x_end - x_start + 1),
        scale_unit=scale_unit,
    )


def test_step_at_x_four_line_stave():
    # 4 lines, 20px apart, within_stave_index 0 (top) .. 3 (bottom).
    lines = [make_flat_line(y=100 + 20 * i, within_idx=i) for i in range(4)]
    stave = Stave(stave_id=0, lines=lines)
    pairs = dict(stave.step_at_x(50))  # {step: y}
    # bottom line (within_idx=3, y=160) is step 0; each line up is +2 steps.
    assert pairs[0] == 160.0
    assert pairs[2] == 140.0
    assert pairs[4] == 120.0
    assert pairs[6] == 100.0


def test_continuous_step_on_a_line():
    lines = [make_flat_line(y=100 + 20 * i, within_idx=i) for i in range(4)]
    stave = Stave(stave_id=0, lines=lines)
    step, flags = stave.continuous_step_at_y(50, 140.0)
    print(f"step at y=140 (a real line): {step}, flags={flags}")
    assert step == 2
    assert flags == []


def test_continuous_step_in_a_space():
    lines = [make_flat_line(y=100 + 20 * i, within_idx=i) for i in range(4)]
    stave = Stave(stave_id=0, lines=lines)
    # Halfway between the y=140 (step 2) and y=120 (step 4) lines -> step 3.
    step, flags = stave.continuous_step_at_y(50, 130.0)
    print(f"step at y=130 (a space): {step}")
    assert step == 3


def test_continuous_step_extrapolates_above_and_below():
    lines = [make_flat_line(y=100 + 20 * i, within_idx=i) for i in range(4)]
    stave = Stave(stave_id=0, lines=lines)
    above, _ = stave.continuous_step_at_y(50, 80.0)   # 20px above top line (step 6) = +2 steps
    below, _ = stave.continuous_step_at_y(50, 180.0)  # 20px below bottom line (step 0) = -2 steps
    assert above == 8
    assert below == -2


def test_y_at_step_inverts_continuous_step_at_y():
    lines = [make_flat_line(y=100 + 20 * i, within_idx=i) for i in range(4)]
    stave = Stave(stave_id=0, lines=lines)
    # On a line, in a space, and extrapolated past both real edges: the
    # debug overlay draws what pitch-finding computed, so this has to be the
    # exact inverse rather than an approximation of it.
    for y in (140.0, 130.0, 133.7, 80.0, 180.0):
        step, _ = stave.continuous_step_at_y(50, y)
        assert stave.y_at_step(50, step) == y


def test_y_at_step_lands_on_the_line_it_names():
    lines = [make_flat_line(y=100 + 20 * i, within_idx=i) for i in range(4)]
    stave = Stave(stave_id=0, lines=lines)
    assert stave.y_at_step(50, 0) == 160.0    # bottom line
    assert stave.y_at_step(50, 6) == 100.0    # top line
    assert stave.y_at_step(50, 3) == 130.0    # space between step2 and step4


def test_y_at_step_uses_scale_unit_on_a_single_line_stave():
    # Mirrors continuous_step_at_y's sparse fallback: one line, so a step is
    # scale_unit/2 = 5px, with higher steps going UP the page.
    lines = [make_flat_line(y=100, within_idx=0, scale_unit=10.0)]
    stave = Stave(stave_id=0, lines=lines)
    assert stave.y_at_step(50, 0) == 100.0
    assert stave.y_at_step(50, 2) == 90.0
    assert stave.y_at_step(50, -2) == 110.0


def test_y_at_step_outside_line_span_returns_none():
    lines = [make_flat_line(y=100, within_idx=0, x_start=0, x_end=50)]
    stave = Stave(stave_id=0, lines=lines)
    assert stave.y_at_step(200, 0) is None


def test_step_at_x_outside_line_span_returns_none():
    lines = [make_flat_line(y=100, within_idx=0, x_start=0, x_end=50)]
    stave = Stave(stave_id=0, lines=lines)
    assert stave.step_at_x(200) is None


def test_sparse_stave_flag_with_one_line():
    lines = [make_flat_line(y=100, within_idx=0, scale_unit=10.0)]
    stave = Stave(stave_id=0, lines=lines)
    step, flags = stave.continuous_step_at_y(50, 100.0)
    assert "sparse_stave_lines" in flags
