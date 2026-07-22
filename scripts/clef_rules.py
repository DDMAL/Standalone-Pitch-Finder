"""
Clef -> absolute pitch anchoring, and diatonic step <-> (pname, octave) math.

CLEF_REFERENCE is a placeholder default register, NOT a musicologically
validated convention -- flagged here (and in the pitch-finding plan doc) as
a tunable constant that needs review before its absolute octave numbers are
trusted for anything beyond prototype debugging. The relative pitch class
(letter name) and stave-step distance from the clef are the reliable part
of the output; the octave number rides on top of this assumption.
"""

NOTE_LETTERS = ["C", "D", "E", "F", "G", "A", "B"]

# clef pitch letter -> assumed octave at the clef's own stave position.
# Covers any clef.<letter>[variant] class (see neume_shapes._CLEF_NAME_RE);
# a letter with no entry here still gets a default (see clef_octave_for),
# flagged, rather than crashing on an unanticipated clef letter.
CLEF_OCTAVE_REFERENCE: dict[str, int] = {
    "C": 4,
    "F": 3,
    "G": 4,
}
_DEFAULT_CLEF_OCTAVE = 4


def clef_octave_for(clef_pname: str) -> tuple[int, list[str]]:
    """Placeholder octave for a clef letter, plus a flag if it wasn't in the table."""
    if clef_pname in CLEF_OCTAVE_REFERENCE:
        return CLEF_OCTAVE_REFERENCE[clef_pname], []
    return _DEFAULT_CLEF_OCTAVE, ["clef_octave_unconfigured"]


def step_to_pitch(step_delta_from_clef: float, clef_pname: str, clef_octave: int) -> tuple[str, int]:
    """Convert a diatonic step distance from a clef into (pname, octave).

    step_delta_from_clef is rounded to the nearest integer step first --
    pitch is only meaningful at whole line/space positions.
    """
    delta = round(step_delta_from_clef)
    idx = NOTE_LETTERS.index(clef_pname) + delta
    pname = NOTE_LETTERS[idx % 7]
    octave = clef_octave + idx // 7
    return pname, octave
