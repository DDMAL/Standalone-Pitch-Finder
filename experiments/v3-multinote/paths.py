"""Where v3 finds things. v3 extends v2 on the same data: v2's notes.csv, crop
rules, backbone, augmentation and training loop are imported from ../v2-square
(and its dataset/), which importing this module puts on the import path. v3's
own modules are named neume_* / *_notes so they don't shadow v2's model.py,
data.py or train.py."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "v2-square"
ROOT = HERE.parents[1]
DATA = ROOT / "data"
CKPT = HERE / "checkpoints"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
NOTES_CSV = V2 / "dataset" / "notes.csv"

for p in (V2 / "dataset", V2):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
