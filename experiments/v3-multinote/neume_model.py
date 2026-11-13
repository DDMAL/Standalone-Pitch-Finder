"""NeumeCNN: one whole neume in, every note's stave step out.

v2's backbone (model._features: three 3 x 3 convolutions, 16 / 32 / 64 channels,
two max-pools) pooled to 8 x 8 bins instead of v2's 8 x 1, so the head can tell
the notes apart from left to right; then FC 4096 -> 256 -> MAX_NOTES x (13 steps
+ NONE). 1.10 M parameters (v2's single-note classifier: 57 k), almost all in the
first FC layer.

Output slot k is the k-th note in MEI order (reading order: a podatus's lower
note first); slots past the neume's last note are NONE, so the number of notes
is the number of slots before the first NONE.
"""
import torch.nn as nn

import paths  # noqa: F401  (v2 on the import path)
from model import NUM_CLASSES, _features  # v2's model.py

MAX_NOTES = 8  # every usable neume but 8 of 187,535 has at most 8 notes
NONE = NUM_CLASSES
NEUME_W = 64   # input width in px: 10 units at 6.4 px per unit (dataset/build_neume_crops.py)


class NeumeCNN(nn.Module):
    """(B, 1, 128, NEUME_W) -> (B, MAX_NOTES, NUM_CLASSES + 1) logits."""

    def __init__(self, pool_cols=8, hidden=256):
        super().__init__()
        self.features = _features(8, 1, pool_cols)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 8 * pool_cols, hidden), nn.ReLU(),
                                  nn.Linear(hidden, MAX_NOTES * (NUM_CLASSES + 1)))

    def forward(self, x):
        return self.head(self.features(x)).view(len(x), MAX_NOTES, NUM_CLASSES + 1)
