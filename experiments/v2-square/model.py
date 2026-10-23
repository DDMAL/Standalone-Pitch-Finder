"""v2 StepCNN / StepCNNClassifier: v1's architecture unchanged
(../v1-hfngl/model.py), with the class range widened for square notation.

Stave step convention as v1: bottom staff line = 0, +2 per line up, odd = space.
"""
import torch.nn as nn

IMG_H, IMG_W = 128, 32

# Usable labels in dataset/notes.csv span -3..9 (v1's Hufnagel set: -1..8).
# Fixed here so a class index always means the same step across runs.
STEP_MIN, STEP_MAX = -3, 9
NUM_CLASSES = STEP_MAX - STEP_MIN + 1


def _features(pool_rows=8, in_ch=1):
    """Identical to v1's backbone, including AdaptiveAvgPool2d((8,1)): keeping
    8 vertical bins is what stopped v1 from mode-collapsing."""
    return nn.Sequential(
        nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 128x32 -> 64x16
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64x16 -> 32x8
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((pool_rows, 1)),                         # 32x8 -> 8x1 at 128 px
    )


class StepCNN(nn.Module):
    """Regression head: one continuous stave-step value."""

    def __init__(self):
        super().__init__()
        self.features = _features()
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 8, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


class StepCNNClassifier(nn.Module):
    """Classification head over NUM_CLASSES step values (cross-entropy).
    pool_rows: vertical bins kept before the head; 16 for a 256-px input keeps
    v1's 16 px per bin. in_ch=2 adds the note's box as a second input channel."""

    def __init__(self, pool_rows=8, in_ch=1):
        super().__init__()
        self.features = _features(pool_rows, in_ch)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * pool_rows, 64), nn.ReLU(),
                                  nn.Linear(64, NUM_CLASSES))

    def forward(self, x):
        return self.head(self.features(x))
