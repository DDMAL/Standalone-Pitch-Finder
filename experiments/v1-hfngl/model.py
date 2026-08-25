"""StepCNN: v1 single-note stave-step regressor, and StepCNNClassifier, the
classification-loss ablation of the same architecture.

Given one glyph's crop, predicts its stave step (same convention as
staff_io.py: bottom detected line = 0, +2 per line up, odd values are
spaces). Only ever trained/evaluated on single-note glyphs -- see
../README.md for scope.
"""
import torch.nn as nn

IMG_H, IMG_W = 128, 32

# Classification ablation: observed truth steps across the full labeled set
# span -1..8 inclusive (n=1205) -- fixed here rather than derived at train
# time so a class index always means the same step across runs.
STEP_MIN, STEP_MAX = -1, 8
NUM_CLASSES = STEP_MAX - STEP_MIN + 1


def _features():
    """Shared conv backbone. AdaptiveAvgPool2d((8,1)) keeps 8 vertical bins
    into the head instead of collapsing the whole spatial map to one value
    per channel -- the original AdaptiveAvgPool2d(1) did that and caused
    mode collapse: for a task that's fundamentally "where vertically is the
    ink", it threw away the one signal the head needs. This fix alone took
    exact-match from ~25-29% to ~80-89% in early runs."""
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 128x32 -> 64x16
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64x16 -> 32x8
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((8, 1)),                                 # 32x8 -> 8x1
    )


class StepCNN(nn.Module):
    """Regression head: one continuous stave-step value. ~56k params."""

    def __init__(self):
        super().__init__()
        self.features = _features()
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(64 * 8, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


class StepCNNClassifier(nn.Module):
    """Same backbone, classification head over NUM_CLASSES step values
    (cross-entropy loss) instead of a single regressed value -- the
    ablation asked: does treating this as classification instead of
    regression change anything, given regression already encodes "close
    steps are close errors" for free through MSE, which cross-entropy over
    unordered classes does not."""

    def __init__(self):
        super().__init__()
        self.features = _features()
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(64 * 8, 64), nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        return self.head(self.features(x))
