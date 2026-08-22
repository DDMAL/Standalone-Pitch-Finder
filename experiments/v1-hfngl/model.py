"""StepCNN: v1 single-note stave-step regressor.

Given one glyph's crop, predicts a single continuous stave step (same
convention as staff_io.py: bottom detected line = 0, +2 per line up, odd
values are spaces). Only ever trained/evaluated on single-note glyphs --
see ../README.md for scope.
"""
import torch.nn as nn

IMG_H, IMG_W = 128, 32


class StepCNN(nn.Module):
    """AdaptiveAvgPool2d((8,1)) keeps 8 vertical bins into the head instead
    of collapsing the whole spatial map to one value per channel -- the
    original AdaptiveAvgPool2d(1) did that and caused mode collapse: for a
    task that's fundamentally "where vertically is the ink", it threw away
    the one signal the head needs. This fix alone took exact-match from
    ~25-29% to ~80-89% in early runs. ~56k params.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 128x32 -> 64x16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64x16 -> 32x8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 1)),                                 # 32x8 -> 8x1
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(64 * 8, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)
