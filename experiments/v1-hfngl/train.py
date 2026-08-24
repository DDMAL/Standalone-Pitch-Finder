"""Train the v1 StepCNN and save checkpoints for evaluate.py to reuse.

    python -u train.py 2>&1 | tee train.log   # -u so progress prints live
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import StepCNN
from data import REAL_LABELED_PAGES, load_real_labeled_page, load_shapes
from split import build_split, SEED
from augment import build_augmented_rows

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
EPOCHS = 300
LR = 1e-3
BATCH = 16
N_AUG = 3  # augmented copies generated per training glyph, see augment.py


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_cnn(Xtr, ytr, seed=SEED, log_every=20):
    set_seed(seed)
    model = StepCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    n = len(Xtr_t)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            pred = model(Xtr_t[b])
            loss = loss_fn(pred, ytr_t[b])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(b)
        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"    epoch {epoch+1:4d}/{EPOCHS}  train MSE={total_loss/n:.4f}", flush=True)
    return model


def main():
    set_seed(SEED)
    shapes = load_shapes()

    print("loading real-labeled pages...", flush=True)
    all_rows = []
    for page in REAL_LABELED_PAGES:
        rows = load_real_labeled_page(page, shapes)
        print(f"  {page}: {len(rows)}", flush=True)
        all_rows += rows

    print("building split...", flush=True)
    train_rows, test_rows = build_split(all_rows)
    print(f"train={len(train_rows)} test={len(test_rows)}", flush=True)

    X_train = np.stack([r["crop"] for r in train_rows])[:, None, :, :]
    y_train = np.array([r["truth"] for r in train_rows], dtype=np.float32)

    CKPT_DIR.mkdir(exist_ok=True)

    print("training (real-only)...", flush=True)
    model_real = train_cnn(X_train, y_train)
    torch.save(model_real.state_dict(), CKPT_DIR / "stepcnn_real.pt")

    print(f"building {N_AUG}x augmented copies of the real training set...", flush=True)
    aug_rows = build_augmented_rows(train_rows, N_AUG, seed=SEED)
    print(f"  {len(aug_rows)} augmented crops", flush=True)
    X_aug = np.stack([r["crop"] for r in aug_rows])[:, None, :, :]
    y_aug = np.array([r["truth"] for r in aug_rows], dtype=np.float32)
    X_train_aug = np.concatenate([X_train, X_aug], axis=0)
    y_train_aug = np.concatenate([y_train, y_aug], axis=0)

    print(f"training (real + {N_AUG}x augmented, staff-error-robustness variant)...", flush=True)
    model_aug = train_cnn(X_train_aug, y_train_aug)
    torch.save(model_aug.state_dict(), CKPT_DIR / "stepcnn_real_aug.pt")

    meta = {
        "seed": SEED, "n_train_real": len(train_rows), "n_test": len(test_rows),
        "n_aug": len(aug_rows), "pages": REAL_LABELED_PAGES,
    }
    (CKPT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved checkpoints to {CKPT_DIR}", flush=True)


if __name__ == "__main__":
    main()
