"""Shared utilities for facial feature rankers.

Contains:
  - Path definitions (ROOT, TRAIN, TEST, BASE)
  - TEST_IDS  – read from the official sample file, never hardcoded
  - KEY_COLS  – the 43 OpenFace columns used by all facial rankers
  - MLP class, CCC loss, train_one()  – the 50-seed CCC-loss MLP recipe

No test-set labels or ground-truth information is stored here.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "Train-MPDD-Young" / "Young"
TEST  = ROOT / "Test-MPDD-Young"  / "Young"
BASE  = ROOT / "official_baseline" / "make_submission_forcodabench"

# ── test IDs from the official sample file ────────────────────────────────────
TEST_IDS: list[int] = pd.read_csv(BASE / "binary_sample.csv")["id"].tolist()

# ── OpenFace feature columns (43 total) ───────────────────────────────────────
# Column indices are into arr[:,1:]  (the first column is the frame number).
# gaze: 0-7  |  AU intensity: 674-690  |  AU presence: 691-708
GAZE_COLS    = list(range(0,   8))
AU_INT_COLS  = list(range(674, 691))
AU_PRES_COLS = list(range(691, 709))
KEY_COLS     = GAZE_COLS + AU_INT_COLS + AU_PRES_COLS   # 43-d

# ── MLP architecture ──────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, d: int, h: int = 96, dr: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h),    nn.GELU(), nn.Dropout(dr),
            nn.Linear(h, h//2), nn.GELU(), nn.Dropout(dr),
            nn.Linear(h//2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ── CCC helpers ───────────────────────────────────────────────────────────────
def ccc_fn(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    cov   = np.mean((a - a.mean()) * (b - b.mean()))
    denom = a.var() + b.var() + (a.mean() - b.mean()) ** 2
    return 0.0 if denom < 1e-12 else 2 * cov / denom

def ccc_loss(a, b):
    cov = ((a - a.mean()) * (b - b.mean())).mean()
    return 1 - 2 * cov / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-8)

# ── Training recipe ───────────────────────────────────────────────────────────
def train_one(Xtr: np.ndarray, ytr: np.ndarray,
              Xte: np.ndarray, seed: int) -> np.ndarray:
    """Train one MLP seed with CCC-loss + mixup; return test predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    m   = MLP(Xtr.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt  = torch.tensor(Xtr, dtype=torch.float32)
    yt  = torch.tensor(ytr, dtype=torch.float32)
    Xv  = torch.tensor(Xte, dtype=torch.float32)
    n   = len(yt)
    best_ccc, best_state = -1.0, None
    for ep in range(400):
        m.train()
        ia  = torch.randint(0, n, (2000,))
        ib  = torch.randint(0, n, (2000,))
        lam = torch.tensor(np.random.beta(0.4, 0.4, 2000), dtype=torch.float32)
        Xm  = lam[:, None] * Xt[ia] + (1 - lam[:, None]) * Xt[ib]
        ym  = lam * yt[ia] + (1 - lam) * yt[ib]
        opt.zero_grad()
        p   = m(Xm)
        loss = ccc_loss(ym, p) + 0.1 * ((p - ym) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if ep % 20 == 0:
            m.eval()
            with torch.no_grad():
                c = ccc_fn(ytr, m(Xt).numpy())
            if c > best_ccc:
                best_ccc   = c
                best_state = {k: v.clone() for k, v in m.state_dict().items()}
    if best_state:
        m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        return np.clip(m(Xv).numpy(), 0, 27)
