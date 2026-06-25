"""AU temporal spectral ranker.  Saves _auspec50.npy.

Feature: per-event FFT band-power ratios + spectral entropy + dominant freq
= 5 stats × 43 KEY_COLS × 3 events = 645-d.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp_utils import TRAIN, TEST, BASE, TEST_IDS, KEY_COLS, train_one
from sklearn.preprocessing import StandardScaler


def spectral_feats(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.mean(0)
    fft = np.abs(np.fft.rfft(arr, axis=0))
    F   = fft.shape[0]; b = F // 3
    lo  = fft[:b].mean(0); mid = fft[b:2*b].mean(0); hi = fft[2*b:].mean(0)
    tot = lo + mid + hi + 1e-8
    p   = fft / (fft.sum(0, keepdims=True) + 1e-8)
    ent = -(p * np.log(p + 1e-12)).sum(0)
    dom = fft.argmax(0) / (F + 1e-8)
    return np.concatenate([lo/tot, mid/tot, hi/tot, ent, dom])   # 215-d


def load_auspec(sid: int, split: str):
    base = (TRAIN/"Video"/"train"/"openface"/str(sid)) if split=="train" \
           else (TEST/"Video"/"openface"/str(sid))
    parts = []
    for e in [1, 2, 3]:
        p = base/f"event_{e}"/f"event_{e}_all.npy"
        if not p.exists(): return None
        parts.append(spectral_feats(
            np.load(str(p)).astype(np.float32)[:,1:][:,KEY_COLS]))
    return np.concatenate(parts)   # 645-d


def main():
    labels = pd.read_csv(TRAIN/"split_labels_train.csv")
    Xtr, phq = [], []
    for sid in labels["ID"].astype(int):
        f = load_auspec(sid, "train")
        if f is None: continue
        Xtr.append(f)
        phq.append(float(labels[labels["ID"].astype(int)==sid].iloc[0]["phq9_score"]))
    Xtr = np.array(Xtr, float); phq = np.array(phq, float)
    Xte = np.array([load_auspec(s, "test") for s in TEST_IDS], float)
    sc  = StandardScaler().fit(Xtr)
    print(f"auspec: train={len(Xtr)} feat={Xtr.shape[1]}  training 50 seeds...")
    spec = np.mean([train_one(sc.transform(Xtr), phq, sc.transform(Xte), s)
                    for s in range(50)], 0)
    np.save(BASE/"_auspec50.npy", spec)
    print(f"  saved _auspec50.npy  std={spec.std():.2f}")


if __name__ == "__main__": main()
