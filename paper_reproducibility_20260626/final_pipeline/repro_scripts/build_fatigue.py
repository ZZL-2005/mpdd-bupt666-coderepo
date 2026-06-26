"""Cross-event facial fatigue ranker.  Saves _fatigue50.npy.

Feature: cross-event delta of per-event facial means
    E2_mean - E1_mean, E3_mean - E2_mean, E3_mean - E1_mean
= 43 cols × 3 deltas = 129-d.
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


def load_fatigue(sid: int, split: str):
    base = (TRAIN/"Video"/"train"/"openface"/str(sid)) if split=="train" \
           else (TEST/"Video"/"openface"/str(sid))
    evs = []
    for e in [1, 2, 3]:
        p = base/f"event_{e}"/f"event_{e}_all.npy"
        if not p.exists(): return None
        arr = np.load(str(p)).astype(np.float32)[:,1:][:,KEY_COLS]
        evs.append(arr.mean(0))
    return np.concatenate([evs[1]-evs[0], evs[2]-evs[1], evs[2]-evs[0]])   # 129-d


def main():
    labels = pd.read_csv(TRAIN/"split_labels_train.csv")
    Xtr, phq = [], []
    for sid in labels["ID"].astype(int):
        f = load_fatigue(sid, "train")
        if f is None: continue
        Xtr.append(f)
        phq.append(float(labels[labels["ID"].astype(int)==sid].iloc[0]["phq9_score"]))
    Xtr = np.array(Xtr, float); phq = np.array(phq, float)
    Xte = np.array([load_fatigue(s, "test") for s in TEST_IDS], float)
    sc  = StandardScaler().fit(Xtr)
    print(f"fatigue: train={len(Xtr)} feat={Xtr.shape[1]}  training 50 seeds...")
    fat = np.mean([train_one(sc.transform(Xtr), phq, sc.transform(Xte), s)
                   for s in range(50)], 0)
    np.save(BASE/"_fatigue50.npy", fat)
    print(f"  saved _fatigue50.npy  std={fat.std():.2f}")


if __name__ == "__main__": main()
