"""All-events facial AU velocity ranker.  Saves _auvel50.npy."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp_utils import TRAIN, TEST, BASE, TEST_IDS, KEY_COLS, train_one
from sklearn.preprocessing import StandardScaler


def load_auvel(sid: int, split: str):
    base = (TRAIN/"Video"/"train"/"openface"/str(sid)) if split=="train" \
           else (TEST/"Video"/"openface"/str(sid))
    parts = []
    for e in [1, 2, 3]:
        p = base/f"event_{e}"/f"event_{e}_all.npy"
        if not p.exists(): return None
        arr = np.load(str(p)).astype(np.float32)[:,1:][:,KEY_COLS]
        parts.append(np.concatenate([arr.mean(0), arr.std(0),
                                     np.abs(np.diff(arr, axis=0)).mean(0)]))
    return np.concatenate(parts)   # 387-d


def main():
    labels = pd.read_csv(TRAIN/"split_labels_train.csv")
    Xtr, phq = [], []
    for sid in labels["ID"].astype(int):
        f = load_auvel(sid, "train")
        if f is None: continue
        Xtr.append(f)
        phq.append(float(labels[labels["ID"].astype(int)==sid].iloc[0]["phq9_score"]))
    Xtr = np.array(Xtr, float); phq = np.array(phq, float)
    Xte = np.array([load_auvel(s, "test") for s in TEST_IDS], float)
    sc  = StandardScaler().fit(Xtr)
    print(f"auvel: train={len(Xtr)} feat={Xtr.shape[1]}  training 50 seeds...")
    auvel = np.mean([train_one(sc.transform(Xtr), phq, sc.transform(Xte), s)
                     for s in range(50)], 0)
    np.save(BASE/"_auvel50.npy", auvel)
    print(f"  saved _auvel50.npy  std={auvel.std():.2f}")


if __name__ == "__main__": main()
