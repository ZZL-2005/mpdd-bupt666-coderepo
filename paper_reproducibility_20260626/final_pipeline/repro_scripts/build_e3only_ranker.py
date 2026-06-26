"""Event-3-only facial ranker.  Saves _e3only50.npy."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp_utils import TRAIN, TEST, BASE, TEST_IDS, KEY_COLS, train_one
from sklearn.preprocessing import StandardScaler


def load_e3only(sid: int, split: str):
    base = (TRAIN/"Video"/"train"/"openface"/str(sid)) if split=="train" \
           else (TEST/"Video"/"openface"/str(sid))
    p = base/"event_3"/"event_3_all.npy"
    if not p.exists(): return None
    arr = np.load(str(p)).astype(np.float32)[:,1:][:,KEY_COLS]
    return np.concatenate([arr.mean(0), arr.std(0),
                           np.abs(np.diff(arr, axis=0)).mean(0)])   # 129-d


def main():
    labels = pd.read_csv(TRAIN/"split_labels_train.csv")
    Xtr, phq = [], []
    for sid in labels["ID"].astype(int):
        f = load_e3only(sid, "train")
        if f is None: continue
        Xtr.append(f)
        phq.append(float(labels[labels["ID"].astype(int)==sid].iloc[0]["phq9_score"]))
    Xtr = np.array(Xtr, float); phq = np.array(phq, float)
    Xte = np.array([load_e3only(s, "test") for s in TEST_IDS], float)
    sc  = StandardScaler().fit(Xtr)
    print(f"e3only: train={len(Xtr)} feat={Xtr.shape[1]}  training 50 seeds...")
    e3 = np.mean([train_one(sc.transform(Xtr), phq, sc.transform(Xte), s)
                  for s in range(50)], 0)
    np.save(BASE/"_e3only50.npy", e3)
    print(f"  saved _e3only50.npy  std={e3.std():.2f}")


if __name__ == "__main__": main()
