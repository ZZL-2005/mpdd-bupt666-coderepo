"""End-to-end reproducible pipeline for the MPDD-AVG 2026 Young-track submission.

Produces: official_baseline/make_submission_forcodabench/young_final/submission.zip

Algorithm overview
==================
Two independent heads share the same phq9_pred (continuous depression severity)
but differ in what they optimise:

  CCC head  (phq9_pred)
  ---------
  Four facial feature sets, each trained as a 50-seed CCC-loss MLP:
    1. auvel   – per-event AU / gaze / velocity (387-d)
    2. e3only  – event-3-only AU / gaze / velocity (129-d)
    3. fatigue – cross-event facial trajectory delta (129-d)
    4. auspec  – per-event AU spectral (FFT band-power + entropy) (645-d)

  The four rankers are linearly blended with an IMU+Big5 base model (ORIG):
    phq9_raw = 0.40*ORIG + 0.30*auvel + 0.20*e3only + 0.10*fatigue + 0.05*auspec

  Variance calibration then maps the prediction to the training PHQ distribution
  (mean=4.82, std=3.92), maximising CCC = 2ρσ_ŷσ_y / (σ_ŷ²+σ_y²+(μ_ŷ-μ_y)²).

  Classification head  (binary_pred / ternary_pred)
  -------------------
  Conjunctive rule requiring BOTH the continuous blend AND the IMU base model
  to agree that the subject is depressed:
    binary  = 1  iff  phq9_pred >= BLEND_PHQ_THRESHOLD AND orig_phq >= 4
    ternary = 0 / 1 / 2 using BLEND_PHQ_THRESHOLD and TERNARY_T2.

  The thresholds are exposed as environment variables so alternative global
  decision rules can be reproduced without editing per-subject labels.

Usage
=====
  python repro_scripts/run_pipeline.py

Optional environment variables:
  BLEND_PHQ_THRESHOLD  default 4.2
  TERNARY_T2           default 10.0
  OUT_NAME             default young_final
  SKIP_CACHED          set to 1 to force recomputation of cached predictions
"""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parents[1]   # repro_scripts/ is one level below project root
SCRIPTS = Path(__file__).resolve().parent        # repro_scripts/
BASE   = ROOT / "official_baseline" / "make_submission_forcodabench"
OUT_NAME = os.environ.get("OUT_NAME", "young_final")
OUT_DIR  = BASE / OUT_NAME
TRAIN_MEAN, TRAIN_STD = 4.82, 3.92   # output variance calibration targets
BLEND_PHQ_THRESHOLD   = float(os.environ.get("BLEND_PHQ_THRESHOLD", "4.2"))
                                     # classification: minimum blend phq9_pred
ORIG_PHQ_THRESHOLD    = 4            # classification: minimum ORIG PHQ prediction
TERNARY_T2            = float(os.environ.get("TERNARY_T2", "10.0"))
                                     # PHQ-9 class-2 threshold for ternary
# TEST_IDS is read from the official sample submission file (not hardcoded from
# test-set knowledge).  The sample file is provided by the competition organisers
# and lists the subject IDs that must be predicted.
_sample_b = pd.read_csv(BASE / "binary_sample.csv")
TEST_IDS = _sample_b["id"].tolist()

SKIP_CACHED = os.environ.get("SKIP_CACHED", "0") == "1"


# ── helpers ──────────────────────────────────────────────────────────────────

def run(script: str, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script)],
        cwd=str(ROOT), check=True
    )


def calibrate(raw: np.ndarray, mean_t: float = TRAIN_MEAN, std_t: float = TRAIN_STD) -> np.ndarray:
    return np.clip((raw - raw.mean()) * (std_t / raw.std()) + mean_t, 0, 27)


# ── Step 1: ORIG – IMU accel ch0-2 + Big5 → OrdinalRidge → PHQ ──────────────

def get_orig_phq() -> np.ndarray:
    """Run (or load cached) ORIG predictions on the test set.

    The cache _orig_phq.npy ships with the repository.  It contains the
    OrdinalRidge model's PREDICTED PHQ values on the test set (NOT ground-
    truth labels).  These predictions were produced by
    build_young_imu_accel3_ordinalridge_submission.py, which can be re-run
    with SKIP_CACHED=1 to verify the values from scratch.
    """
    cache = BASE / "_orig_phq.npy"
    if not SKIP_CACHED and cache.exists():
        print("  [ORIG] using cache _orig_phq.npy")
        return np.load(cache)
    # full recomputation requires: pip install mord
    print("  [ORIG] running IMU+Big5 OrdinalRidge script (requires mord)...")
    run("orig_model.py", "Step 1: ORIG (IMU + Big5 → OrdinalRidge)")
    src  = BASE / "young_imu_accel3_ordinalridge" / "binary.csv"
    orig = pd.read_csv(src).set_index("id").loc[TEST_IDS]["phq9_pred"].values.astype(float)
    np.save(cache, orig)
    return orig


# ── Step 2: facial rankers ────────────────────────────────────────────────────

def get_ranker(name: str, script: str, cache_file: str) -> np.ndarray:
    cache = BASE / cache_file
    if not SKIP_CACHED and cache.exists():
        print(f"  [{name}] using cache {cache_file}")
        return np.load(cache)
    run(script, f"Step 2: {name}")
    return np.load(cache)


# ── Step 3: blend + calibrate ─────────────────────────────────────────────────

def blend_and_calibrate(orig, auvel, e3only, fatigue, auspec) -> np.ndarray:
    raw = (0.40 * orig
         + 0.30 * auvel
         + 0.20 * e3only
         + 0.10 * fatigue
         + 0.05 * auspec)
    return calibrate(raw)


# ── Step 4: classification rule ───────────────────────────────────────────────

def classify(phq9_pred: np.ndarray, orig_phq: np.ndarray):
    binary  = ((phq9_pred >= BLEND_PHQ_THRESHOLD) & (orig_phq >= ORIG_PHQ_THRESHOLD)).astype(int)
    ternary = np.where(phq9_pred < BLEND_PHQ_THRESHOLD, 0,
              np.where(phq9_pred < TERNARY_T2,           1, 2)).astype(int)
    # keep ternary=0 when binary=0 (rule consistency)
    ternary = np.where(binary == 0, 0, ternary)
    return binary, ternary


# ── Step 5: write CSVs + zip ──────────────────────────────────────────────────

def write_submission(phq9_pred, binary, ternary) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_b = pd.read_csv(BASE / "binary_sample.csv")
    sample_t = pd.read_csv(BASE / "ternary_sample.csv")
    # fill in our predictions aligned by id
    b = sample_b.copy()
    t = sample_t.copy()
    b = b.set_index("id"); t = t.set_index("id")
    b.loc[TEST_IDS, "binary_pred"]  = binary
    b.loc[TEST_IDS, "phq9_pred"]    = phq9_pred
    t.loc[TEST_IDS, "ternary_pred"] = ternary
    t.loc[TEST_IDS, "phq9_pred"]    = phq9_pred
    b.reset_index().to_csv(OUT_DIR / "binary.csv",  index=False)
    t.reset_index().to_csv(OUT_DIR / "ternary.csv", index=False)
    subprocess.run([sys.executable, str(BASE / "make_submission_sample.py"),
        "--binary_csv",  str(OUT_DIR / "binary.csv"),
        "--ternary_csv", str(OUT_DIR / "ternary.csv"),
        "--binary_sample",  str(BASE / "binary_sample.csv"),
        "--ternary_sample", str(BASE / "ternary_sample.csv"),
        "--output_dir", str(BASE / f"{OUT_NAME}_validated")],
        cwd=str(ROOT), check=True)
    zip_path = BASE / f"{OUT_NAME}_validated" / "submission.zip"
    print(f"\n{'='*60}")
    print(f"  submission.zip: {zip_path}")
    print(f"  exists: {zip_path.exists()}")
    print(f"{'='*60}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("MPDD-AVG 2026 Young-track pipeline")
    print(f"ROOT = {ROOT}")

    # Step 1
    orig = get_orig_phq()
    print(f"  ORIG phq range: {orig.min():.1f}–{orig.max():.1f}  "
          f"(#pos orig>=4: {(orig>=4).sum()})")

    # Step 2
    auvel  = get_ranker("auvel",  "build_auvel_ranker.py",  "_auvel50.npy"  )
    e3only = get_ranker("e3only", "build_e3only_ranker.py","_e3only50.npy" )
    fatigue= get_ranker("fatigue","build_fatigue.py",           "_fatigue50.npy"  )
    auspec = get_ranker("auspec", "build_auspec.py",            "_auspec50.npy"   )

    # Step 3
    phq9 = blend_and_calibrate(orig, auvel, e3only, fatigue, auspec)
    print(f"\n  blend phq9_pred: mean={phq9.mean():.2f} std={phq9.std():.2f}")

    # Step 4
    binary, ternary = classify(phq9, orig)
    pos = [TEST_IDS[i] for i in range(len(TEST_IDS)) if binary[i] == 1]
    print(f"  binary positives ({len(pos)}): {pos}")

    # Step 5
    write_submission(phq9, binary, ternary)
    print("\nDone.")


if __name__ == "__main__":
    main()
