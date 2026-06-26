from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_young_strong_prior_router_9x3 import ITEMS, load_audit_text, strong_prior_score_v4


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "official_baseline" / "make_submission_forcodabench"
SAMPLE_BINARY = BASE / "binary_sample.csv"
SAMPLE_TERNARY = BASE / "ternary_sample.csv"
HELPER = BASE / "make_submission_sample.py"

DEFAULT_ASR_FEATURES = ROOT / "obs/experiments/young_asr_phq9_evidence_test/young_asr_phq9_evidence_features.csv"
DEFAULT_ASR_AUDIT = ROOT / "obs/experiments/young_asr_phq9_evidence_test/young_asr_phq9_evidence_audit.jsonl"
BIGFIVE = ROOT / "data/Train-MPDD-Young/Young/big_five_scores_extracted.csv"
OUT = BASE / "young_asr_v4_submission"


def read_sample_ids() -> list[int]:
    with SAMPLE_BINARY.open("r", encoding="utf-8-sig", newline="") as f:
        return [int(row["id"]) for row in csv.DictReader(f)]


def phq_to_binary(phq: np.ndarray) -> np.ndarray:
    return (np.asarray(phq, dtype=float) >= 5.0).astype(int)


def phq_to_ternary(phq: np.ndarray) -> np.ndarray:
    phq = np.asarray(phq, dtype=float)
    return np.where(phq >= 10.0, 2, np.where(phq >= 5.0, 1, 0)).astype(int)


def load_test_rows(asr_features: Path, asr_audit: Path) -> tuple[pd.DataFrame, dict[int, str]]:
    sample_ids = read_sample_ids()
    asr = pd.read_csv(asr_features)
    asr["ID"] = asr["ID"].astype(int)
    missing_asr = sorted(set(sample_ids) - set(asr["ID"].astype(int)))
    if missing_asr:
        raise ValueError(f"Missing ASR features for official test IDs: {missing_asr}")
    asr = asr[asr["ID"].isin(sample_ids)].copy()
    asr = asr.rename(columns={c: f"asr__{c}" for c in asr.columns if c != "ID"})

    bigfive = pd.read_csv(BIGFIVE)
    bigfive["ID"] = bigfive["ID"].astype(int)
    missing_bigfive = sorted(set(sample_ids) - set(bigfive["ID"].astype(int)))
    if missing_bigfive:
        raise ValueError(f"Missing Big-Five rows for official test IDs: {missing_bigfive}")
    bigfive = bigfive.rename(columns={c: f"bigfive__{c}" for c in bigfive.columns if c != "ID"})

    df = pd.DataFrame({"ID": sample_ids}).merge(asr, on="ID", how="left").merge(bigfive, on="ID", how="left")
    feature_cols = [c for c in df.columns if c != "ID"]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    texts = load_audit_text(asr_audit)
    missing_text = sorted(set(sample_ids) - set(texts))
    if missing_text:
        raise ValueError(f"Missing ASR audit text for official test IDs: {missing_text}")
    return df, texts


def write_submission(out_dir: Path, prediction_df: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = prediction_df["id"].astype(int).to_numpy()
    phq = prediction_df["phq9_pred"].astype(float).clip(0.0, 27.0).to_numpy()
    binary = pd.DataFrame({"id": ids, "binary_pred": phq_to_binary(phq), "phq9_pred": np.round(phq, 6)})
    ternary = pd.DataFrame({"id": ids, "ternary_pred": phq_to_ternary(phq), "phq9_pred": np.round(phq, 6)})
    binary.to_csv(out_dir / "binary.csv", index=False)
    ternary.to_csv(out_dir / "ternary.csv", index=False)

    validated = Path(str(out_dir) + "_validated")
    subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "--binary_csv",
            str(out_dir / "binary.csv"),
            "--ternary_csv",
            str(out_dir / "ternary.csv"),
            "--binary_sample",
            str(SAMPLE_BINARY),
            "--ternary_sample",
            str(SAMPLE_TERNARY),
            "--output_dir",
            str(validated),
        ],
        check=True,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build Young test submission from fixed ASR V4 strong prior.")
    parser.add_argument("--asr_features", type=Path, default=DEFAULT_ASR_FEATURES)
    parser.add_argument("--asr_audit", type=Path, default=DEFAULT_ASR_AUDIT)
    parser.add_argument("--out_dir", type=Path, default=OUT)
    args = parser.parse_args()

    df, texts = load_test_rows(args.asr_features, args.asr_audit)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        sid = int(row["ID"])
        result = strong_prior_score_v4(row, texts[sid])
        item_scores = {f"v4_prior_{item}": int(result.item_scores[item]) for item in ITEMS}
        rows.append(
            {
                "id": sid,
                "phq9_pred": float(np.clip(result.raw_sum, 0.0, 27.0)),
                "v4_region": result.region,
                "v4_item_vector_phq1_to_phq9": "|".join(str(item_scores[f"v4_prior_{item}"]) for item in ITEMS),
                "asr_text_excerpt": texts[sid][:300],
                **item_scores,
            }
        )
    pred = pd.DataFrame(rows)
    pred = pd.DataFrame({"id": read_sample_ids()}).merge(pred, on="id", how="left")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out_dir / "asr_v4_prediction_detail.csv", index=False, encoding="utf-8-sig")
    write_submission(args.out_dir, pred[["id", "phq9_pred"]])

    report = {
        "asr_features": str(args.asr_features),
        "asr_audit": str(args.asr_audit),
        "output_dir": str(args.out_dir),
        "validated_submission_zip": str(Path(str(args.out_dir) + "_validated") / "submission.zip"),
        "n_ids": int(len(pred)),
        "phq_summary": {
            "min": float(pred["phq9_pred"].min()),
            "mean": float(pred["phq9_pred"].mean()),
            "max": float(pred["phq9_pred"].max()),
        },
        "region_counts": {str(k): int(v) for k, v in pred["v4_region"].value_counts().sort_index().items()},
        "binary_counts": {
            str(k): int(v)
            for k, v in pd.Series(phq_to_binary(pred["phq9_pred"].to_numpy(float))).value_counts().sort_index().items()
        },
        "ternary_counts": {
            str(k): int(v)
            for k, v in pd.Series(phq_to_ternary(pred["phq9_pred"].to_numpy(float))).value_counts().sort_index().items()
        },
    }
    (args.out_dir / "submission_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (Path(str(args.out_dir) + "_validated") / "submission_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
