#!/usr/bin/env python
"""Build Young submission with accel-3 IMU + Big-Five OrdinalRidge.

This reproduces the user-provided 0.6662-style configuration: only the first
three IMU channels are used for per-dimension band/autocorrelation/peak
features, plus aggregate acceleration magnitude/correlation features and parsed
Big-Five scores.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mord import OrdinalRidge
from scipy import signal
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/Train-MPDD-Young/Young"
TEST = ROOT / "data/Test-MPDD-Young/Young"
BASE = ROOT / "official_baseline/make_submission_forcodabench"
HELPER = BASE / "make_submission_sample.py"
SAMPLE_BINARY = BASE / "binary_sample.csv"
SAMPLE_TERNARY = BASE / "ternary_sample.csv"
ORDINAL_ALPHA = float(os.environ.get("ORDINAL_ALPHA", "1.0"))
FEATURE_SET = os.environ.get("FEATURE_SET", "all")
OUT_NAME = os.environ.get("OUT_NAME", "young_imu_accel3_ordinalridge")
OUT_DIR = BASE / OUT_NAME
TEST_IDS = [1, 5, 7, 13, 15, 22, 28, 33, 34, 40, 42, 44, 47, 58, 74, 83, 85, 89, 90, 93, 105, 110]


def bandpass_filter(sig: np.ndarray, lowcut: float, highcut: float, fs: int = 50, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, sig)


def extract_fast_imu_features(imu_data: np.ndarray, fs: int = 50) -> dict[str, float]:
    features: dict[str, float] = {}
    for dim in range(3):
        sig = imu_data[:, dim]
        low_band = bandpass_filter(sig, 0.5, 1.5, fs)
        mid_band = bandpass_filter(sig, 1.5, 3.0, fs)
        high_band = bandpass_filter(sig, 3.0, 10.0, fs)

        total_energy = np.sum(sig**2)
        features[f"dim{dim}_low_band_energy_ratio"] = float(np.sum(low_band**2) / (total_energy + 1e-8))
        features[f"dim{dim}_mid_band_energy_ratio"] = float(np.sum(mid_band**2) / (total_energy + 1e-8))
        features[f"dim{dim}_high_band_energy_ratio"] = float(np.sum(high_band**2) / (total_energy + 1e-8))
        features[f"dim{dim}_low_band_cv"] = float(np.std(low_band) / (np.abs(np.mean(low_band)) + 1e-8))
        features[f"dim{dim}_mid_band_cv"] = float(np.std(mid_band) / (np.abs(np.mean(mid_band)) + 1e-8))

        autocorr = np.correlate(sig - np.mean(sig), sig - np.mean(sig), mode="full")
        autocorr = autocorr[len(autocorr) // 2 :] / autocorr[len(autocorr) // 2]
        peaks, _ = signal.find_peaks(autocorr[1:100], height=0.3)
        features[f"dim{dim}_autocorr_first_peak"] = float(autocorr[peaks[0] + 1]) if len(peaks) > 0 else 0.0
        features[f"dim{dim}_dominant_period"] = float((peaks[0] + 1) / fs) if len(peaks) > 0 else 0.0

        peaks, _ = signal.find_peaks(sig, distance=fs // 4, prominence=0.5)
        if len(peaks) > 1:
            intervals = np.diff(peaks) / fs
            features[f"dim{dim}_peak_interval_mean"] = float(np.mean(intervals))
            features[f"dim{dim}_peak_interval_std"] = float(np.std(intervals))
            features[f"dim{dim}_peak_interval_cv"] = float(np.std(intervals) / (np.mean(intervals) + 1e-8))
        else:
            features[f"dim{dim}_peak_interval_mean"] = 0.0
            features[f"dim{dim}_peak_interval_std"] = 0.0
            features[f"dim{dim}_peak_interval_cv"] = 0.0

    accel_x, accel_y, accel_z = imu_data[:, 0], imu_data[:, 1], imu_data[:, 2]
    features["corr_xy"] = float(np.corrcoef(accel_x, accel_y)[0, 1])
    features["corr_xz"] = float(np.corrcoef(accel_x, accel_z)[0, 1])
    features["corr_yz"] = float(np.corrcoef(accel_y, accel_z)[0, 1])

    accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    features["accel_mag_mean"] = float(np.mean(accel_mag))
    features["accel_mag_std"] = float(np.std(accel_mag))
    features["accel_mag_cv"] = float(np.std(accel_mag) / (np.mean(accel_mag) + 1e-8))
    return features


def extract_personality_scores_robust(text: str) -> dict[str, float]:
    scores: dict[str, float] = {}

    match1 = re.search(r"Agreeableness and Conscientiousness scores of (\d+)", text)
    if match1:
        score = float(match1.group(1))
        scores["Agreeableness"] = score
        scores["Conscientiousness"] = score

    match2 = re.search(r"Agreeableness and Conscientiousness scores are both (\d+)", text)
    if match2:
        score = float(match2.group(1))
        scores["Agreeableness"] = score
        scores["Conscientiousness"] = score

    match3 = re.search(r"Agreeableness, Conscientiousness, and Neuroticism scores are all (\d+)", text)
    if match3:
        score = float(match3.group(1))
        scores["Agreeableness"] = score
        scores["Conscientiousness"] = score
        scores["Neuroticism"] = score

    for trait in ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]:
        if trait not in scores:
            match = re.search(rf"{trait} score of (\d+)", text)
            scores[trait] = float(match.group(1)) if match else np.nan
    return scores


def load_feature_table(split: str, labels: pd.DataFrame | None, descriptions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ids = labels["ID"].astype(int).tolist() if labels is not None else TEST_IDS
    for sample_id in ids:
        if split == "train":
            imu_file = TRAIN / "IMU/train" / str(sample_id) / f"{sample_id}.npy"
        else:
            imu_file = TEST / "IMU" / str(sample_id) / f"{sample_id}.npy"
        if not imu_file.exists():
            continue

        features = extract_fast_imu_features(np.load(imu_file))
        desc_row = descriptions[descriptions["ID"].astype(int) == int(sample_id)]
        if len(desc_row) == 0:
            continue
        features.update(extract_personality_scores_robust(str(desc_row.iloc[0]["Descriptions"])))
        features["ID"] = int(sample_id)
        if labels is not None:
            label_row = labels[labels["ID"].astype(int) == int(sample_id)].iloc[0]
            features["phq9_score"] = float(label_row["phq9_score"])
        rows.append(features)
    return pd.DataFrame(rows).dropna()


def phq_to_binary(phq: np.ndarray) -> np.ndarray:
    return (np.asarray(phq, dtype=float) >= 5.0).astype(int)


def phq_to_ternary(phq: np.ndarray) -> np.ndarray:
    phq = np.asarray(phq, dtype=float)
    return np.where(phq >= 10.0, 2, np.where(phq >= 5.0, 1, 0)).astype(int)


def select_feature_cols(train_df: pd.DataFrame, feature_set: str) -> list[str]:
    all_cols = [col for col in train_df.columns if col not in ["ID", "phq9_score"]]
    if feature_set == "all":
        return all_cols

    personality = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
    if feature_set == "no_cv":
        return [col for col in all_cols if not col.endswith("_cv")]
    if feature_set == "energy_period_corr_mag_personality":
        return [
            col
            for col in all_cols
            if (
                "energy_ratio" in col
                or "autocorr" in col
                or "dominant_period" in col
                or col.startswith("corr_")
                or col.startswith("accel_mag")
                or col in personality
            )
        ]
    if feature_set == "energy_corr_mag_personality":
        return [
            col
            for col in all_cols
            if ("energy_ratio" in col or col.startswith("corr_") or col.startswith("accel_mag") or col in personality)
        ]
    if feature_set.startswith("top") and feature_set.endswith("_spearman"):
        top_k = int(feature_set.removeprefix("top").removesuffix("_spearman"))
        y = train_df["phq9_score"].values
        ranked = []
        for col in all_cols:
            rho = pd.Series(train_df[col]).corr(pd.Series(y), method="spearman")
            ranked.append((col, abs(float(rho))))
        return [col for col, _ in sorted(ranked, key=lambda item: item[1], reverse=True)[:top_k]]
    raise ValueError(f"Unknown FEATURE_SET={feature_set!r}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(TRAIN / "split_labels_train.csv")
    descriptions = pd.read_csv(TRAIN / "descriptions.csv")

    train_df = load_feature_table("train", labels, descriptions)
    test_df = load_feature_table("test", None, descriptions)
    feature_cols = select_feature_cols(train_df, FEATURE_SET)

    x_train = train_df[feature_cols].values
    y_phq9_train = train_df["phq9_score"].values
    unique_phq9 = np.sort(np.unique(y_phq9_train))
    phq9_to_ordinal = {score: i for i, score in enumerate(unique_phq9)}
    ordinal_to_phq9 = {i: score for i, score in enumerate(unique_phq9)}
    y_ordinal_train = np.array([phq9_to_ordinal[score] for score in y_phq9_train])

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    model = OrdinalRidge(alpha=ORDINAL_ALPHA)
    model.fit(x_train_scaled, y_ordinal_train)

    x_test_scaled = scaler.transform(test_df[feature_cols].values)
    y_ordinal_pred = model.predict(x_test_scaled)
    y_phq9_pred = np.array([ordinal_to_phq9[int(o)] for o in y_ordinal_pred], dtype=float)
    binary_pred = phq_to_binary(y_phq9_pred)
    ternary_pred = phq_to_ternary(y_phq9_pred)

    binary_df = pd.DataFrame(
        {
            "id": test_df["ID"].astype(int).values,
            "binary_pred": binary_pred.astype(int),
            "phq9_pred": y_phq9_pred,
        }
    )
    ternary_df = pd.DataFrame(
        {
            "id": test_df["ID"].astype(int).values,
            "ternary_pred": ternary_pred.astype(int),
            "phq9_pred": y_phq9_pred,
        }
    )
    detail_df = test_df[["ID"]].rename(columns={"ID": "id"}).copy()
    detail_df["ordinal_pred"] = y_ordinal_pred.astype(int)
    detail_df["phq9_pred"] = y_phq9_pred
    detail_df["binary_pred"] = binary_pred.astype(int)
    detail_df["ternary_pred"] = ternary_pred.astype(int)

    binary_df.to_csv(OUT_DIR / "binary.csv", index=False, encoding="utf-8")
    ternary_df.to_csv(OUT_DIR / "ternary.csv", index=False, encoding="utf-8")
    detail_df.to_csv(OUT_DIR / "prediction_detail.csv", index=False, encoding="utf-8-sig")

    validated = Path(str(OUT_DIR) + "_validated")
    subprocess.run(
        [
            str(Path(sys.executable)),
            str(HELPER),
            "--binary_csv",
            str(OUT_DIR / "binary.csv"),
            "--ternary_csv",
            str(OUT_DIR / "ternary.csv"),
            "--binary_sample",
            str(SAMPLE_BINARY),
            "--ternary_sample",
            str(SAMPLE_TERNARY),
            "--output_dir",
            str(validated),
        ],
        cwd=ROOT,
        check=True,
    )

    report = {
        "method": "imu_accel3_bigfive_ordinalridge",
        "ordinal_alpha": float(ORDINAL_ALPHA),
        "feature_set": FEATURE_SET,
        "train_n": int(len(train_df)),
        "test_n": int(len(test_df)),
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "unique_train_phq9": [float(x) for x in unique_phq9.tolist()],
        "phq_summary": {
            "min": float(np.min(y_phq9_pred)),
            "mean": float(np.mean(y_phq9_pred)),
            "max": float(np.max(y_phq9_pred)),
        },
        "binary_counts": {str(k): int(v) for k, v in pd.Series(binary_pred).value_counts().sort_index().items()},
        "ternary_counts": {str(k): int(v) for k, v in pd.Series(ternary_pred).value_counts().sort_index().items()},
        "output_dir": str(OUT_DIR),
        "validated_submission_zip": str(validated / "submission.zip"),
    }
    (OUT_DIR / "submission_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (validated / "submission_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(detail_df.to_string(index=False))


if __name__ == "__main__":
    main()
