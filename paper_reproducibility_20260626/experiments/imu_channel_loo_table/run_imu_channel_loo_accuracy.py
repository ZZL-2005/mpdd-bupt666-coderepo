#!/usr/bin/env python
"""LOO accuracy experiment for IMU-only channel-count settings.

For each first-k IMU channel setting (k = 3/6/9/12), this script extracts the
same ORIG-style IMU hand-crafted features and evaluates deterministic
Leave-One-Out predictions on the Young training set.

No BigFive/personality features are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from run_imu_channel_seed_stability import find_paths, load_table, make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"E:\mpdd2026", help="MPDD workspace root.")
    parser.add_argument("--out-dir", default=".", help="Directory for result files.")
    parser.add_argument("--channel-counts", default="3,6,9,12")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--fs", type=int, default=50)
    return parser.parse_args()


def concordance_cc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_true = y_true.mean()
    mean_pred = y_pred.mean()
    var_true = np.mean((y_true - mean_true) ** 2)
    var_pred = np.mean((y_pred - mean_pred) ** 2)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom <= 1e-12:
        return 0.0
    return float(2.0 * cov / denom)


def phq_to_label2(phq: np.ndarray) -> np.ndarray:
    return (np.asarray(phq, dtype=float) >= 5.0).astype(int)


def phq_to_label3(phq: np.ndarray) -> np.ndarray:
    phq = np.asarray(phq, dtype=float)
    out = np.zeros_like(phq, dtype=int)
    out[(phq >= 5.0) & (phq < 10.0)] = 1
    out[phq >= 10.0] = 2
    return out


def compute_metrics(y_true: np.ndarray, pred_raw: np.ndarray, pred_clipped: np.ndarray,
                    label2_true: np.ndarray, label3_true: np.ndarray) -> dict[str, float]:
    pred_label2 = phq_to_label2(pred_clipped)
    pred_label3 = phq_to_label3(pred_clipped)

    rmse_raw = mean_squared_error(y_true, pred_raw) ** 0.5
    rmse_clipped = mean_squared_error(y_true, pred_clipped) ** 0.5
    ccc_clipped = concordance_cc(y_true, pred_clipped)
    f1_binary = f1_score(label2_true, pred_label2, zero_division=0)
    kappa3_quadratic = cohen_kappa_score(label3_true, pred_label3, weights="quadratic")

    return {
        "mae_raw": float(mean_absolute_error(y_true, pred_raw)),
        "rmse_raw": float(rmse_raw),
        "ccc_raw": concordance_cc(y_true, pred_raw),
        "r2_raw": float(r2_score(y_true, pred_raw)),
        "mae_clipped": float(mean_absolute_error(y_true, pred_clipped)),
        "rmse_clipped": float(rmse_clipped),
        "ccc_clipped": ccc_clipped,
        "r2_clipped": float(r2_score(y_true, pred_clipped)),
        "within_2_phq_acc": float(np.mean(np.abs(y_true - pred_clipped) <= 2.0)),
        "within_3_phq_acc": float(np.mean(np.abs(y_true - pred_clipped) <= 3.0)),
        "label2_acc": float(accuracy_score(label2_true, pred_label2)),
        "label2_balanced_acc": float(balanced_accuracy_score(label2_true, pred_label2)),
        "label2_f1": float(f1_binary),
        "label3_acc": float(accuracy_score(label3_true, pred_label3)),
        "label3_macro_f1": float(f1_score(label3_true, pred_label3, average="macro", zero_division=0)),
        "label3_kappa": float(cohen_kappa_score(label3_true, pred_label3)),
        "label3_quadratic_kappa": float(kappa3_quadratic),
        "challenge_like_score": float((f1_binary + ccc_clipped + kappa3_quadratic) / 3.0),
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_counts = [int(x) for x in args.channel_counts.split(",") if x.strip()]
    model_names = ["linear", "ridge"]
    paths = find_paths(data_root)

    labels = pd.read_csv(paths["labels"]).sort_values("ID")
    train_ids = labels["ID"].astype(int).tolist()
    y = labels["phq9_score"].to_numpy(float)
    label2 = labels["label2"].astype(int).to_numpy()
    label3 = labels["label3"].astype(int).to_numpy()

    pred_rows = []
    summary_rows = []
    metadata = {
        "data_root": str(data_root),
        "out_dir": str(out_dir.resolve()),
        "channel_counts": channel_counts,
        "models": model_names,
        "cv": "LeaveOneOut",
        "ridge_alpha": args.ridge_alpha,
        "fs": args.fs,
        "train_n": len(train_ids),
        "uses_bigfive": False,
        "label2_threshold": "predicted phq9 >= 5",
        "label3_thresholds": "0: phq9 < 5; 1: 5 <= phq9 < 10; 2: phq9 >= 10",
        "challenge_like_score": "(label2_f1 + ccc_clipped + label3_quadratic_kappa) / 3",
    }

    loo = LeaveOneOut()
    for n_channels in channel_counts:
        print(f"[features] n_channels={n_channels}")
        train_df = load_table("train", paths, train_ids, n_channels=n_channels, fs=args.fs).set_index("ID").loc[train_ids]
        feature_cols = list(train_df.columns)
        x = train_df[feature_cols].to_numpy(float)
        feature_dim = len(feature_cols)

        for model_name in model_names:
            print(f"  [model] {model_name}, feature_dim={feature_dim}")
            preds_raw = np.zeros(len(train_ids), dtype=float)
            for fold, (fit_idx, held_idx) in enumerate(loo.split(x)):
                held_pos = int(held_idx[0])
                scaler = StandardScaler()
                x_fit = scaler.fit_transform(x[fit_idx])
                x_held = scaler.transform(x[held_idx])
                model = make_model(model_name, args.ridge_alpha)
                model.fit(x_fit, y[fit_idx])
                preds_raw[held_pos] = float(model.predict(x_held)[0])

                pred_clipped = float(np.clip(preds_raw[held_pos], 0.0, 27.0))
                pred_rows.append(
                    {
                        "n_channels": n_channels,
                        "feature_dim": feature_dim,
                        "model": model_name,
                        "fold": fold,
                        "id": int(train_ids[held_pos]),
                        "phq9_true": float(y[held_pos]),
                        "phq9_pred_raw": float(preds_raw[held_pos]),
                        "phq9_pred_clipped": pred_clipped,
                        "label2_true": int(label2[held_pos]),
                        "label2_pred": int(phq_to_label2([pred_clipped])[0]),
                        "label3_true": int(label3[held_pos]),
                        "label3_pred": int(phq_to_label3([pred_clipped])[0]),
                    }
                )

            preds_clipped = np.clip(preds_raw, 0.0, 27.0)
            metrics = compute_metrics(y, preds_raw, preds_clipped, label2, label3)
            summary_rows.append(
                {
                    "n_channels": n_channels,
                    "feature_dim": feature_dim,
                    "model": model_name,
                    "n_train": len(train_ids),
                    **metrics,
                }
            )

    pred_df = pd.DataFrame(pred_rows)
    summary = pd.DataFrame(summary_rows).sort_values(["model", "n_channels"])

    pred_df.to_csv(out_dir / "loo_predictions.csv", index=False)
    summary.to_csv(out_dir / "loo_accuracy_summary.csv", index=False)
    with open(out_dir / "loo_accuracy_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nLOO accuracy summary:")
    cols = [
        "n_channels",
        "model",
        "feature_dim",
        "mae_clipped",
        "rmse_clipped",
        "ccc_clipped",
        "label2_acc",
        "label2_f1",
        "label3_acc",
        "label3_quadratic_kappa",
        "challenge_like_score",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote LOO results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
