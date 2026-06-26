from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "高低维对比实验" / "3_OpenFace子集裁剪" / "openface_pruned_results.json"


def load_json() -> dict:
    return json.loads(SOURCE.read_text(encoding="utf-8"))


def build_subset_table(data: dict) -> pd.DataFrame:
    rows = []
    for name, r in data["feature_subset_results"].items():
        rows.append(
            {
                "subset": name,
                "n_features": int(r["n_features"]),
                "direct_ridge_binacc": float(r["direct_ridge_binacc"]),
                "direct_ridge_terf1": float(r["direct_ridge_terf1"]),
                "logistic_binacc": float(r["logistic_binacc"]),
                "logistic_binf1": float(r["logistic_binf1"]),
                "residual_binacc": float(r["residual_binacc"]),
                "residual_terf1": float(r["residual_terf1"]),
            }
        )
    return pd.DataFrame(rows)


def build_personality_table(data: dict) -> pd.DataFrame:
    r = data["personality_baseline"]
    return pd.DataFrame(
        [
            {
                "baseline": "Personality BigFive",
                "binary_accuracy": float(r["binary_accuracy"]),
                "binary_f1": float(r["binary_f1"]),
                "ternary_accuracy": float(r["ternary_accuracy"]),
                "ternary_f1": float(r["ternary_f1"]),
            }
        ]
    )


def build_au_tables(data: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corr_rows = []
    for au, payload in data["au_correlations"].items():
        label = payload.get("label", "")
        for stat, r in payload["correlations"].items():
            corr_rows.append(
                {
                    "au": au,
                    "label": label,
                    "stat": stat,
                    "r": float(r["r"]),
                    "abs_r": abs(float(r["r"])),
                    "pval": float(r["pval"]),
                    "bootstrap_std": float(r["bootstrap_std"]),
                    "ci95_low": float(r["ci_95"][0]),
                    "ci95_high": float(r["ci_95"][1]),
                    "stability": float(r["stability"]),
                }
            )
    corr = pd.DataFrame(corr_rows).sort_values(["abs_r", "stability"], ascending=[False, False])

    mean_rank = pd.DataFrame(
        [{"au": row[0], "mean_r": float(row[1]), "abs_mean_r": abs(float(row[1]))} for row in data["au_ranking_mean"]]
    )
    dyn = pd.DataFrame(
        [{"au": row[0], "stat": row[1], "r": float(row[2]), "abs_r": abs(float(row[2]))} for row in data["au_dynamic_range_top5"]]
    )
    return corr, mean_rank, dyn


def build_best_table(data: dict) -> pd.DataFrame:
    fs = data["feature_subset_results"]
    return pd.DataFrame(
        [
            {
                "criterion": "best_direct_ridge_binacc",
                "subset": data["best_direct_ridge"],
                "metric_value": fs[data["best_direct_ridge"]]["direct_ridge_binacc"],
            },
            {
                "criterion": "best_logistic_binacc",
                "subset": data["best_logistic"],
                "metric_value": fs[data["best_logistic"]]["logistic_binacc"],
            },
            {
                "criterion": "best_residual_binacc",
                "subset": data["best_residual"],
                "metric_value": fs[data["best_residual"]]["residual_binacc"],
            },
            {
                "criterion": "best_residual_ternary_f1",
                "subset": data["best_terf1"],
                "metric_value": fs[data["best_terf1"]]["residual_terf1"],
            },
        ]
    )


def plot_subset_metrics(subset: pd.DataFrame) -> None:
    order = subset.sort_values("n_features")["subset"].tolist()
    s = subset.set_index("subset").loc[order].reset_index()
    x = np.arange(len(s))

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    metrics1 = [
        ("direct_ridge_binacc", "Direct Ridge bin acc", "#0072b2"),
        ("logistic_binacc", "Logistic bin acc", "#d55e00"),
        ("residual_binacc", "Residual bin acc", "#009e73"),
    ]
    for metric, label, color in metrics1:
        axes[0].plot(x, s[metric], marker="o", linewidth=2.2, label=label, color=color)
        for xi, yi in zip(x, s[metric]):
            axes[0].text(xi, yi + 0.011, f"{yi:.2f}", ha="center", fontsize=7, color=color)
    axes[0].set_ylabel("Binary accuracy")
    axes[0].set_ylim(0.43, 0.86)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    axes[0].set_title("OpenFace semantic subsets: binary metrics")

    metrics2 = [
        ("direct_ridge_terf1", "Direct Ridge ternary F1", "#0072b2"),
        ("logistic_binf1", "Logistic binary F1", "#d55e00"),
        ("residual_terf1", "Residual ternary F1", "#009e73"),
    ]
    for metric, label, color in metrics2:
        axes[1].plot(x, s[metric], marker="o", linewidth=2.2, label=label, color=color)
        for xi, yi in zip(x, s[metric]):
            axes[1].text(xi, yi + 0.011, f"{yi:.2f}", ha="center", fontsize=7, color=color)
    axes[1].set_ylabel("F1")
    axes[1].set_ylim(0.25, 0.72)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, ncol=3, loc="upper left")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{name}\n{int(n)}d" for name, n in zip(s["subset"], s["n_features"])], rotation=0, fontsize=8)
    axes[1].set_title("OpenFace semantic subsets: F1 metrics")

    fig.suptitle("Complete OpenFace semantic subset pruning results", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "01_openface_subset_all_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_key_comparison(subset: pd.DataFrame, personality: pd.DataFrame) -> None:
    chosen = ["All_features", "Gaze_only", "AU_plus_Gaze", "AU_all_stats", "Dynamic_range", "Per_event"]
    s = subset.set_index("subset").loc[chosen].reset_index()
    x = np.arange(len(s))

    fig, ax = plt.subplots(figsize=(11, 5.2))
    width = 0.28
    ax.bar(x - width, s["direct_ridge_binacc"], width, label="Direct Ridge bin acc", color="#0072b2")
    ax.bar(x, s["residual_binacc"], width, label="Residual bin acc", color="#009e73")
    ax.bar(x + width, s["residual_terf1"], width, label="Residual ternary F1", color="#d55e00")
    pers_acc = float(personality["binary_accuracy"].iloc[0])
    ax.axhline(pers_acc, color="#555555", linestyle="--", linewidth=1.3, label=f"BigFive binary acc {pers_acc:.3f}")
    for i, r in s.iterrows():
        ax.text(i - width, r["direct_ridge_binacc"] + 0.01, f"{r['direct_ridge_binacc']:.2f}", ha="center", fontsize=8)
        ax.text(i, r["residual_binacc"] + 0.01, f"{r['residual_binacc']:.2f}", ha="center", fontsize=8)
        ax.text(i + width, r["residual_terf1"] + 0.01, f"{r['residual_terf1']:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.subset}\n{int(r.n_features)}d" for r in s.itertuples()], fontsize=8)
    ax.set_ylim(0.30, 0.92)
    ax.set_ylabel("Score")
    ax.set_title("Key OpenFace subset comparisons")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "02_openface_key_subset_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_au_rankings(mean_rank: pd.DataFrame, dyn: pd.DataFrame) -> None:
    top_mean = mean_rank.sort_values("abs_mean_r", ascending=False).head(10).iloc[::-1]
    top_dyn = dyn.sort_values("abs_r", ascending=False).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].barh(top_mean["au"], top_mean["mean_r"], color=np.where(top_mean["mean_r"] >= 0, "#0072b2", "#d55e00"))
    axes[0].axvline(0, color="#333333", linewidth=1)
    axes[0].set_title("Top AU mean correlations with PHQ-9")
    axes[0].set_xlabel("Pearson r")
    axes[0].grid(True, axis="x", alpha=0.25)
    for y, v in enumerate(top_mean["mean_r"]):
        axes[0].text(v + (0.008 if v >= 0 else -0.008), y, f"{v:.2f}", va="center",
                     ha="left" if v >= 0 else "right", fontsize=8)

    labels = [f"{r.au}_{r.stat}" for r in top_dyn.itertuples()]
    axes[1].barh(labels, top_dyn["r"], color=np.where(top_dyn["r"] >= 0, "#0072b2", "#d55e00"))
    axes[1].axvline(0, color="#333333", linewidth=1)
    axes[1].set_title("Dynamic-range top5 correlations")
    axes[1].set_xlabel("Pearson r")
    axes[1].grid(True, axis="x", alpha=0.25)
    for y, v in enumerate(top_dyn["r"]):
        axes[1].text(v + (0.008 if v >= 0 else -0.008), y, f"{v:.2f}", va="center",
                     ha="left" if v >= 0 else "right", fontsize=8)

    fig.suptitle("Per-AU signal summaries", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "03_openface_au_rankings.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_json()
    subset = build_subset_table(data)
    personality = build_personality_table(data)
    corr, mean_rank, dyn = build_au_tables(data)
    best = build_best_table(data)

    subset.to_csv(OUT / "openface_feature_subset_results_complete.csv", index=False)
    personality.to_csv(OUT / "openface_personality_baseline.csv", index=False)
    corr.to_csv(OUT / "openface_au_correlation_long.csv", index=False)
    mean_rank.to_csv(OUT / "openface_au_mean_ranking.csv", index=False)
    dyn.to_csv(OUT / "openface_dynamic_range_top5.csv", index=False)
    best.to_csv(OUT / "openface_best_methods.csv", index=False)

    plot_subset_metrics(subset)
    plot_key_comparison(subset, personality)
    plot_au_rankings(mean_rank, dyn)

    print(OUT)
    print(subset.sort_values("n_features").to_string(index=False))
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
