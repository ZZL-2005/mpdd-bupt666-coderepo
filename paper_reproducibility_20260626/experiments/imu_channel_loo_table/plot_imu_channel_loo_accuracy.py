from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SUMMARY_CSV = HERE / "loo_accuracy_summary.csv"
PRED_CSV = HERE / "loo_predictions.csv"


def savefig(path: Path) -> None:
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(path)


def plot_regression(summary: pd.DataFrame) -> None:
    channels = sorted(summary["n_channels"].unique())
    colors = {"linear": "#d55e00", "ridge": "#0072b2"}
    labels = {"linear": "LinearRegression", "ridge": "Ridge(alpha=1.0)"}
    metrics = [
        ("mae_clipped", "LOO MAE, lower is better"),
        ("rmse_clipped", "LOO RMSE, lower is better"),
        ("ccc_clipped", "LOO CCC, higher is better"),
        ("within_3_phq_acc", "PHQ within +/-3 accuracy, higher is better"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for model in ["linear", "ridge"]:
            sub = summary[summary["model"] == model].sort_values("n_channels")
            ax.plot(sub["n_channels"], sub[metric], marker="o", linewidth=2.4,
                    color=colors[model], label=labels[model])
            for x, y in zip(sub["n_channels"], sub[metric]):
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=8, color=colors[model])
        ax.set_title(title)
        ax.set_xticks(channels)
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("Number of IMU channels")
    axes[1, 1].set_xlabel("Number of IMU channels")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Leave-One-Out PHQ-9 regression accuracy", y=1.02, fontsize=14)
    fig.tight_layout()
    savefig(HERE / "10_loo_regression_accuracy.png")


def plot_classification(summary: pd.DataFrame) -> None:
    channels = sorted(summary["n_channels"].unique())
    colors = {"linear": "#d55e00", "ridge": "#0072b2"}
    labels = {"linear": "LinearRegression", "ridge": "Ridge(alpha=1.0)"}
    metrics = [
        ("label2_acc", "Binary label accuracy"),
        ("label2_f1", "Binary label F1"),
        ("label3_acc", "3-class label accuracy"),
        ("label3_quadratic_kappa", "3-class quadratic kappa"),
        ("challenge_like_score", "Challenge-like score"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes_flat = axes.ravel()
    for ax, (metric, title) in zip(axes_flat, metrics):
        for model in ["linear", "ridge"]:
            sub = summary[summary["model"] == model].sort_values("n_channels")
            ax.plot(sub["n_channels"], sub[metric], marker="o", linewidth=2.4,
                    color=colors[model], label=labels[model])
            for x, y in zip(sub["n_channels"], sub[metric]):
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=8, color=colors[model])
        ax.set_title(title)
        ax.set_xticks(channels)
        ax.grid(True, alpha=0.25)
    axes_flat[-1].axis("off")
    for ax in axes[1, :2]:
        ax.set_xlabel("Number of IMU channels")
    axes_flat[0].legend(frameon=False)
    fig.suptitle("Leave-One-Out derived classification accuracy", y=1.02, fontsize=14)
    fig.tight_layout()
    savefig(HERE / "11_loo_classification_accuracy.png")


def plot_error_boxplot(pred: pd.DataFrame) -> None:
    channels = [3, 6, 9, 12]
    colors = {"linear": "#d55e00", "ridge": "#0072b2"}
    labels = {"linear": "LinearRegression", "ridge": "Ridge(alpha=1.0)"}
    pred = pred.copy()
    pred["abs_error"] = (pred["phq9_true"] - pred["phq9_pred_clipped"]).abs()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, model in zip(axes, ["linear", "ridge"]):
        data = [
            pred[(pred["model"] == model) & (pred["n_channels"] == ch)]["abs_error"].values
            for ch in channels
        ]
        bp = ax.boxplot(data, tick_labels=[str(ch) for ch in channels], patch_artist=True, showfliers=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[model])
            patch.set_alpha(0.35)
            patch.set_edgecolor(colors[model])
        for key in ["whiskers", "caps", "medians"]:
            for line in bp[key]:
                line.set_color(colors[model])
                line.set_linewidth(1.4)
        ax.set_title(labels[model])
        ax.set_xlabel("Number of IMU channels")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Absolute PHQ-9 error")
    fig.suptitle("LOO absolute error distribution over 88 training subjects", y=1.03, fontsize=14)
    fig.tight_layout()
    savefig(HERE / "12_loo_absolute_error_boxplot.png")


def plot_scatter(pred: pd.DataFrame) -> None:
    channels = [3, 6, 9, 12]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    for row, model in enumerate(["linear", "ridge"]):
        for col, ch in enumerate(channels):
            ax = axes[row, col]
            sub = pred[(pred["model"] == model) & (pred["n_channels"] == ch)]
            ax.scatter(sub["phq9_true"], sub["phq9_pred_clipped"], s=18, alpha=0.75,
                       color="#d55e00" if model == "linear" else "#0072b2")
            ax.plot([0, 27], [0, 27], color="#444444", linewidth=1, alpha=0.6)
            ax.set_title(f"{model}, {ch}ch")
            ax.grid(True, alpha=0.2)
            if row == 1:
                ax.set_xlabel("True PHQ-9")
            if col == 0:
                ax.set_ylabel("LOO predicted PHQ-9")
    fig.suptitle("LOO true vs predicted PHQ-9", y=1.02, fontsize=14)
    fig.tight_layout()
    savefig(HERE / "13_loo_true_vs_pred_scatter.png")


def main() -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    pred = pd.read_csv(PRED_CSV)
    plot_regression(summary)
    plot_classification(summary)
    plot_error_boxplot(pred)
    plot_scatter(pred)


if __name__ == "__main__":
    main()
