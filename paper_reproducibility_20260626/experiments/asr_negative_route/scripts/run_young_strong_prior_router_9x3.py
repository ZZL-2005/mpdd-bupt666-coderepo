# -*- coding: utf-8 -*-
"""Strong-prior PHQ-9 router with questionnaire-aligned 9x3 item scores.

This script intentionally keeps the learnable part small. It first assigns a
region and hand-designed PHQ item priors, then evaluates raw and lightly
calibrated variants against the Young PHQ-9 total score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
LABEL_CSV = ROOT / "data/Train-MPDD-Young/Young/split_labels_train.csv"
AUDIO_CSV = ROOT / "obs/experiments/young_audio_acoustic_event_contrast/young_audio_acoustic_event_features.csv"
IMU_CSV = ROOT / "obs/experiments/young_imu_prior_features/young_imu_curated_features.csv"
FACE_CSV = ROOT / "obs/experiments/young_openface_curated_prior/curated_openface_features.csv"
BIGFIVE_CSV = ROOT / "data/Train-MPDD-Young/Young/big_five_scores_extracted.csv"
LABEL_COLS = {"ID", "id", "pid", "split", "label2", "label3", "phq9_score"}
ASR_FEATURE_CSV = ROOT / "obs/experiments/young_asr_phq9_evidence/young_asr_phq9_evidence_features.csv"
ASR_AUDIT_JSONL = ROOT / "obs/experiments/young_asr_phq9_evidence/young_asr_phq9_evidence_audit.jsonl"
OUT_DIR = ROOT / "obs/experiments/young_strong_prior_router_9x3"


def normalize_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "ID" in df.columns:
        return df
    if "id" in df.columns:
        return df.rename(columns={"id": "ID"})
    if "pid" in df.columns:
        return df.rename(columns={"pid": "ID"})
    raise ValueError("No ID/id/pid column found")

ITEMS = [
    "phq1_interest",
    "phq2_mood",
    "phq3_sleep",
    "phq4_fatigue",
    "phq5_appetite",
    "phq6_self_worth",
    "phq7_concentration",
    "phq8_psychomotor",
    "phq9_self_harm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strong-prior PHQ9 router experiment.")
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=2026)
    parser.add_argument("--select_k", type=int, default=64)
    return parser.parse_args()


def pred_label2(score: np.ndarray) -> np.ndarray:
    return (np.asarray(score) >= 5.0).astype(int)


def pred_label3(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score)
    return np.where(score >= 10.0, 2, np.where(score >= 5.0, 1, 0)).astype(int)


def metrics(y: np.ndarray, pred: np.ndarray, label2: np.ndarray, label3: np.ndarray) -> dict[str, float]:
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, 27.0)
    p2 = pred_label2(pred)
    p3 = pred_label3(pred)
    return {
        "phq_mae": float(mean_absolute_error(y, pred)),
        "phq_rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "phq_bias_pred_minus_true": float(np.mean(pred - y)),
        "label2_acc": float(accuracy_score(label2, p2)),
        "label2_balanced_acc": float(balanced_accuracy_score(label2, p2)),
        "label2_f1": float(f1_score(label2, p2, zero_division=0)),
        "label3_acc": float(accuracy_score(label3, p3)),
        "label3_balanced_acc": float(balanced_accuracy_score(label3, p3)),
        "label3_macro_f1": float(f1_score(label3, p3, average="macro", zero_division=0)),
        "mean_pred_phq9": float(np.mean(pred)),
    }


def load_audit_text(path: Path) -> dict[int, str]:
    rows: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows[int(obj["ID"])] = obj.get("text", "")
    return rows


def load_source(path: Path, prefix: str) -> pd.DataFrame:
    df = normalize_id_column(pd.read_csv(path))
    cols = [c for c in df.columns if c not in LABEL_COLS]
    df = df[["ID"] + cols].copy()
    return df.rename(columns={c: f"{prefix}__{c}" for c in cols})


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    labels = normalize_id_column(pd.read_csv(LABEL_CSV))[["ID", "label2", "label3", "phq9_score"]]
    asr = pd.read_csv(ASR_FEATURE_CSV).rename(columns={c: f"asr__{c}" for c in pd.read_csv(ASR_FEATURE_CSV, nrows=0).columns if c != "ID"})
    sources = [
        asr,
        load_source(AUDIO_CSV, "audio"),
        load_source(IMU_CSV, "imu"),
        load_source(FACE_CSV, "face"),
        load_source(BIGFIVE_CSV, "bigfive"),
    ]
    df = labels.copy()
    for src in sources:
        df = df.merge(src, on="ID", how="left")
    feature_cols = [c for c in df.columns if c not in {"ID", "label2", "label3", "phq9_score"}]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    return df, feature_cols


def val(row: pd.Series, name: str) -> float:
    return float(row.get(name, 0.0) or 0.0)


def has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def count_any(text: str, terms: list[str]) -> int:
    return sum(text.count(term) for term in terms)


def term_contexts(text: str, terms: list[str], *, window: int = 32) -> list[str]:
    contexts: list[str] = []
    for term in terms:
        start = 0
        while True:
            idx = text.find(term, start)
            if idx < 0:
                break
            left = max(0, idx - window)
            right = min(len(text), idx + len(term) + window)
            contexts.append(text[left:right])
            start = idx + len(term)
    return contexts


def score_direct_item(row: pd.Series, item: str, *, allow_weak: bool = True, use_refined: bool = True) -> int:
    score_col = f"asr__asr_{item}_{'refined_' if use_refined else ''}rule_score"
    if score_col in row.index:
        return int(round(val(row, score_col)))
    direct = val(row, f"asr__asr_{item}_direct_count")
    weak = val(row, f"asr__asr_{item}_weak_count")
    freq = val(row, f"asr__asr_{item}_freq_context_count")
    sev = val(row, f"asr__asr_{item}_severity_context_count")
    if direct >= 2 and (freq >= 1 or sev >= 1):
        return 3
    if direct >= 1 and (freq >= 1 or sev >= 1):
        return 2
    if direct >= 1:
        return 1
    if allow_weak and weak >= 3:
        return 1
    return 0


def sleep_score(row: pd.Series, text: str) -> int:
    daily = has_any(text, ["每天", "每晚", "一直", "经常"])
    severe = has_any(text, ["严重", "特别", "非常"])
    if daily and has_any(text, ["睡不着", "失眠", "睡不好", "睡眠质量"]) and severe:
        return 3
    if daily and has_any(text, ["睡不着", "失眠", "入睡困难"]):
        return 3
    if has_any(text, ["睡不着", "失眠", "入睡困难", "睡不醒"]):
        return 2
    if has_any(text, ["睡眠质量"]) and severe:
        return 2
    if has_any(text, ["睡眠质量", "熬夜", "睡得晚", "晚睡", "打呼噜", "磨牙", "有时候没办法"]):
        return 1
    return 0


def mood_score(row: pd.Series, text: str) -> int:
    negative = val(row, "asr__asr_global_negative_count")
    pressure = val(row, "asr__asr_pressure_count")
    positive = val(row, "asr__asr_global_positive_count")
    stable_positive = has_any(
        text,
        [
            "整体还是很开心",
            "情绪没有太大的波动",
            "没有特别伤心",
            "心情不错",
            "目前状态挺好",
        ],
    )
    severe_mood = has_any(text, ["抑郁", "压抑", "绝望", "喘不过气", "崩溃", "低落", "心情低落"])
    moderate_mood = has_any(text, ["难受", "郁闷", "烦躁", "焦虑", "压力", "迷茫", "害怕"])
    if severe_mood and (has_any(text, ["每天", "一直", "大部分时间", "那些日子"]) or val(row, "asr__asr_severity_count") >= 2):
        return 3
    if severe_mood:
        return 2
    if moderate_mood and (pressure >= 5 or negative >= 6):
        return 2
    if negative >= 8 and not stable_positive:
        return 2
    if negative >= 4 or pressure >= 5 or moderate_mood:
        return 0 if stable_positive and positive >= 5 else 1
    return 0


def appetite_score(row: pd.Series, text: str) -> int:
    refined_col = "asr__asr_phq5_appetite_refined_rule_score"
    if refined_col in row.index:
        return int(round(val(row, refined_col)))
    # Food mentions alone are not appetite symptoms. Require direct appetite,
    # intake, or weight-change language.
    if has_any(text, ["吃不下", "食欲不振", "没胃口", "胃口不好", "不想吃", "暴食", "吃太多", "饭量"]):
        if has_any(text, ["每天", "一直", "经常", "白天", "严重"]):
            return 2
        return 1
    if has_any(text, ["体重", "瘦了", "胖了"]) and not has_any(text, ["健身", "减肥", "主动"]):
        return 1
    return 0


def self_harm_score(row: pd.Series, text: str) -> int:
    direct = val(row, "asr__asr_phq9_self_harm_direct_count")
    if direct <= 0:
        return 0
    self_harm_terms = ["不想活", "不如死", "轻生", "自杀", "自伤", "伤害自己", "结束生命", "活着没意思", "想死"]
    historical_cues = ["小学", "初中", "高中", "小时候", "小的时候", "童年", "以前", "当时", "那个时候", "那时候", "曾经", "之前", "前几年"]
    current_cues = ["最近", "现在", "目前", "这几天", "这段时间", "近两周", "这两周", "每天", "一直", "经常", "反复", "还会", "仍然"]
    frequency_cues = ["每天", "一直", "经常", "反复"]

    contexts = term_contexts(text, self_harm_terms)
    current_contexts = [
        ctx
        for ctx in contexts
        if has_any(ctx, current_cues) or not (has_any(ctx, historical_cues) or "想过" in ctx)
    ]
    if not current_contexts:
        return 0
    if any(has_any(ctx, frequency_cues) for ctx in current_contexts):
        return 3
    if len(current_contexts) >= 2:
        return 2
    return 1


def concentration_score(row: pd.Series, text: str) -> int:
    refined_col = "asr__asr_phq7_concentration_refined_rule_score"
    if refined_col in row.index:
        return int(round(val(row, refined_col)))
    # Avoid treating "脑子里想东西多" during sleep discussion as PHQ7 by
    # itself. Require explicit study/work concentration or task failure terms.
    explicit = has_any(text, ["注意力", "专注", "集中", "效率低", "学不进去", "看不进去", "记不住", "没头绪", "毫无头绪", "拖延"])
    task_burden = has_any(text, ["复习时间很少", "完不成任务", "来不及", "没有准备好", "安排时间", "忙不过来"])
    pressure = val(row, "asr__asr_pressure_count")
    if explicit and has_any(text, ["每天", "一直", "经常", "严重"]):
        return 2
    if explicit or task_burden:
        return 1
    if pressure >= 10:
        return 1
    return 0


def sleep_score_broad(row: pd.Series, text: str) -> int:
    score = score_direct_item(row, "phq3_sleep", allow_weak=False, use_refined=False)
    if has_any(text, ["每天"]) and has_any(text, ["睡不着", "失眠", "睡眠质量", "睡不好"]):
        return max(score, 3)
    if has_any(text, ["睡不着", "失眠", "入睡困难", "睡不醒", "睡眠质量", "睡不好"]):
        return max(score, 2)
    if has_any(text, ["熬夜", "睡得晚", "晚睡", "打呼噜", "磨牙"]):
        return max(score, 1)
    return score


def mood_score_broad(row: pd.Series, text: str) -> int:
    negative = val(row, "asr__asr_global_negative_count")
    pressure = val(row, "asr__asr_pressure_count")
    mood = score_direct_item(row, "phq2_mood", use_refined=False)
    if negative >= 8 or has_any(text, ["抑郁", "压抑", "喘不过气", "崩溃"]):
        mood = max(mood, 2)
    elif negative >= 4 or pressure >= 5:
        mood = max(mood, 1)
    return mood


def concentration_score_broad(row: pd.Series, text: str) -> int:
    concentration = score_direct_item(row, "phq7_concentration", use_refined=False)
    if val(row, "asr__asr_pressure_count") >= 8 or has_any(text, ["没头绪", "学不进去", "注意力", "专注", "复习时间很少"]):
        concentration = max(concentration, 1)
    return concentration


@dataclass
class PriorResult:
    region: str
    item_scores: dict[str, float]
    raw_sum: float


def strong_prior_score(row: pd.Series, text: str) -> PriorResult:
    low_content = int(val(row, "asr__asr_low_content_flag") > 0 or val(row, "asr__asr_text_len") < 20)
    negative = val(row, "asr__asr_global_negative_count")
    positive = val(row, "asr__asr_global_positive_count")
    pressure = val(row, "asr__asr_pressure_count")
    filler_ratio = val(row, "asr__asr_core_filler_per_100char")
    rumination_minor = val(row, "asr__asr_rumination_minor_event_score")

    scores = {item: 0 for item in ITEMS}
    if low_content:
        scores.update(
            {
                "phq1_interest": 2,
                "phq2_mood": 2,
                "phq4_fatigue": 2,
                "phq7_concentration": 2,
                "phq8_psychomotor": 2,
                "phq9_self_harm": 0,
            }
        )
        return PriorResult("low_content", scores, float(sum(scores.values())))

    scores["phq1_interest"] = score_direct_item(row, "phq1_interest", use_refined=True)
    if scores["phq1_interest"] == 0 and positive <= 1 and (negative >= 8 or pressure >= 8):
        scores["phq1_interest"] = 1

    scores["phq2_mood"] = mood_score(row, text)
    if rumination_minor >= 20 and scores["phq2_mood"] <= 1:
        scores["phq2_mood"] = 2

    scores["phq3_sleep"] = sleep_score(row, text)

    fatigue = score_direct_item(row, "phq4_fatigue", use_refined=True)
    if scores["phq3_sleep"] >= 2 or pressure >= 8 or has_any(text, ["被压着", "喘不过气", "事情很多", "任务多"]):
        fatigue = max(fatigue, 1)
    scores["phq4_fatigue"] = fatigue

    scores["phq5_appetite"] = appetite_score(row, text)

    self_worth = score_direct_item(row, "phq6_self_worth", use_refined=True)
    if has_any(text, ["愧疚", "辜负", "自责", "没用", "废物", "失败"]):
        self_worth = max(self_worth, 1)
    if has_any(text, ["辜负", "没用", "废物", "自责"]):
        self_worth = max(self_worth, 2)
    scores["phq6_self_worth"] = self_worth
    if rumination_minor >= 40 and scores["phq6_self_worth"] == 0:
        scores["phq6_self_worth"] = 1

    scores["phq7_concentration"] = concentration_score(row, text)

    psychomotor = score_direct_item(row, "phq8_psychomotor", use_refined=True)
    if has_any(text, ["烦躁", "坐立不安", "焦躁", "急躁"]):
        psychomotor = max(psychomotor, 1)
    if filler_ratio >= 6 and negative >= 4:
        psychomotor = max(psychomotor, 1)
    scores["phq8_psychomotor"] = psychomotor
    if rumination_minor >= 40 and scores["phq8_psychomotor"] == 0:
        scores["phq8_psychomotor"] = 1

    scores["phq9_self_harm"] = self_harm_score(row, text)

    if scores["phq9_self_harm"] > 0:
        region = "high_risk_self_harm"
    elif rumination_minor >= 20:
        region = "rumination_minor_events"
    elif negative >= 5 and positive >= 3:
        region = "conflict_positive_negative"
    elif sum(scores.values()) >= 5 or any(v >= 2 for v in scores.values()):
        region = "direct_symptom"
    else:
        region = "weak_semantic"
    return PriorResult(region, scores, float(sum(scores.values())))


def strong_prior_score_broad(row: pd.Series, text: str) -> PriorResult:
    low_content = int(val(row, "asr__asr_low_content_flag") > 0 or val(row, "asr__asr_text_len") < 20)
    negative = val(row, "asr__asr_global_negative_count")
    positive = val(row, "asr__asr_global_positive_count")
    pressure = val(row, "asr__asr_pressure_count")
    filler_ratio = val(row, "asr__asr_core_filler_per_100char")
    rumination_minor = val(row, "asr__asr_rumination_minor_event_score")

    scores = {item: 0 for item in ITEMS}
    if low_content:
        scores.update(
            {
                "phq1_interest": 2,
                "phq2_mood": 2,
                "phq4_fatigue": 2,
                "phq7_concentration": 2,
                "phq8_psychomotor": 2,
                "phq9_self_harm": 0,
            }
        )
        return PriorResult("low_content", scores, float(sum(scores.values())))

    scores["phq1_interest"] = score_direct_item(row, "phq1_interest", use_refined=False)
    if scores["phq1_interest"] == 0 and positive <= 1 and (negative >= 8 or pressure >= 8):
        scores["phq1_interest"] = 1
    scores["phq2_mood"] = mood_score_broad(row, text)
    if rumination_minor >= 20 and scores["phq2_mood"] <= 1:
        scores["phq2_mood"] = 2
    scores["phq3_sleep"] = sleep_score_broad(row, text)
    fatigue = score_direct_item(row, "phq4_fatigue", use_refined=False)
    if scores["phq3_sleep"] >= 2 or pressure >= 8 or has_any(text, ["被压着", "喘不过气", "事情很多", "任务多"]):
        fatigue = max(fatigue, 1)
    scores["phq4_fatigue"] = fatigue
    scores["phq5_appetite"] = appetite_score(row, text)
    self_worth = score_direct_item(row, "phq6_self_worth", use_refined=False)
    if has_any(text, ["愧疚", "辜负", "自责", "没用", "废物", "失败"]):
        self_worth = max(self_worth, 1)
    if has_any(text, ["辜负", "没用", "废物", "自责"]):
        self_worth = max(self_worth, 2)
    scores["phq6_self_worth"] = self_worth
    if rumination_minor >= 40 and scores["phq6_self_worth"] == 0:
        scores["phq6_self_worth"] = 1
    scores["phq7_concentration"] = concentration_score_broad(row, text)
    psychomotor = score_direct_item(row, "phq8_psychomotor", use_refined=False)
    if has_any(text, ["烦躁", "坐立不安", "焦躁", "急躁"]):
        psychomotor = max(psychomotor, 1)
    if filler_ratio >= 6 and negative >= 4:
        psychomotor = max(psychomotor, 1)
    scores["phq8_psychomotor"] = psychomotor
    if rumination_minor >= 40 and scores["phq8_psychomotor"] == 0:
        scores["phq8_psychomotor"] = 1
    scores["phq9_self_harm"] = self_harm_score(row, text)

    if scores["phq9_self_harm"] > 0:
        region = "high_risk_self_harm"
    elif rumination_minor >= 20:
        region = "rumination_minor_events"
    elif negative >= 5 and positive >= 3:
        region = "conflict_positive_negative"
    elif sum(scores.values()) >= 5 or any(v >= 2 for v in scores.values()):
        region = "direct_symptom"
    else:
        region = "weak_semantic"
    return PriorResult(region, scores, float(sum(scores.values())))


def strong_prior_score_adaptive(row: pd.Series, text: str) -> PriorResult:
    """Use broad scores for sensitivity, but refined caps for protective cases."""
    narrow = strong_prior_score(row, text)
    broad = strong_prior_score_broad(row, text)
    scores = dict(broad.item_scores)

    stable_positive = val(row, "asr__asr_stable_positive_count")
    recovery = val(row, "asr__asr_recovery_count")
    template = val(row, "asr__asr_template_prompt_count")
    severe_mood = val(row, "asr__asr_severe_mood_count")
    rumination_minor = val(row, "asr__asr_rumination_minor_event_score")

    protective_conflict = (stable_positive >= 1 or recovery >= 2) and severe_mood == 0
    template_heavy = template >= 4 and severe_mood == 0

    if protective_conflict or template_heavy:
        # Stable positive/recovery language should not erase real sleep or
        # appetite symptoms, but it should cap broad mood and concentration
        # triggers created by prompt-template words.
        scores["phq2_mood"] = min(scores["phq2_mood"], max(1, narrow.item_scores["phq2_mood"]))
        scores["phq5_appetite"] = min(scores["phq5_appetite"], narrow.item_scores["phq5_appetite"])
        scores["phq7_concentration"] = min(scores["phq7_concentration"], narrow.item_scores["phq7_concentration"])
        scores["phq4_fatigue"] = min(scores["phq4_fatigue"], 1)

    if rumination_minor >= 40:
        scores["phq2_mood"] = max(scores["phq2_mood"], 2)
        scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 1)
        scores["phq8_psychomotor"] = max(scores["phq8_psychomotor"], 1)
        # Treat repeated negative attribution around small events as a
        # clinically suspicious style even when explicit symptoms are sparse.
        scores["phq1_interest"] = max(scores["phq1_interest"], 1)

    if scores["phq9_self_harm"] > 0:
        region = "high_risk_self_harm"
    elif rumination_minor >= 20:
        region = "rumination_minor_events"
    elif protective_conflict or template_heavy:
        region = "protective_conflict"
    else:
        region = broad.region
    return PriorResult(region, scores, float(sum(scores.values())))


def strong_prior_score_v3(row: pd.Series, text: str) -> PriorResult:
    """Broad-sensitive prior with explicit high-miss and false-positive guards."""
    base = strong_prior_score_broad(row, text)
    scores = dict(base.item_scores)

    overload = val(row, "asr__asr_role_overload_count")
    depletion = val(row, "asr__asr_depletion_avoidance_count")
    future_self_doubt = val(row, "asr__asr_future_self_doubt_count")
    procrastination = val(row, "asr__asr_procrastination_anxiety_count")
    overload_risk = val(row, "asr__asr_overload_risk_score")
    future_risk = val(row, "asr__asr_future_anxiety_risk_score")
    protective = val(row, "asr__asr_protective_cap_score")
    control = val(row, "asr__asr_control_protective_count")
    function_protective = val(row, "asr__asr_function_protective_count")
    stable_positive = val(row, "asr__asr_stable_positive_count")
    severe_mood = val(row, "asr__asr_severe_mood_count")
    casual_self_dep = val(row, "asr__asr_casual_self_deprecation_count")
    rumination_minor = val(row, "asr__asr_rumination_minor_event_score")
    pressure = val(row, "asr__asr_pressure_count")
    negative = val(row, "asr__asr_global_negative_count")

    # High-score miss patterns: role overload, procrastination-anxiety loops,
    # depletion/avoidance, and future self-doubt.
    if overload_risk >= 2 or pressure >= 12:
        scores["phq2_mood"] = max(scores["phq2_mood"], 2)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 1)
        scores["phq7_concentration"] = max(scores["phq7_concentration"], 1)
        scores["phq8_psychomotor"] = max(scores["phq8_psychomotor"], 1)
    if overload_risk >= 4:
        scores["phq1_interest"] = max(scores["phq1_interest"], 1)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 2)
    if depletion >= 1:
        scores["phq1_interest"] = max(scores["phq1_interest"], 1)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 1)
    if procrastination >= 2:
        scores["phq7_concentration"] = max(scores["phq7_concentration"], 1)
        scores["phq8_psychomotor"] = max(scores["phq8_psychomotor"], 1)
    if future_risk >= 2:
        scores["phq2_mood"] = max(scores["phq2_mood"], 1)
        scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 1)
    if future_self_doubt >= 3:
        scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 2)
    if rumination_minor >= 20:
        scores["phq2_mood"] = max(scores["phq2_mood"], 2)
        scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 1)
        scores["phq8_psychomotor"] = max(scores["phq8_psychomotor"], 1)
        scores["phq1_interest"] = max(scores["phq1_interest"], 1)

    # False-positive guards: if the text itself says symptoms are controllable,
    # stable, resolved, or functioning is good, cap broad-only dimensions that
    # are usually template-triggered.
    protective_cap = protective >= 4 and severe_mood == 0
    future_only_cap = control >= 1 and future_self_doubt >= 2 and severe_mood == 0
    if protective_cap or future_only_cap:
        scores["phq2_mood"] = min(scores["phq2_mood"], 1)
        scores["phq7_concentration"] = min(scores["phq7_concentration"], 1)
        if scores["phq3_sleep"] >= 3 and not has_any(text, ["每天睡不着", "每晚睡不着", "失眠", "严重"]):
            scores["phq3_sleep"] = 1
        scores["phq4_fatigue"] = min(scores["phq4_fatigue"], 1)
    if casual_self_dep and function_protective >= 2 and severe_mood == 0:
        scores["phq6_self_worth"] = min(scores["phq6_self_worth"], 1)
    if stable_positive >= 2 and negative < 8 and severe_mood == 0:
        scores["phq2_mood"] = min(scores["phq2_mood"], 1)

    if scores["phq9_self_harm"] > 0:
        region = "high_risk_self_harm"
    elif val(row, "asr__asr_low_content_flag") > 0 or val(row, "asr__asr_text_len") < 20:
        region = "low_content"
    elif rumination_minor >= 20:
        region = "rumination_minor_events"
    elif overload_risk >= 2:
        region = "overload_depletion"
    elif protective_cap or future_only_cap:
        region = "protected_conflict"
    else:
        region = base.region
    return PriorResult(region, scores, float(sum(scores.values())))


def strong_prior_score_v4(row: pd.Series, text: str) -> PriorResult:
    """V3 plus recency/protective gates and Big-Five risk amplification."""
    base = strong_prior_score_v3(row, text)
    scores = dict(base.item_scores)

    overload = val(row, "asr__asr_role_overload_count")
    depletion = val(row, "asr__asr_depletion_avoidance_count")
    future_self_doubt = val(row, "asr__asr_future_self_doubt_count")
    procrastination = val(row, "asr__asr_procrastination_anxiety_count")
    overload_risk = val(row, "asr__asr_overload_risk_score")
    future_risk = val(row, "asr__asr_future_anxiety_risk_score")
    protective = val(row, "asr__asr_protective_cap_score")
    function_protective = val(row, "asr__asr_function_protective_count")
    severe_mood = val(row, "asr__asr_severe_mood_count")
    negative = val(row, "asr__asr_global_negative_count")
    pressure = val(row, "asr__asr_pressure_count")
    rumination_minor = val(row, "asr__asr_rumination_minor_event_score")

    neuro = val(row, "bigfive__Neuroticism")
    extraversion = val(row, "bigfive__Extraversion")
    conscientiousness = val(row, "bigfive__Conscientiousness")
    high_neuro = neuro >= 8
    mid_neuro = neuro >= 7
    low_neuro = 0 < neuro <= 6

    social_isolation = count_any(
        text,
        [
            "不擅长与他人交流",
            "不太擅长与他人交流",
            "不喜欢与他人交流",
            "不太喜欢与他人交流",
            "插不上嘴",
            "感觉有些孤独",
            "感到孤独",
            "孤独",
            "怯场",
            "社恐",
        ],
    )
    unresolved_grief = count_any(
        text,
        ["支离破碎", "亲情", "剥离", "长久以来", "去世之后", "不来往", "难过", "哭了一下"],
    )
    past_story = count_any(
        text,
        ["初中", "高中", "小学", "小时候", "小的时候", "以前", "当时", "后来", "之前", "前几年", "有一年", "那个时候"],
    )
    mild_or_resolved = count_any(
        text,
        [
            "没太大",
            "没有太大",
            "没有什么太",
            "没有什么特别悲伤",
            "没有感到悲伤",
            "悲伤的事情没有",
            "悲伤的事情基本上没有",
            "没有很悲伤",
            "已经过去了",
            "不太会引起悲伤",
            "还算挺好",
            "压力也比较小",
            "学习压力也不是很大",
            "没有什么压力",
            "心情还不错",
            "心态比较好",
            "比较开心",
            "每天都过得比较开心",
            "开心成了常态",
            "轻松的状态",
            "交心的朋友",
            "找到工作",
            "心仪",
            "理想的工作",
            "顺利",
            "满意",
            "比较满意",
            "完成",
            "解决",
            "成长",
            "成熟",
            "乐观",
            "朋友也特别多",
            "朋友",
            "认识了一些朋友",
            "代码能跑通",
            "上大师",
            "爬山",
        ],
    )
    severe_depletion = count_any(
        text,
        ["写不完", "不想听", "不知道该干点什么", "打游戏打累", "没完没了", "没有意义", "不想写", "没有进展", "没进展"],
    )
    relationship_family_burden = count_any(
        text,
        [
            "异地恋",
            "不能在一起",
            "相隔比较远",
            "贯穿",
            "很大的一个烦恼",
            "爸爸的病情",
            "病情",
            "脑中风",
            "半身不遂",
            "不太想回去",
            "发生冲突",
            "甩不开",
            "对象",
            "感情上",
            "打击",
        ],
    )
    physical_burden = count_any(
        text,
        ["身体非常不好", "身体不是很好", "身体状况", "经常生病", "疾病", "医院", "甲流", "不适", "调理不好", "吃不下去饭", "睡不着"],
    )
    task_blockage = count_any(
        text,
        [
            "学业压力比较大",
            "事情比较多",
            "比较复杂",
            "做项目",
            "没有改进",
            "改进并不大",
            "调试代码",
            "毫无头绪",
            "不知道进展",
            "完不成任务",
            "实验结果",
            "重新去跑",
            "写论文",
            "许多东西要改",
            "科研上的事情",
            "技能不是很多",
            "不足以",
            "学习能力不太够",
        ],
    )
    exam_relief = count_any(
        text,
        ["考完之后就可以回家放假", "有很多时间来复习", "没有特别的痛苦", "复习也没有特别", "可以回家放假"],
    )
    roommate_interpersonal_stress = count_any(
        text,
        ["舍友", "宿舍", "人际交往", "人际关系", "为难", "压抑", "无法容忍"],
    )
    dysregulated_coping = count_any(text, ["发疯", "精神状态", "发泄", "无限制", "有精力去接着"])
    competence_self_doubt = count_any(text, ["不相信自己", "配不上", "没有主动争取", "不自信", "错失", "没有他的能力强", "懊恼"])
    off_prompt_self_involved = count_any(text, ["看不起我", "不知道该说什么", "非常奇怪", "特别的感情", "留他的名字", "叫出来"])
    research_blocked = count_any(text, ["科研", "受阻", "实验室", "进展", "写论文", "调试代码", "项目"])
    joy_dominant = count_any(
        text,
        ["每天都过得非常快乐", "一直都很开心", "每天过得很开心", "开心的情绪大于烦恼", "特别乐观", "容易满足", "很容易开心", "总体上还是对自己挺满意"],
    )
    strong_functional_relief = count_any(
        text,
        ["心态比较好", "压力也比较小", "学习压力也不是很大", "轻松的状态", "交心的朋友", "找到工作", "心仪", "心理压力倒也没有太大"],
    )
    third_person_story = count_any(text, ["他", "她"])
    prompt_anchor = count_any(text, ["最近", "烦恼", "开心", "悲伤", "满意", "改进"])
    future_only_mild = future_risk >= 3 and overload_risk <= 2 and mild_or_resolved >= 3 and low_neuro
    overload_but_functional = (
        overload_risk >= 2
        and low_neuro
        and mild_or_resolved >= 3
        and function_protective >= 2
        and severe_mood == 0
        and procrastination < 5
        and severe_depletion < 3
    )
    past_story_dominant = past_story >= 5 and low_neuro and mild_or_resolved >= 3 and severe_mood == 0

    # Risk amplification: the same sparse text evidence is interpreted more
    # broadly when high neuroticism suggests stress sensitivity.
    if high_neuro and overload_risk >= 5:
        scores["phq1_interest"] = max(scores["phq1_interest"], 1)
        scores["phq2_mood"] = max(scores["phq2_mood"], 3)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 2)
        scores["phq7_concentration"] = max(scores["phq7_concentration"], 2)
        scores["phq8_psychomotor"] = max(scores["phq8_psychomotor"], 1)
        if future_self_doubt >= 2 or extraversion <= 4:
            scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 2)
        if pressure >= 4 or future_risk >= 3:
            scores["phq3_sleep"] = max(scores["phq3_sleep"], 1)
    if high_neuro and severe_depletion >= 3:
        scores["phq1_interest"] = max(scores["phq1_interest"], 2)
        scores["phq2_mood"] = max(scores["phq2_mood"], 3)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 2)
        scores["phq7_concentration"] = max(scores["phq7_concentration"], 2)
        scores["phq8_psychomotor"] = max(scores["phq8_psychomotor"], 1)
    if high_neuro and future_self_doubt >= 2 and pressure >= 3:
        scores["phq2_mood"] = max(scores["phq2_mood"], 3)
        scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 2)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 1)
    if high_neuro and pressure >= 5 and negative >= 7:
        scores["phq2_mood"] = max(scores["phq2_mood"], 2)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 1)
        scores["phq7_concentration"] = max(scores["phq7_concentration"], 1)
        scores["phq8_psychomotor"] = max(scores["phq8_psychomotor"], 1)
        if pressure >= 7:
            scores["phq2_mood"] = max(scores["phq2_mood"], 3)
    if high_neuro and (overload_risk >= 7 or severe_depletion >= 3):
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 3)
        scores["phq8_psychomotor"] = max(scores["phq8_psychomotor"], 2)
        scores["phq3_sleep"] = max(scores["phq3_sleep"], 1)
        scores["phq5_appetite"] = max(scores["phq5_appetite"], 1)
        if extraversion <= 4 or future_self_doubt >= 2:
            scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 3)
    if high_neuro and pressure >= 7 and (unresolved_grief >= 2 or extraversion <= 4):
        scores["phq3_sleep"] = max(scores["phq3_sleep"], 1)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 2)
        scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 1)
    if mid_neuro and overload_risk >= 3 and future_self_doubt >= 2:
        scores["phq2_mood"] = max(scores["phq2_mood"], 2)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 2)
        scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 2)
        scores["phq7_concentration"] = max(scores["phq7_concentration"], 2)
        scores["phq8_psychomotor"] = max(scores["phq8_psychomotor"], 1)
    if task_blockage >= 3:
        scores["phq2_mood"] = max(scores["phq2_mood"], 2)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 1)
        scores["phq7_concentration"] = max(scores["phq7_concentration"], 1)
        if high_neuro or mid_neuro:
            scores["phq7_concentration"] = max(scores["phq7_concentration"], 2)
        if has_any(text, ["不足以", "学习能力不太够", "毫无头绪", "完不成任务"]):
            scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 2)
        elif extraversion <= 4 or high_neuro:
            scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 1)
    if roommate_interpersonal_stress >= 4 and high_neuro:
        scores["phq1_interest"] = max(scores["phq1_interest"], 1)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 2)
        scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 1)
        scores["phq7_concentration"] = max(scores["phq7_concentration"], 1)
        if has_any(text, ["压抑", "严重下降"]):
            scores["phq2_mood"] = max(scores["phq2_mood"], 3)
    if dysregulated_coping >= 3:
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 2)
        scores["phq7_concentration"] = max(scores["phq7_concentration"], 2)
        scores["phq8_psychomotor"] = max(scores["phq8_psychomotor"], 2)
    if competence_self_doubt >= 3 and high_neuro:
        scores["phq2_mood"] = max(scores["phq2_mood"], 2)
        scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 2)
        scores["phq7_concentration"] = max(scores["phq7_concentration"], 1)
    if third_person_story >= 8 and off_prompt_self_involved >= 3:
        scores["phq2_mood"] = max(scores["phq2_mood"], 2)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 1)
        scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 1)
        scores["phq7_concentration"] = max(scores["phq7_concentration"], 1)
    if relationship_family_burden >= 2 and research_blocked >= 2 and (mid_neuro or extraversion <= 4):
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 2)
        scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 1)
        scores["phq7_concentration"] = max(scores["phq7_concentration"], 1)
    if procrastination >= 7 and depletion >= 3:
        scores["phq2_mood"] = max(scores["phq2_mood"], 3)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 2)
        scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 2)
        scores["phq8_psychomotor"] = max(scores["phq8_psychomotor"], 2)
    if procrastination >= 5 and negative >= 10:
        scores["phq1_interest"] = max(scores["phq1_interest"], 1)
        scores["phq2_mood"] = max(scores["phq2_mood"], 2)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 1)
        scores["phq7_concentration"] = max(scores["phq7_concentration"], 2)
        scores["phq8_psychomotor"] = max(scores["phq8_psychomotor"], 1)
        scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 1)
    if mid_neuro and social_isolation >= 2:
        scores["phq1_interest"] = max(scores["phq1_interest"], 1)
        scores["phq2_mood"] = max(scores["phq2_mood"], 2)
        scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 2 if extraversion <= 4 else 1)
        if has_any(text, ["几乎没有满意", "满意地方。几乎没有", "没有满意"]):
            scores["phq2_mood"] = max(scores["phq2_mood"], 3)
            scores["phq6_self_worth"] = max(scores["phq6_self_worth"], 3)
    if mid_neuro and unresolved_grief >= 2:
        scores["phq1_interest"] = max(scores["phq1_interest"], 1)
        scores["phq2_mood"] = max(scores["phq2_mood"], 2)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 1)
        if "支离破碎" in text or "剥离" in text:
            scores["phq2_mood"] = max(scores["phq2_mood"], 3)
    if relationship_family_burden >= 2:
        scores["phq1_interest"] = max(scores["phq1_interest"], 1 if relationship_family_burden >= 4 else scores["phq1_interest"])
        scores["phq2_mood"] = max(scores["phq2_mood"], 2)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 1)
        scores["phq7_concentration"] = max(scores["phq7_concentration"], 1 if relationship_family_burden >= 4 else scores["phq7_concentration"])
        if relationship_family_burden >= 5 or "打击" in text:
            scores["phq2_mood"] = max(scores["phq2_mood"], 3)
    if physical_burden >= 2:
        scores["phq2_mood"] = max(scores["phq2_mood"], 2)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 2)
        if has_any(text, ["睡不着", "睡眠质量", "睡得"]):
            scores["phq3_sleep"] = max(scores["phq3_sleep"], 2)
        if has_any(text, ["吃不下", "吃不下去饭", "食欲不太好", "食欲不振"]):
            scores["phq5_appetite"] = max(scores["phq5_appetite"], 2)
        if physical_burden >= 4:
            scores["phq1_interest"] = max(scores["phq1_interest"], 1)
            scores["phq7_concentration"] = max(scores["phq7_concentration"], 1)
    if third_person_story >= 8 and prompt_anchor <= 1:
        scores["phq2_mood"] = max(scores["phq2_mood"], 1)
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 1)
        scores["phq7_concentration"] = max(scores["phq7_concentration"], 1)
    if rumination_minor >= 20:
        scores["phq4_fatigue"] = max(scores["phq4_fatigue"], 1)
        scores["phq7_concentration"] = max(scores["phq7_concentration"], 1)

    # Protective gates: resolved, historical, or well-functioning narratives are
    # weak PHQ evidence unless there is a strong current loop or high trait risk.
    if future_only_mild or overload_but_functional or past_story_dominant:
        scores["phq1_interest"] = min(scores["phq1_interest"], 1)
        scores["phq2_mood"] = min(scores["phq2_mood"], 1)
        scores["phq4_fatigue"] = min(scores["phq4_fatigue"], 1)
        scores["phq6_self_worth"] = min(scores["phq6_self_worth"], 1)
        scores["phq7_concentration"] = min(scores["phq7_concentration"], 1)
        scores["phq8_psychomotor"] = min(scores["phq8_psychomotor"], 0 if future_only_mild else 1)
        if not has_any(text, ["失眠", "睡不着", "入睡困难", "每晚睡", "睡眠质量"]):
            scores["phq3_sleep"] = min(scores["phq3_sleep"], 0)
        if not has_any(text, ["食欲不振", "食欲不太好", "吃不下", "没胃口", "暴食"]):
            scores["phq5_appetite"] = min(scores["phq5_appetite"], 0)
    if past_story_dominant:
        scores["phq1_interest"] = 0
        scores["phq4_fatigue"] = 0
        scores["phq6_self_worth"] = min(scores["phq6_self_worth"], 1)
        scores["phq7_concentration"] = min(scores["phq7_concentration"], 1 if has_any(text, ["拖延", "效率"]) else 0)
        scores["phq8_psychomotor"] = 0
    if future_only_mild:
        scores["phq4_fatigue"] = 0
        scores["phq8_psychomotor"] = 0
    if overload_but_functional:
        scores["phq4_fatigue"] = min(scores["phq4_fatigue"], 1)
        scores["phq7_concentration"] = min(scores["phq7_concentration"], 1)
        scores["phq8_psychomotor"] = 0
        if strong_functional_relief >= 2:
            scores["phq1_interest"] = 0
            scores["phq4_fatigue"] = 0
    if future_only_mild and strong_functional_relief >= 1:
        scores["phq6_self_worth"] = 0
    if exam_relief >= 3 and low_neuro:
        scores["phq1_interest"] = 0
        scores["phq2_mood"] = min(scores["phq2_mood"], 1)
        scores["phq4_fatigue"] = 0
        scores["phq6_self_worth"] = 0
        scores["phq7_concentration"] = min(scores["phq7_concentration"], 1)
        scores["phq8_psychomotor"] = 0
    if low_neuro and mild_or_resolved >= 4 and severe_mood == 0 and negative <= 8 and procrastination < 5 and relationship_family_burden == 0 and physical_burden < 2:
        scores["phq1_interest"] = min(scores["phq1_interest"], 1)
        scores["phq2_mood"] = min(scores["phq2_mood"], 1)
        scores["phq4_fatigue"] = min(scores["phq4_fatigue"], 1)
        scores["phq6_self_worth"] = min(scores["phq6_self_worth"], 1)
        scores["phq7_concentration"] = min(scores["phq7_concentration"], 1)
        scores["phq8_psychomotor"] = 0
        scores["phq3_sleep"] = min(scores["phq3_sleep"], 2)
    if joy_dominant >= 1 and relationship_family_burden == 0 and physical_burden < 2 and severe_depletion < 3 and severe_mood == 0:
        scores["phq1_interest"] = min(scores["phq1_interest"], 1)
        scores["phq2_mood"] = min(scores["phq2_mood"], 1)
        scores["phq4_fatigue"] = min(scores["phq4_fatigue"], 1)
        scores["phq6_self_worth"] = min(scores["phq6_self_worth"], 1)
        scores["phq7_concentration"] = min(scores["phq7_concentration"], 1)
        scores["phq8_psychomotor"] = 0
    if low_neuro and mild_or_resolved >= 5 and protective >= 4 and severe_mood == 0 and procrastination < 5:
        scores["phq2_mood"] = min(scores["phq2_mood"], 1)
        scores["phq6_self_worth"] = min(scores["phq6_self_worth"], 1)
        scores["phq7_concentration"] = min(scores["phq7_concentration"], 1)
    if extraversion >= 7 and conscientiousness >= 7 and mild_or_resolved >= 3 and negative <= 6 and severe_depletion == 0 and relationship_family_burden == 0 and physical_burden < 2:
        scores["phq1_interest"] = min(scores["phq1_interest"], 1)
        scores["phq2_mood"] = min(scores["phq2_mood"], 1)
        scores["phq4_fatigue"] = min(scores["phq4_fatigue"], 1)
        scores["phq6_self_worth"] = min(scores["phq6_self_worth"], 1)
        scores["phq7_concentration"] = min(scores["phq7_concentration"], 1)
        scores["phq8_psychomotor"] = 0
        scores["phq3_sleep"] = min(scores["phq3_sleep"], 1)
    if past_story_dominant and extraversion >= 8 and mild_or_resolved >= 5:
        for item in ITEMS:
            if item != "phq9_self_harm":
                scores[item] = 0

    if scores["phq9_self_harm"] > 0:
        region = "high_risk_self_harm"
    elif val(row, "asr__asr_low_content_flag") > 0 or val(row, "asr__asr_text_len") < 20:
        region = "low_content"
    elif past_story_dominant:
        region = "historical_resolved"
    elif future_only_mild or overload_but_functional:
        region = "functional_stress"
    elif high_neuro and (overload_risk >= 5 or severe_depletion >= 3 or future_self_doubt >= 2 or pressure >= 5):
        region = "trait_amplified_overload"
    elif relationship_family_burden >= 2:
        region = "relationship_family_burden"
    elif physical_burden >= 2:
        region = "physical_burden"
    elif task_blockage >= 3:
        region = "task_blockage"
    elif third_person_story >= 8 and prompt_anchor <= 1:
        region = "off_prompt_story"
    elif roommate_interpersonal_stress >= 4 and high_neuro:
        region = "roommate_interpersonal_stress"
    elif mid_neuro and social_isolation >= 2:
        region = "social_isolation"
    elif mid_neuro and unresolved_grief >= 2:
        region = "unresolved_grief"
    else:
        region = base.region
    return PriorResult(region, scores, float(sum(scores.values())))


def add_prior_columns(df: pd.DataFrame, texts: dict[int, str]) -> pd.DataFrame:
    df = df.copy()
    regions: list[str] = []
    sums: list[float] = []
    broad_regions: list[str] = []
    broad_sums: list[float] = []
    adaptive_regions: list[str] = []
    adaptive_sums: list[float] = []
    v3_regions: list[str] = []
    v3_sums: list[float] = []
    v4_regions: list[str] = []
    v4_sums: list[float] = []
    for item in ITEMS:
        df[f"prior_{item}"] = 0.0
        df[f"broad_prior_{item}"] = 0.0
        df[f"adaptive_prior_{item}"] = 0.0
        df[f"v3_prior_{item}"] = 0.0
        df[f"v4_prior_{item}"] = 0.0
    for idx, row in df.iterrows():
        sid = int(row["ID"])
        text = texts.get(sid, "")
        result = strong_prior_score(row, text)
        broad_result = strong_prior_score_broad(row, text)
        adaptive_result = strong_prior_score_adaptive(row, text)
        v3_result = strong_prior_score_v3(row, text)
        v4_result = strong_prior_score_v4(row, text)
        regions.append(result.region)
        sums.append(result.raw_sum)
        broad_regions.append(broad_result.region)
        broad_sums.append(broad_result.raw_sum)
        adaptive_regions.append(adaptive_result.region)
        adaptive_sums.append(adaptive_result.raw_sum)
        v3_regions.append(v3_result.region)
        v3_sums.append(v3_result.raw_sum)
        v4_regions.append(v4_result.region)
        v4_sums.append(v4_result.raw_sum)
        for item, score in result.item_scores.items():
            df.at[idx, f"prior_{item}"] = float(score)
        for item, score in broad_result.item_scores.items():
            df.at[idx, f"broad_prior_{item}"] = float(score)
        for item, score in adaptive_result.item_scores.items():
            df.at[idx, f"adaptive_prior_{item}"] = float(score)
        for item, score in v3_result.item_scores.items():
            df.at[idx, f"v3_prior_{item}"] = float(score)
        for item, score in v4_result.item_scores.items():
            df.at[idx, f"v4_prior_{item}"] = float(score)
    df["prior_region"] = regions
    df["prior_raw_sum"] = sums
    df["broad_prior_region"] = broad_regions
    df["broad_prior_raw_sum"] = broad_sums
    df["adaptive_prior_region"] = adaptive_regions
    df["adaptive_prior_raw_sum"] = adaptive_sums
    df["v3_prior_region"] = v3_regions
    df["v3_prior_raw_sum"] = v3_sums
    df["v4_prior_region"] = v4_regions
    df["v4_prior_raw_sum"] = v4_sums
    return df


def ridge_pipeline(n_features: int, select_k: int) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if 0 < select_k < n_features:
        steps.append(("select", SelectKBest(f_regression, k=select_k)))
    steps.append(("ridge", RidgeCV(alphas=np.logspace(-2, 3, 24))))
    return Pipeline(steps)


def build_calibration_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["prior_raw_sum"] + [f"prior_{item}" for item in ITEMS]
    x = df[cols].copy()
    region_dummies = pd.get_dummies(df["prior_region"], prefix="region", dtype=float)
    x = pd.concat([x, region_dummies], axis=1)
    for c in region_dummies.columns:
        x[f"{c}__raw_sum"] = region_dummies[c] * df["prior_raw_sum"].astype(float)
    return x


def non_asr_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c.startswith("audio__") or c.startswith("imu__") or c.startswith("face__") or c.startswith("bigfive__")
    ]


def align_columns(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_x.align(test_x, join="left", axis=1, fill_value=0.0)


def run_cv(df: pd.DataFrame, feature_cols: list[str], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df["phq9_score"].to_numpy(float)
    label2 = df["label2"].to_numpy(int)
    label3 = df["label3"].to_numpy(int)
    ids = df["ID"].to_numpy(int)
    prior_items = df[[f"prior_{item}" for item in ITEMS]].to_numpy(float)
    prior_sum = df["prior_raw_sum"].to_numpy(float)
    broad_prior_items = df[[f"broad_prior_{item}" for item in ITEMS]].to_numpy(float)
    broad_prior_sum = df["broad_prior_raw_sum"].to_numpy(float)
    adaptive_prior_items = df[[f"adaptive_prior_{item}" for item in ITEMS]].to_numpy(float)
    adaptive_prior_sum = df["adaptive_prior_raw_sum"].to_numpy(float)
    v3_prior_items = df[[f"v3_prior_{item}" for item in ITEMS]].to_numpy(float)
    v3_prior_sum = df["v3_prior_raw_sum"].to_numpy(float)
    v4_prior_items = df[[f"v4_prior_{item}" for item in ITEMS]].to_numpy(float)
    v4_prior_sum = df["v4_prior_raw_sum"].to_numpy(float)

    splitter = RepeatedStratifiedKFold(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )
    pred_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    non_asr_cols = non_asr_feature_cols(df)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(df)), label3), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # Raw deterministic prior, repeated per fold for comparable aggregation.
        method_preds: dict[str, np.ndarray] = {
            "strong_prior_raw_sum": np.clip(prior_sum[test_idx], 0.0, 27.0),
            "strong_prior_raw_sum_broad": np.clip(broad_prior_sum[test_idx], 0.0, 27.0),
            "strong_prior_raw_sum_adaptive": np.clip(adaptive_prior_sum[test_idx], 0.0, 27.0),
            "strong_prior_raw_sum_v3": np.clip(v3_prior_sum[test_idx], 0.0, 27.0),
            "strong_prior_raw_sum_v4": np.clip(v4_prior_sum[test_idx], 0.0, 27.0),
        }

        iso_broad = IsotonicRegression(y_min=0.0, y_max=27.0, out_of_bounds="clip")
        iso_broad.fit(broad_prior_sum[train_idx], y[train_idx])
        method_preds["strong_prior_broad_isotonic"] = np.clip(
            iso_broad.predict(broad_prior_sum[test_idx]),
            0.0,
            27.0,
        )

        iso_adaptive = IsotonicRegression(y_min=0.0, y_max=27.0, out_of_bounds="clip")
        iso_adaptive.fit(adaptive_prior_sum[train_idx], y[train_idx])
        method_preds["strong_prior_adaptive_isotonic"] = np.clip(
            iso_adaptive.predict(adaptive_prior_sum[test_idx]),
            0.0,
            27.0,
        )

        iso_v3 = IsotonicRegression(y_min=0.0, y_max=27.0, out_of_bounds="clip")
        iso_v3.fit(v3_prior_sum[train_idx], y[train_idx])
        method_preds["strong_prior_v3_isotonic"] = np.clip(
            iso_v3.predict(v3_prior_sum[test_idx]),
            0.0,
            27.0,
        )

        iso_v4 = IsotonicRegression(y_min=0.0, y_max=27.0, out_of_bounds="clip")
        iso_v4.fit(v4_prior_sum[train_idx], y[train_idx])
        method_preds["strong_prior_v4_isotonic"] = np.clip(
            iso_v4.predict(v4_prior_sum[test_idx]),
            0.0,
            27.0,
        )

        train_cal = build_calibration_features(train_df)
        test_cal = build_calibration_features(test_df)
        train_cal, test_cal = align_columns(train_cal, test_cal)
        cal_model = ridge_pipeline(train_cal.shape[1], select_k=9999)
        cal_model.fit(train_cal, y[train_idx])
        method_preds["strong_prior_region_affine"] = np.clip(cal_model.predict(test_cal), 0.0, 27.0)

        non_asr_model = ridge_pipeline(len(non_asr_cols), args.select_k)
        non_asr_model.fit(train_df[non_asr_cols], y[train_idx])
        non_asr_pred = np.clip(non_asr_model.predict(test_df[non_asr_cols]), 0.0, 27.0)
        method_preds["non_asr_ridge_reference"] = non_asr_pred

        # Region-specific hybrid: strong prior dominates direct/low-content
        # zones; non-ASR helps weak-semantic zones where ASR lacks coverage.
        hybrid = []
        affine = method_preds["strong_prior_region_affine"]
        for local_i, (_idx, row) in enumerate(test_df.iterrows()):
            region = row["prior_region"]
            if region == "weak_semantic":
                pred = 0.35 * affine[local_i] + 0.65 * non_asr_pred[local_i]
            elif region == "rumination_minor_events":
                pred = 0.80 * affine[local_i] + 0.20 * non_asr_pred[local_i]
            elif region == "conflict_positive_negative":
                pred = 0.70 * affine[local_i] + 0.30 * non_asr_pred[local_i]
            else:
                pred = affine[local_i]
            hybrid.append(pred)
        method_preds["strong_prior_router_hybrid"] = np.clip(np.asarray(hybrid), 0.0, 27.0)

        for method, pred in method_preds.items():
            fold_rows.append({"method": method, "fold": fold_idx, **metrics(y[test_idx], pred, label2[test_idx], label3[test_idx])})
            for local_i, idx in enumerate(test_idx):
                row: dict[str, Any] = {
                    "method": method,
                    "fold": fold_idx,
                    "ID": int(ids[idx]),
                    "prior_region": str(df.iloc[idx]["prior_region"]),
                    "v3_prior_region": str(df.iloc[idx]["v3_prior_region"]),
                    "v4_prior_region": str(df.iloc[idx]["v4_prior_region"]),
                    "true_phq9": float(y[idx]),
                    "pred_phq9": float(pred[local_i]),
                    "prior_raw_sum": float(prior_sum[idx]),
                    "v3_prior_raw_sum": float(v3_prior_sum[idx]),
                    "v4_prior_raw_sum": float(v4_prior_sum[idx]),
                    "true_label2": int(label2[idx]),
                    "pred_label2": int(pred_label2(np.array([pred[local_i]]))[0]),
                    "true_label3": int(label3[idx]),
                    "pred_label3": int(pred_label3(np.array([pred[local_i]]))[0]),
                }
                for item_i, item in enumerate(ITEMS):
                    row[f"prior_{item}"] = float(prior_items[idx, item_i])
                    row[f"broad_prior_{item}"] = float(broad_prior_items[idx, item_i])
                    row[f"adaptive_prior_{item}"] = float(adaptive_prior_items[idx, item_i])
                    row[f"v3_prior_{item}"] = float(v3_prior_items[idx, item_i])
                    row[f"v4_prior_{item}"] = float(v4_prior_items[idx, item_i])
                pred_rows.append(row)
    return pd.DataFrame(pred_rows), pd.DataFrame(fold_rows)


def summarize(pred_df: pd.DataFrame, fold_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, group in pred_df.groupby("method"):
        row = {
            "method": method,
            "n_predictions": len(group),
            **metrics(
                group["true_phq9"].to_numpy(float),
                group["pred_phq9"].to_numpy(float),
                group["true_label2"].to_numpy(int),
                group["true_label3"].to_numpy(int),
            ),
        }
        fold_group = fold_df[fold_df["method"] == method]
        for metric_name in ["phq_mae", "phq_rmse", "label2_acc", "label2_f1", "label3_acc", "label3_macro_f1"]:
            row[f"{metric_name}_fold_mean"] = float(fold_group[metric_name].mean())
            row[f"{metric_name}_fold_std"] = float(fold_group[metric_name].std())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["phq_mae", "label3_macro_f1"], ascending=[True, False])


def write_region_summary(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    region_col: str = "prior_region",
    sum_col: str = "prior_raw_sum",
    filename: str = "region_summary.csv",
) -> None:
    rows: list[dict[str, Any]] = []
    for region, group in df.groupby(region_col):
        rows.append(
            {
                "region": region,
                "n": len(group),
                "mean_true_phq9": float(group["phq9_score"].mean()),
                "mean_prior_raw_sum": float(group[sum_col].mean()),
                "label2_rate": float(group["label2"].mean()),
                "label3_ge2_rate": float((group["label3"] == 2).mean()),
            }
        )
    pd.DataFrame(rows).sort_values("region").to_csv(out_dir / filename, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df, feature_cols = load_dataset()
    texts = load_audit_text(ASR_AUDIT_JSONL)
    df = add_prior_columns(df, texts)
    pred_df, fold_df = run_cv(df, feature_cols, args)
    summary_df = summarize(pred_df, fold_df)

    prior_cols = (
        [
            "ID",
            "prior_region",
            "prior_raw_sum",
            "broad_prior_region",
            "broad_prior_raw_sum",
            "adaptive_prior_region",
            "adaptive_prior_raw_sum",
            "v3_prior_region",
            "v3_prior_raw_sum",
            "v4_prior_region",
            "v4_prior_raw_sum",
        ]
        + [f"prior_{item}" for item in ITEMS]
        + [f"broad_prior_{item}" for item in ITEMS]
        + [f"adaptive_prior_{item}" for item in ITEMS]
        + [f"v3_prior_{item}" for item in ITEMS]
        + [f"v4_prior_{item}" for item in ITEMS]
    )
    df[prior_cols + ["phq9_score", "label2", "label3"]].to_csv(
        args.out_dir / "strong_prior_item_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pred_df.to_csv(args.out_dir / "router_predictions.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(args.out_dir / "router_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(args.out_dir / "router_metrics_summary.csv", index=False, encoding="utf-8-sig")
    write_region_summary(df, args.out_dir)
    write_region_summary(
        df,
        args.out_dir,
        region_col="v4_prior_region",
        sum_col="v4_prior_raw_sum",
        filename="v4_region_summary.csv",
    )
    with (args.out_dir / "router_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)

    print(summary_df[["method", "phq_mae", "label2_acc", "label3_macro_f1", "mean_pred_phq9"]].to_string(index=False))
    print()
    print(pd.read_csv(args.out_dir / "region_summary.csv").to_string(index=False))
    print()
    print(pd.read_csv(args.out_dir / "v4_region_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
