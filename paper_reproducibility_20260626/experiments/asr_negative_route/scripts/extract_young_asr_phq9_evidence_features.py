# -*- coding: utf-8 -*-
"""Extract PHQ-9 item-aware evidence features from Young event_1 ASR text.

The output is intentionally transparent and rule based. These features are not
treated as PHQ item labels; they are structured evidence channels that a
supervised model can calibrate against the observed PHQ-9 total score.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASR = ROOT / "obs/asr/young_dashscope_asr_event1_dashscope_clean_with_id17_chunked.jsonl"
DEFAULT_OUT_DIR = ROOT / "obs/experiments/young_asr_phq9_evidence"

ITEMS = {
    "phq1_interest": {
        "direct": [
            "没兴趣",
            "没有兴趣",
            "提不起兴趣",
            "没意思",
            "不想做",
            "不想干",
            "不想参加",
            "什么都不想",
            "开心不起来",
            "不快乐",
            "兴趣下降",
        ],
        "weak": ["没动力", "动力不足", "懒", "逃避", "无聊", "不再像"],
        "protective": [
            "开心",
            "快乐",
            "高兴",
            "喜欢",
            "满意",
            "充实",
            "出去玩",
            "旅游",
            "滑雪",
            "聚餐",
            "朋友",
            "健身",
            "编程",
            "看剧",
        ],
    },
    "phq2_mood": {
        "direct": [
            "心情低落",
            "低落",
            "沮丧",
            "绝望",
            "抑郁",
            "压抑",
            "难过",
            "悲伤",
            "伤心",
            "烦躁",
            "烦恼",
            "焦虑",
            "压力",
            "喘不过气",
            "崩溃",
            "难受",
            "郁闷",
            "不开心",
        ],
        "weak": ["麻烦", "为难", "困扰", "迷茫", "想家", "尴尬", "担心", "害怕", "受不了"],
        "protective": ["开心", "快乐", "高兴", "心情不错", "情绪稳定", "挺好", "满意"],
    },
    "phq3_sleep": {
        "direct": [
            "睡不着",
            "失眠",
            "睡眠质量",
            "早醒",
            "睡不安稳",
            "睡不好",
            "熬夜",
            "睡得晚",
            "睡太多",
            "睡不醒",
            "入睡困难",
            "睡眠下降",
            "睡眠不足",
            "多梦",
        ],
        "weak": ["打呼噜", "磨牙", "晚睡", "作息", "晚上", "宿舍"],
        "protective": ["睡得好", "睡眠不错", "睡眠挺好"],
    },
    "phq4_fatigue": {
        "direct": [
            "疲惫",
            "疲劳",
            "没精力",
            "无精打采",
            "没活力",
            "乏力",
            "力不从心",
            "精力不足",
            "喘不过气",
        ],
        "weak": ["累", "困", "压力大", "事情很多", "忙", "课程多", "任务多", "熬夜", "睡眠差", "睡不醒", "被压着"],
        "protective": ["精力充沛", "有活力", "充实"],
    },
    "phq5_appetite": {
        "direct": [
            "吃不下",
            "食欲不振",
            "没胃口",
            "胃口不好",
            "不想吃",
            "暴食",
            "吃太多",
            "饭量",
            "体重",
            "瘦了",
            "胖了",
            "减重",
            "减肥",
        ],
        "weak": ["吃饭", "食堂", "聚餐", "做饭", "外卖"],
        "protective": ["吃饭", "聚餐", "做饭", "外卖", "胃口好"],
    },
    "phq6_self_worth": {
        "direct": [
            "自责",
            "失败",
            "没用",
            "废物",
            "糟糕",
            "拖累",
            "对不起",
            "愧疚",
            "内疚",
            "不配",
            "后悔",
            "羞愧",
            "re:让.{0,8}失望",
        ],
        "weak": ["做不好", "完不成", "来不及", "没有准备好", "不满意", "遗憾", "迷茫", "担心"],
        "protective": ["满意", "自信", "完成", "理想", "进步", "负责"],
    },
    "phq7_concentration": {
        "direct": [
            "注意力",
            "专注",
            "集中",
            "效率低",
            "学不进去",
            "看不进去",
            "记不住",
            "脑子",
            "没头绪",
            "毫无头绪",
            "拖延",
        ],
        "weak": ["复习", "考试", "作业", "课程压力", "任务", "安排时间", "来不及", "忙不过来"],
        "protective": ["图书馆", "科学馆", "学习", "完成", "安排", "计划"],
    },
    "phq8_psychomotor": {
        "direct": [
            "动作慢",
            "说话慢",
            "反应慢",
            "迟钝",
            "坐立不安",
            "烦躁",
            "急躁",
            "焦躁",
            "停不下来",
            "不安",
        ],
        "weak": ["紧张", "压力", "喘不过气", "压着", "着急", "慌"],
        "protective": [],
    },
    "phq9_self_harm": {
        "direct": ["不想活", "不如死", "轻生", "自杀", "自伤", "伤害自己", "结束生命", "活着没意思", "想死"],
        # Keep this channel narrow. Generic death/bereavement mentions are not
        # PHQ9 self-harm evidence and caused false positives in audit cases.
        "weak": [],
        "protective": [],
    },
}

FREQUENCY_TERMS = ["每天", "一直", "经常", "总是", "大部分时间", "很多时候", "反复", "持续", "长期", "一整天", "那些日子", "最近"]
SEVERITY_TERMS = ["严重", "特别", "非常", "太", "崩溃", "受不了", "喘不过气", "压抑", "无力", "强烈", "极其", "很"]
NEGATION_TERMS = ["没有", "没", "不", "无", "并不", "不是", "没啥", "不会"]
CORE_FILLERS = ["嗯", "呃", "啊", "额", "唔"]
DISCOURSE_FILLERS = ["就是", "然后", "那个", "这个", "的话", "可能", "其实", "比较", "反正"]
GLOBAL_NEGATIVE_TERMS = sorted({term for item in ITEMS.values() for term in item["direct"] + item["weak"] if not term.startswith("re:")})
GLOBAL_POSITIVE_TERMS = ["开心", "快乐", "高兴", "喜欢", "满意", "充实", "朋友", "旅游", "聚餐", "完成", "理想", "挺好", "不错"]
PRESSURE_TERMS = ["压力", "考试", "复习", "作业", "课程", "任务", "部门", "工作", "科研", "实验室", "就业", "升学", "考研"]
FOOD_TERMS = ["吃饭", "吃", "饭", "食堂", "聚餐", "做饭", "外卖", "胃口", "食欲"]
SOCIAL_TERMS = ["朋友", "同学", "舍友", "室友", "家长", "家人", "老师", "班级", "社交", "人际"]
TEMPLATE_TERMS = ["烦恼的事", "烦恼的事情", "开心的事", "开心的事情", "悲伤的事", "悲伤的事情", "想要改进", "满意的地方"]
STABLE_POSITIVE_TERMS = ["整体还是很开心", "情绪没有太大的波动", "没有特别伤心", "心情不错", "目前状态挺好", "生活挺好", "挺开心的"]
RECOVERY_TERMS = ["后来也感觉没啥", "后来没啥", "没啥的", "还行", "可以控制", "没有影响", "问题不大", "挺好的"]
SEVERE_MOOD_TERMS = ["抑郁", "压抑", "绝望", "喘不过气", "崩溃", "低落", "心情低落", "受不了"]
RUMINATION_TERMS = ["烦恼", "比较多", "运气不好", "倒霉", "尴尬", "反复", "一直", "总是", "又", "每次"]
MINOR_EVENT_TERMS = ["摔", "跌", "钉子", "排不上队", "食堂", "尴尬", "小测", "作业", "排队"]
SLEEP_CONTEXT_TERMS = ["入睡", "睡觉", "睡眠", "睡不着", "失眠", "晚上", "每晚"]
ATTENTION_EXPLICIT_TERMS = ["注意力", "专注", "集中", "效率低", "学不进去", "看不进去", "记不住", "没头绪", "毫无头绪", "拖延"]
TASK_FAILURE_TERMS = ["完不成", "来不及", "没有准备好", "没准备好", "安排时间", "忙不过来", "复习时间很少"]
ROLE_OVERLOAD_TERMS = ["学生工作", "部门工作", "学生干部", "工作以及", "平衡", "超出我的想象", "一堆作业", "没完没了", "写不完", "任务太多", "事情太多", "deadline", "赶在", "一边拖一边焦虑", "拖延"]
DEPLETION_AVOIDANCE_TERMS = ["不想听", "不想写", "不想做", "打累了", "不知道该干点什么", "没有目标", "力不从心", "懒惰", "内耗", "拖延"]
FUTURE_SELF_DOUBT_TERMS = ["不够优秀", "能力不够", "学到的技能不是很多", "不足以", "迷茫", "困惑", "就业压力", "升学压力", "考研", "未来", "出路"]
PROCRASTINATION_ANXIETY_TERMS = ["一边拖一边焦虑", "拖延", "deadline", "赶在", "来不及", "没写完", "没有写完", "没做完", "没有做完"]
FUTURE_DISTANT_TERMS = ["还有一年", "才大三", "以后", "未来", "之后", "战线", "规划", "准备"]
CONTROL_PROTECTIVE_TERMS = ["可以接受", "可控制", "没有影响", "问题不大", "相信自己", "能解决", "已经过去", "还算稳定", "情绪方面还算比较稳定", "无忧无虑", "做自己喜欢做的事情"]
FUNCTION_PROTECTIVE_TERMS = ["完成", "解决", "收获", "成长", "成熟", "独立完成", "找到工作", "推免完成", "朋友", "运动", "打羽毛球", "健身", "出去玩"]
CASUAL_SELF_DEPRECATION_TERMS = ["像一个废物一样", "废物一样", "懒一点", "太懒惰"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Young ASR PHQ-9 evidence features.")
    parser.add_argument("--asr", type=Path, default=DEFAULT_ASR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def repair_mojibake(text: str) -> str:
    markers = ("鍡", "鏈", "鐨", "锛", "紝", "灏", "槸", "€")
    if not isinstance(text, str) or not any(marker in text for marker in markers):
        return text
    try:
        return text.encode("gb18030").decode("utf-8")
    except UnicodeError:
        return text


def load_asr_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["sample_id"] = int(obj["sample_id"])
            obj["text"] = repair_mojibake(obj.get("text", ""))
            rows.append(obj)
    rows.sort(key=lambda row: row["sample_id"])
    return rows


def count_term(text: str, term: str) -> int:
    if term.startswith("re:"):
        return len(re.findall(term[3:], text))
    return text.count(term)


def count_terms(text: str, terms: list[str]) -> int:
    return sum(count_term(text, term) for term in terms)


def find_spans(text: str, terms: list[str]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for term in terms:
        if term.startswith("re:"):
            for match in re.finditer(term[3:], text):
                spans.append((match.start(), match.end(), match.group(0)))
            continue
        start = 0
        while True:
            idx = text.find(term, start)
            if idx < 0:
                break
            spans.append((idx, idx + len(term), term))
            start = idx + max(1, len(term))
    spans.sort()
    return spans


def count_context_terms(text: str, spans: list[tuple[int, int, str]], terms: list[str], window: int = 18) -> int:
    count = 0
    for start, end, _term in spans:
        lo = max(0, start - window)
        hi = min(len(text), end + window)
        count += count_terms(text[lo:hi], terms)
    return count


def count_negated_spans(text: str, spans: list[tuple[int, int, str]], window: int = 8) -> int:
    count = 0
    for start, end, _term in spans:
        lo = max(0, start - window)
        # Only inspect the prefix before the matched symptom. Many Chinese
        # symptom phrases contain negation characters themselves, e.g.
        # "睡不着" and "吃不下"; counting the full span would falsely negate
        # the symptom.
        if count_terms(text[lo:start], NEGATION_TERMS):
            count += 1
    return count


def count_template_spans(text: str, spans: list[tuple[int, int, str]], window: int = 8) -> int:
    count = 0
    for start, end, _term in spans:
        lo = max(0, start - window)
        hi = min(len(text), end + window)
        if count_terms(text[lo:hi], TEMPLATE_TERMS):
            count += 1
    return count


def count_sleep_context_spans(text: str, spans: list[tuple[int, int, str]], window: int = 18) -> int:
    count = 0
    for start, end, _term in spans:
        lo = max(0, start - window)
        hi = min(len(text), end + window)
        if count_terms(text[lo:hi], SLEEP_CONTEXT_TERMS):
            count += 1
    return count


def current_self_harm_spans(text: str, spans: list[tuple[int, int, str]], window: int = 32) -> list[tuple[int, int, str]]:
    historical_cues = ["小学", "初中", "高中", "小时候", "小的时候", "童年", "以前", "当时", "那个时候", "那时候", "曾经", "之前", "前几年"]
    current_cues = ["最近", "现在", "目前", "这几天", "这段时间", "近两周", "这两周", "每天", "一直", "经常", "反复", "还会", "仍然"]
    current: list[tuple[int, int, str]] = []
    for start, end, term in spans:
        lo = max(0, start - window)
        hi = min(len(text), end + window)
        ctx = text[lo:hi]
        if count_terms(ctx, current_cues) or not (count_terms(ctx, historical_cues) or "想过" in ctx):
            current.append((start, end, term))
    return current


def remove_terms_length(text: str, terms: list[str]) -> int:
    length = len(text)
    for term in terms:
        if term.startswith("re:"):
            continue
        length -= len(term) * text.count(term)
    return max(0, length)


def sentence_count(text: str) -> int:
    parts = [part for part in re.split(r"[。！？!?；;，,\s]+", text) if part]
    return len(parts)


def rule_score(direct: int, weak: int, freq_context: int, severity_context: int, negated: int, low_content: int, item: str) -> int:
    if item == "phq9_self_harm":
        return min(3, max(0, direct - negated))
    if low_content:
        return 1 if item in {"phq1_interest", "phq2_mood", "phq4_fatigue", "phq8_psychomotor"} else 0
    signal = max(0, direct - negated)
    if signal >= 2 and (freq_context > 0 or severity_context > 1):
        return 3
    if signal >= 1 and (freq_context > 0 or severity_context > 0):
        return 2
    if signal >= 1 or weak >= 2:
        return 1
    return 0


def cap_score(score: int, cap: int) -> int:
    return max(0, min(score, cap))


def refined_item_score(
    item: str,
    raw_score: int,
    text: str,
    direct_spans: list[tuple[int, int, str]],
    weak_spans: list[tuple[int, int, str]],
    protective_count: int,
    template_context_count: int,
    sleep_context_count: int,
) -> int:
    score = int(raw_score)
    if item == "phq2_mood":
        if template_context_count and not count_terms(text, SEVERE_MOOD_TERMS):
            score = cap_score(score, 1)
        if count_terms(text, STABLE_POSITIVE_TERMS) and protective_count >= 3 and not count_terms(text, SEVERE_MOOD_TERMS):
            score = cap_score(score, 1)
        if count_terms(text, RECOVERY_TERMS) and not count_terms(text, SEVERE_MOOD_TERMS):
            score = cap_score(score, 1)
    elif item == "phq3_sleep":
        if "睡眠质量" in [span[2] for span in direct_spans] and not count_terms(text, ["睡不着", "失眠", "入睡困难", "每天", "每晚", "严重"]):
            score = cap_score(score, 1)
    elif item == "phq5_appetite":
        if not count_terms(text, ["吃不下", "食欲不振", "没胃口", "胃口不好", "不想吃", "暴食", "吃太多", "饭量", "体重", "瘦了", "胖了"]):
            score = 0
        if count_terms(text, ["健身", "减肥", "主动", "锻炼"]) and not count_terms(text, ["吃不下", "食欲不振", "没胃口", "胃口不好", "暴食", "吃太多"]):
            score = 0
    elif item == "phq7_concentration":
        explicit = count_terms(text, ATTENTION_EXPLICIT_TERMS)
        task_failure = count_terms(text, TASK_FAILURE_TERMS)
        if explicit == 0 and task_failure == 0:
            score = 0
        if sleep_context_count and explicit == 0 and task_failure == 0:
            score = 0
    elif item == "phq9_self_harm":
        if not current_self_harm_spans(text, direct_spans):
            score = 0
    return int(score)


def extract_features(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    text = row.get("text", "")
    core_filler_count = count_terms(text, CORE_FILLERS)
    discourse_filler_count = count_terms(text, DISCOURSE_FILLERS)
    filler_count = core_filler_count + discourse_filler_count
    text_len = len(text)
    content_len = remove_terms_length(text, CORE_FILLERS + DISCOURSE_FILLERS)
    low_content = int(text_len < 20)
    limited_content = int(text_len < 80)
    n_sent = sentence_count(text)

    features: dict[str, Any] = {
        "ID": row["sample_id"],
        "asr_text_len": text_len,
        "asr_content_len": content_len,
        "asr_sentence_count": n_sent,
        "asr_avg_sentence_len": text_len / max(n_sent, 1),
        "asr_low_content_flag": low_content,
        "asr_limited_content_flag": limited_content,
        "asr_core_filler_count": core_filler_count,
        "asr_discourse_filler_count": discourse_filler_count,
        "asr_filler_count": filler_count,
        "asr_core_filler_per_100char": 100.0 * core_filler_count / max(text_len, 1),
        "asr_filler_per_100char": 100.0 * filler_count / max(text_len, 1),
        "asr_global_negative_count": count_terms(text, GLOBAL_NEGATIVE_TERMS),
        "asr_global_positive_count": count_terms(text, GLOBAL_POSITIVE_TERMS),
        "asr_pressure_count": count_terms(text, PRESSURE_TERMS),
        "asr_frequency_count": count_terms(text, FREQUENCY_TERMS),
        "asr_severity_count": count_terms(text, SEVERITY_TERMS),
        "asr_food_mention_count": count_terms(text, FOOD_TERMS),
        "asr_social_mention_count": count_terms(text, SOCIAL_TERMS),
        "asr_template_prompt_count": count_terms(text, TEMPLATE_TERMS),
        "asr_stable_positive_count": count_terms(text, STABLE_POSITIVE_TERMS),
        "asr_recovery_count": count_terms(text, RECOVERY_TERMS),
        "asr_severe_mood_count": count_terms(text, SEVERE_MOOD_TERMS),
        "asr_rumination_count": count_terms(text, RUMINATION_TERMS),
        "asr_minor_event_count": count_terms(text, MINOR_EVENT_TERMS),
        "asr_rumination_minor_event_score": count_terms(text, RUMINATION_TERMS) * count_terms(text, MINOR_EVENT_TERMS),
        "asr_attention_explicit_count": count_terms(text, ATTENTION_EXPLICIT_TERMS),
        "asr_task_failure_count": count_terms(text, TASK_FAILURE_TERMS),
        "asr_sleep_context_count": count_terms(text, SLEEP_CONTEXT_TERMS),
        "asr_role_overload_count": count_terms(text, ROLE_OVERLOAD_TERMS),
        "asr_depletion_avoidance_count": count_terms(text, DEPLETION_AVOIDANCE_TERMS),
        "asr_future_self_doubt_count": count_terms(text, FUTURE_SELF_DOUBT_TERMS),
        "asr_procrastination_anxiety_count": count_terms(text, PROCRASTINATION_ANXIETY_TERMS),
        "asr_future_distant_count": count_terms(text, FUTURE_DISTANT_TERMS),
        "asr_control_protective_count": count_terms(text, CONTROL_PROTECTIVE_TERMS),
        "asr_function_protective_count": count_terms(text, FUNCTION_PROTECTIVE_TERMS),
        "asr_casual_self_deprecation_count": count_terms(text, CASUAL_SELF_DEPRECATION_TERMS),
        "asr_overload_risk_score": count_terms(text, ROLE_OVERLOAD_TERMS) + count_terms(text, DEPLETION_AVOIDANCE_TERMS) + count_terms(text, PROCRASTINATION_ANXIETY_TERMS),
        "asr_future_anxiety_risk_score": count_terms(text, FUTURE_SELF_DOUBT_TERMS) + count_terms(text, PROCRASTINATION_ANXIETY_TERMS),
        "asr_protective_cap_score": count_terms(text, CONTROL_PROTECTIVE_TERMS) + count_terms(text, FUNCTION_PROTECTIVE_TERMS) + count_terms(text, STABLE_POSITIVE_TERMS) + count_terms(text, RECOVERY_TERMS),
    }
    audit: dict[str, Any] = {
        "ID": row["sample_id"],
        "text": text,
        "items": {},
    }

    rule_sum = 0
    for item_name, item_terms in ITEMS.items():
        direct_spans = find_spans(text, item_terms["direct"])
        weak_spans = find_spans(text, item_terms["weak"])
        protective_spans = find_spans(text, item_terms["protective"])
        direct_count = len(direct_spans)
        weak_count = len(weak_spans)
        protective_count = len(protective_spans)
        freq_context_count = count_context_terms(text, direct_spans + weak_spans, FREQUENCY_TERMS)
        severity_context_count = count_context_terms(text, direct_spans + weak_spans, SEVERITY_TERMS)
        negated_count = count_negated_spans(text, direct_spans)
        template_context_count = count_template_spans(text, direct_spans + weak_spans)
        sleep_context_count = count_sleep_context_spans(text, direct_spans + weak_spans)
        score = rule_score(
            direct_count,
            weak_count,
            freq_context_count,
            severity_context_count,
            negated_count,
            low_content,
            item_name,
        )
        refined_score = refined_item_score(
            item_name,
            score,
            text,
            direct_spans,
            weak_spans,
            protective_count,
            template_context_count,
            sleep_context_count,
        )
        rule_sum += score
        prefix = f"asr_{item_name}"
        features[f"{prefix}_direct_count"] = direct_count
        features[f"{prefix}_weak_count"] = weak_count
        features[f"{prefix}_protective_count"] = protective_count
        features[f"{prefix}_freq_context_count"] = freq_context_count
        features[f"{prefix}_severity_context_count"] = severity_context_count
        features[f"{prefix}_negated_count"] = negated_count
        features[f"{prefix}_template_context_count"] = template_context_count
        features[f"{prefix}_sleep_context_count"] = sleep_context_count
        features[f"{prefix}_has_direct"] = int(direct_count > 0)
        features[f"{prefix}_has_weak"] = int(weak_count > 0)
        features[f"{prefix}_has_protective"] = int(protective_count > 0)
        features[f"{prefix}_evidence_strength"] = max(
            0,
            2 * direct_count + weak_count + freq_context_count + severity_context_count - protective_count - negated_count,
        )
        features[f"{prefix}_rule_score"] = score
        features[f"{prefix}_refined_rule_score"] = refined_score
        audit["items"][item_name] = {
            "direct": [span[2] for span in direct_spans],
            "weak": [span[2] for span in weak_spans],
            "protective": [span[2] for span in protective_spans],
            "rule_score": score,
            "refined_rule_score": refined_score,
            "template_context_count": template_context_count,
            "sleep_context_count": sleep_context_count,
        }

    features["asr_phq_rule_sum"] = rule_sum
    features["asr_phq_refined_rule_sum"] = sum(
        int(features[f"asr_{item_name}_refined_rule_score"]) for item_name in ITEMS
    )
    return features, audit


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_asr_rows(args.asr)
    feature_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for row in rows:
        features, audit = extract_features(row)
        feature_rows.append(features)
        audits.append(audit)

    feature_path = args.out_dir / "young_asr_phq9_evidence_features.csv"
    audit_path = args.out_dir / "young_asr_phq9_evidence_audit.jsonl"
    with feature_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)
    with audit_path.open("w", encoding="utf-8") as f:
        for audit in audits:
            f.write(json.dumps(audit, ensure_ascii=False) + "\n")

    print(f"wrote {len(feature_rows)} rows to {feature_path}")
    print(f"wrote audit to {audit_path}")


if __name__ == "__main__":
    main()
