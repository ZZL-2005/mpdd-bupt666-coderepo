# ASR strong prior + IMU key models

打包日期：2026-05-06

这份是精简参考包，只保留两条关键探索：

1. ASR V4 强先验，CodaBench Score `0.457751`。
2. IMU accel3 + BigFive OrdinalRidge，CodaBench Score 约 `0.6702`。

其他音频、多专家、9x3、反馈 probe、融合小修都没有展示。

## 目录

```text
obs/scripts/
  transcribe_young_dashscope_asr.py
  extract_young_asr_phq9_evidence_features.py
  run_young_strong_prior_router_9x3.py
  build_young_asr_v4_test_submission.py
  run_young_asr_v4_test_pipeline.py
  build_young_imu_accel3_ordinalridge_submission.py

obs/asr/
  young_dashscope_asr_event1_dashscope_clean_with_id17_chunked.jsonl
  young_test_dashscope_asr_event1_clean.jsonl

obs/experiments/young_asr_phq9_evidence/
obs/experiments/young_asr_phq9_evidence_test_clean/
obs/experiments/young_strong_prior_router_9x3/

official_baseline/make_submission_forcodabench/
  young_asr_v4_submission_clean(_validated)/
  young_accel3_raw(_validated)/
  young_imu_accel3_ordinalridge(_validated)/
```

## ASR 转写

ASR 转写已经放进包里，接收方不需要重新调用 ASR：

- 训练侧 Young event_1：`obs/asr/young_dashscope_asr_event1_dashscope_clean_with_id17_chunked.jsonl`
- 测试侧 Young event_1：`obs/asr/young_test_dashscope_asr_event1_clean.jsonl`

包里也包含已经由这些转写抽取好的 ASR PHQ-9 evidence features 和 audit 文件：

- `obs/experiments/young_asr_phq9_evidence/young_asr_phq9_evidence_features.csv`
- `obs/experiments/young_asr_phq9_evidence/young_asr_phq9_evidence_audit.jsonl`
- `obs/experiments/young_asr_phq9_evidence_test_clean/young_asr_phq9_evidence_features.csv`
- `obs/experiments/young_asr_phq9_evidence_test_clean/young_asr_phq9_evidence_audit.jsonl`

如果只是复查 ASR V4 结果，直接看 `young_asr_v4_submission_clean_validated/submission.zip` 和 `RESULTS.md` 即可。

## 两条模型线

### 1. ASR V4 strong prior

核心脚本：

```powershell
python obs/scripts/extract_young_asr_phq9_evidence_features.py --asr obs/asr/young_dashscope_asr_event1_dashscope_clean_with_id17_chunked.jsonl --out_dir obs/experiments/young_asr_phq9_evidence
python obs/scripts/run_young_strong_prior_router_9x3.py --n_splits 5 --n_repeats 10 --select_k 64
python obs/scripts/build_young_asr_v4_test_submission.py --asr_features obs/experiments/young_asr_phq9_evidence_test_clean/young_asr_phq9_evidence_features.csv --asr_audit obs/experiments/young_asr_phq9_evidence_test_clean/young_asr_phq9_evidence_audit.jsonl --out_dir official_baseline/make_submission_forcodabench/young_asr_v4_submission_clean
```

说明：`run_young_strong_prior_router_9x3.py` 中的 V4 规则可直接读，`V4_DESIGN.md` 解释了 PHQ-9 item 对齐和 region router。

### 2. IMU accel3 + BigFive OrdinalRidge

核心脚本：

```powershell
python obs/scripts/build_young_imu_accel3_ordinalridge_submission.py
```

该模型只用 IMU 前 3 个通道构造频带能量、周期、自相关、peak interval、三轴相关和加速度幅值统计，再拼接 BigFive 分数，用 `mord.OrdinalRidge(alpha=1.0)` 预测 PHQ total，binary/ternary 从 PHQ 阈值派生。

leaderboard 约 `0.6702` 的提交在：

```text
official_baseline/make_submission_forcodabench/young_accel3_raw_validated/submission.zip
```

`young_imu_accel3_ordinalridge/` 是训练脚本直接生成的 IMU 明细输出，`young_accel3_raw/` 是对应的 leaderboard 记录候选目录。

## 数据要求

包内没有原始数据集。完整重跑 IMU 或从头构建 ASR 测试提交时，仍需要本地具备：

```text
data/Train-MPDD-Young/Young/
data/Test-MPDD-Young/Young/
```

ASR 音频转写已经提供，因此不需要重新访问 DashScope，除非要重新转写或替换 ASR 版本。

## 依赖

```bash
pip install -r requirements_reference.txt
```

主要依赖是 `numpy`、`pandas`、`scipy`、`scikit-learn`、`mord` 和 `requests`。
