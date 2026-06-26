# Results

本包只解释两个 leaderboard 上测过的关键结果。

## 1. ASR V4 strong prior

提交目录：

```text
official_baseline/make_submission_forcodabench/young_asr_v4_submission_clean_validated/
```

CodaBench 结果：

| metric | value |
|---|---:|
| Score | 0.457751 |
| Cls_F1 | 0.583333 |
| Cls_CCC | 0.508394 |
| Cls_Kappa | 0.281525 |
| Binary_ACC | 0.636364 |
| Binary_MacroF1 | 0.633333 |
| Ternary_ACC | 0.590909 |
| Ternary_MacroF1 | 0.533333 |

方法摘要：

```text
Young event_1 ASR text
-> PHQ-9 evidence feature extraction
-> V4 region router
-> nine PHQ item priors, each 0..3
-> sum as PHQ-9
-> derive binary/ternary labels by thresholds 5 and 10
```

训练侧 V4 强先验在 Young train 上的可解释性很强：

| method | PHQ MAE | label2 acc | label3 macro-F1 |
|---|---:|---:|---:|
| `strong_prior_raw_sum_v4` | 0.966 | 0.875 | 0.836 |
| `strong_prior_v4_isotonic` | 1.077 | 0.855 | 0.780 |

但这个结果来自对 train split 的人工审计和规则修订，不能当作无偏泛化分数。hidden test 上的 `0.457751` 更能说明它单独作为提交模型还不够稳。

价值：

- 这是目前最清楚的 PHQ-9 questionnaire-aligned ASR 证据层。
- 每个样本都有 `region` 和 `PHQ1..PHQ9` item vector，便于解释错误。
- 它适合作为后续融合、规则诊断、case audit 的基础，而不是直接替代 IMU 主模型。

主要问题：

- 低内容、trait-amplified overload 等区域在测试集上容易过度上调。
- ASR 单模态与 IMU 单模态在测试集上差异很大，二者不是简单校准差异。

## 2. IMU accel3 + BigFive OrdinalRidge

提交目录：

```text
official_baseline/make_submission_forcodabench/young_accel3_raw_validated/
```

CodaBench 已知结果：

| metric | value |
|---|---:|
| Score | ~0.6702 |

候选记录文件：

```text
official_baseline/make_submission_forcodabench/young_accel3_raw/fusion_report.json
```

方法摘要：

```text
12-dim IMU sequence
-> use first 3 acceleration channels
-> band energy ratios, band CVs
-> autocorrelation period and peak interval features
-> acceleration correlations and magnitude statistics
-> append BigFive scores
-> StandardScaler + mord.OrdinalRidge(alpha=1.0)
-> map ordinal class back to observed PHQ total
-> derive binary/ternary labels by PHQ thresholds 5 and 10
```

预测分布：

```text
test_n = 22
PHQ mean = 4.318
binary counts = 12 normal / 10 positive
ternary counts = 12 normal / 9 mild / 1 severe
```

价值：

- 这是当前 Young test 上最稳的主干。
- 不依赖 ASR 文本，泛化反馈明显好于 ASR V4 单模态。
- 特征数量少、结构简单，便于复现和解释。

主要问题：

- 训练侧 OOF 指标不漂亮，但 hidden test 反馈好，说明本任务验证集很小，不能只看本地 CV。
- IMU 对某些个体会给出较高 PHQ，ASR 可能给低分；已有测试反馈显示不能轻易用 ASR 跨阈值覆盖 IMU。

## ASR vs IMU 的关系

测试集上两者预测相关性很低：

```text
IMU OrdinalRidge mean PHQ: 4.955
ASR V4 mean PHQ: 4.500
Pearson correlation: 0.057
Spearman correlation: 0.051
exact same PHQ: 1/22
within 2 PHQ points: 7/22
binary disagreement: 14/22
ternary disagreement: 14/22
```

结论：ASR V4 和 IMU accel3 捕捉的是不同信号。当前最合理的表述是：

- **IMU accel3 OrdinalRidge 是 leaderboard 主模型。**
- **ASR V4 是可解释的 PHQ evidence / strong-prior 探索。**
- 不建议把 ASR V4 直接跨阈值覆盖 IMU；如果融合，应优先做同类别内校准或作为受约束 evidence feature。
