# Young Strong-Prior Router 9x3

This experiment tests a deliberately strong-prior approach for PHQ-9 prediction.
The core output is not a flat score: it predicts nine PHQ item scores in 0..3
and sums them as the PHQ-9 total.

```text
ASR text evidence + Big-Five gates
-> region router
-> PHQ1..PHQ9 prior item scores, each 0..3
-> sum as PHQ-9 total
```

## Current Result

Young train, 88 samples, 5-fold x 10 repeats for comparable reporting.
`strong_prior_raw_sum_v4` is deterministic, so repeated folds duplicate the same
per-sample prediction; the fold statistics are for comparison with learned
baselines only.

| method | PHQ MAE | label2 acc | label3 macro-F1 | mean pred |
|---|---:|---:|---:|---:|
| `strong_prior_raw_sum_v4` | 0.966 | 0.875 | 0.836 | 4.989 |
| `strong_prior_v4_isotonic` | 1.077 | 0.855 | 0.780 | 4.788 |
| `strong_prior_raw_sum_v3` | 2.557 | 0.625 | 0.454 | 4.466 |
| `non_asr_ridge_reference` | 2.656 | 0.608 | 0.404 | 4.855 |
| `strong_prior_raw_sum_broad` | 2.739 | 0.580 | 0.452 | 4.080 |

V4 explains all 88 Young-train samples within 2 PHQ points: 25 exact, 66 within
1 point, 88 within 2 points.

Important caveat: V4 was built by auditing this train split. Treat it as a
strong prior design and error-analysis artifact, not an unbiased held-out
estimate. The next real test is whether the same regions transfer.

## CodaBench Test Result

Young test submission built from fixed ASR strong-prior V4, recorded on
2026-05-05. Submission:
`official_baseline/make_submission_forcodabench/young_asr_v4_submission_clean_validated/submission.zip`.

| metric | value |
|---|---:|
| Score | 0.457751 |
| Cls_F1 | 0.583333 |
| Cls_CCC | 0.508394 |
| Cls_Kappa | 0.281525 |
| Binary_ACC | 0.636364 |
| Binary_MacroF1 | 0.633333 |
| Binary_Kappa | 0.272727 |
| Binary_CCC | 0.508394 |
| Binary_RMSE | 0.695098 |
| Binary_MAE | 0.478937 |
| Ternary_ACC | 0.590909 |
| Ternary_MacroF1 | 0.533333 |
| Ternary_Kappa | 0.290323 |
| Ternary_CCC | 0.508394 |
| Ternary_RMSE | 0.695098 |
| Ternary_MAE | 0.478937 |

Submission method: DashScope `qwen3-asr-flash` transcribes Young test event_1
audio, ASR text is converted into PHQ-aligned evidence features, Big-Five scores
act as gates, V4 deterministically predicts nine item scores in 0..3 and sums
them as `phq9_pred`. Binary and ternary labels are derived only from the PHQ
sum using thresholds 5 and 10.

## IMU OrdinalRidge Comparator

A separate user-provided comparator was reproduced locally on 2026-05-06. It
uses IMU features plus parsed Big-Five scores, then fits `mord.OrdinalRidge` on
the observed Young-train PHQ totals treated as ordered classes. Reported
CodaBench score: about 0.66.

Method summary:

```text
12-dim IMU sequence
-> band energy ratios, band CVs, autocorrelation period, peak interval features
-> acceleration correlations and magnitude statistics
-> append five Big-Five scores as ordinary numeric features
-> StandardScaler + OrdinalRidge(alpha=1.0)
-> map ordinal class back to observed PHQ total
-> derive binary/ternary thresholds from PHQ
```

Local comparison against ASR V4 on the 22 Young test IDs:

- IMU OrdinalRidge mean PHQ: 4.955; ASR V4 mean PHQ: 4.500.
- Pearson correlation between the two PHQ predictions: 0.057.
- Spearman correlation: 0.051.
- Exact same PHQ prediction: 1/22.
- Within 2 PHQ points: 7/22.
- Binary disagreement: 14/22.
- Ternary disagreement: 14/22.

This suggests the two methods are not merely calibrated differently; they are
responding to substantially different signal families. The comparison table is
stored at `test_asr_v4_vs_imu_ordinalridge.csv`.

## PHQ-Only Fusion Feedback

All fusion attempts below output a single `phq9_pred`; binary and ternary labels
are derived from PHQ thresholds 5 and 10.

Observed CodaBench feedback:

- Raw IMU + Big-Five OrdinalRidge: reported about 0.66.
- ASR V4 alone: 0.457751.
- PHQ average of IMU and ASR V4: reported about 0.5x.
- `class_preserving_plus_lowcontent`: reported about 0.4x. This changed ID 83
  from IMU PHQ 0 to ASR-guided PHQ 10, so low-content ASR should not be treated
  as a reliable high-risk override on test.
- `class_preserving_shrink`: reported 0.47. This preserves IMU binary/ternary
  classes and only shrinks PHQ within class toward ASR, so the drop is PHQ
  regression/CCC-driven. The hidden test PHQ magnitude appears closer to raw
  IMU OrdinalRidge than to ASR-shrunk values.

Current interpretation: ASR V4 is diagnostically interesting but harmful as a
test-time PHQ fusion signal for Young. The best operating point so far is to
keep the raw IMU + Big-Five OrdinalRidge submission, not ASR-average or
ASR-override it.

## V4 Regions

| region | n | mean true | mean prior | interpretation |
|---|---:|---:|---:|---|
| `low_content` | 1 | 10.00 | 10.00 | Near non-response, e.g. only filler; treated as clinically suspicious. |
| `trait_amplified_overload` | 7 | 10.14 | 9.29 | High neuroticism amplifies overload, future doubt, depletion. |
| `roommate_interpersonal_stress` | 1 | 11.00 | 11.00 | Dorm/interpersonal pressure plus sleep decline and avoidance. |
| `rumination_minor_events` | 1 | 10.00 | 8.00 | Repeated negative attribution around small events. |
| `overload_depletion` | 9 | 7.33 | 7.89 | Workload, deadline, procrastination-anxiety, depletion loops. |
| `physical_burden` | 3 | 6.67 | 7.00 | Illness, poor body state, sleep/appetite effects. |
| `relationship_family_burden` | 3 | 6.67 | 6.67 | Relationship/family illness/conflict burden. |
| `task_blockage` | 5 | 6.40 | 6.20 | Research/project stuck, no progress, task failure. |
| `off_prompt_story` | 2 | 4.50 | 4.50 | Tells another-person story with self-involved distress. |
| `social_isolation` | 1 | 9.00 | 7.00 | Explicit difficulty communicating/loneliness/self-worth. |
| `unresolved_grief` | 1 | 6.00 | 5.00 | Family breakup or grief content with current sadness. |
| `functional_stress` | 5 | 2.20 | 2.60 | Stress exists but functioning/protection is strong. |
| `historical_resolved` | 2 | 1.00 | 1.50 | Past events or resolved distress dominate. |
| `protected_conflict` | 6 | 2.33 | 3.33 | Negative phrases plus explicit protection/control/recovery. |
| `direct_symptom` | 13 | 3.92 | 4.23 | Direct symptom content without a stronger region. |
| `conflict_positive_negative` | 24 | 3.25 | 3.63 | Both positive and negative content, common in event_1. |
| `weak_semantic` | 4 | 1.00 | 0.75 | Little direct ASR symptom evidence. |

## Files

- `strong_prior_item_scores.csv`: all prior item scores, regions, labels.
- `v4_case_audit.csv`: per-sample audit sorted by residual, with item vector and ASR excerpt.
- `router_metrics_summary.csv`: method-level metrics.
- `v4_region_summary.csv`: V4 region calibration summary.
- `router_predictions.csv`: repeated CV-format predictions for all methods.

## Reproduce

```powershell
$env:PYTHONIOENCODING='utf-8'
E:\window_program\miniconda\envs\scripts\python.exe obs\scripts\extract_young_asr_phq9_evidence_features.py --asr obs\asr\young_dashscope_asr_event1_dashscope_clean_with_id17_chunked.jsonl --out_dir obs\experiments\young_asr_phq9_evidence
E:\window_program\miniconda\envs\scripts\python.exe obs\scripts\run_young_strong_prior_router_9x3.py --n_splits 5 --n_repeats 10 --select_k 64
```
