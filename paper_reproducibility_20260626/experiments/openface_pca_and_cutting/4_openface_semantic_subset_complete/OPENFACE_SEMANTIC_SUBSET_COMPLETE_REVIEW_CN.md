# OpenFace 语义子集裁剪完整性核对

整理日期：2026-06-26

来源文件：

```text
高低维对比实验/3_OpenFace子集裁剪/openface_pruned_results.json
高低维对比实验/3_OpenFace子集裁剪/openface_pruned_analysis.py
```

## 结论先说

之前展示的 OpenFace 语义子集裁剪结果 **覆盖了主要 subset 结论，但不是完整展示**。

主要遗漏：

1. `Per_event` 这一行没有展示。它和 `AU_plus_Gaze` 数值完全相同，因为脚本里两者在当前特征构造下都等价于“所有非 dynamic 的 AU + gaze per-event 特征”。
2. 没有展示 `personality_baseline`。
3. 没有展示 AU 逐项相关性分析，包括 `au_ranking_mean`、`au_dynamic_range_top5` 和完整 `au_correlations`。
4. 没有展示 JSON 中保存的 `best_direct_ridge`、`best_logistic`、`best_residual`、`best_terf1` 汇总字段。

## 子集定义

脚本中 OpenFace subject vector 包含两部分：

```text
1. per-event features:
   event_1 / event_2 / event_3 内的 AU intensity、AU presence、gaze 统计

2. dynamic features:
   event_1 与 event_2/event_3 平均状态的差异
```

每个 event 内提取：

```text
AU intensity: mean, std, max, range, pct_active
AU presence: mean, std
Gaze: mean, std
```

## 完整 subset 结果

| Subset | 维度 | Direct Ridge binary acc | Direct Ridge ternary F1 | Logistic binary acc | Logistic binary F1 | Residual binary acc | Residual ternary F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gaze_only | 36 | 0.5909 | 0.3598 | 0.5000 | 0.4990 | **0.8182** | 0.5505 |
| AU_intensity_mean_only | 51 | 0.5455 | 0.4179 | 0.5000 | 0.4990 | 0.5909 | 0.4720 |
| AU_intensity_std_only | 51 | 0.5000 | 0.5431 | 0.4545 | 0.4500 | 0.5909 | 0.5823 |
| Dynamic_range | 80 | 0.6364 | 0.3667 | 0.5000 | 0.4905 | 0.7727 | 0.3905 |
| AU_presence_stats | 108 | 0.6364 | 0.4737 | 0.5000 | 0.4905 | 0.5455 | 0.3952 |
| AU_intensity_stats | 255 | 0.5455 | 0.3056 | **0.5909** | **0.5901** | 0.5909 | 0.4069 |
| AU_all_stats | 363 | 0.6818 | **0.6746** | 0.5455 | 0.5455 | 0.7273 | **0.6500** |
| AU_plus_Gaze | 399 | **0.7273** | 0.4462 | 0.5455 | 0.5455 | 0.7273 | 0.5807 |
| Per_event | 399 | **0.7273** | 0.4462 | 0.5455 | 0.5455 | 0.7273 | 0.5807 |
| All_features | 479 | 0.6818 | 0.4620 | 0.5455 | 0.5455 | 0.7273 | 0.5250 |

## 最优项

| 标准 | 最优 subset | 数值 |
|---|---|---:|
| Direct Ridge binary acc | AU_plus_Gaze / Per_event | 0.7273 |
| Logistic binary acc | AU_intensity_stats | 0.5909 |
| Residual binary acc | Gaze_only | 0.8182 |
| Residual ternary F1 | AU_all_stats | 0.6500 |

## Personality baseline

| baseline | binary accuracy | binary F1 | ternary accuracy | ternary F1 |
|---|---:|---:|---:|---:|
| Personality BigFive | 0.8636 | 0.8611 | 0.6818 | 0.4598 |

注意：OpenFace residual 指标不是 OpenFace 单模态指标。它的含义是：先用 personality/Ridge 预测 PHQ-9，再用 OpenFace 子集预测残差，最终组合后算分类指标。

## AU 相关性补充结果

`au_ranking_mean` 给出 AU intensity mean 与 PHQ-9 的相关性排序，绝对值最高的几个是：

| AU | mean r |
|---|---:|
| AU09 | 0.1872 |
| AU06 | 0.1406 |
| AU10 | 0.1264 |
| AU05 | 0.1157 |
| AU12 | 0.1025 |
| AU04 | -0.0926 |
| AU15 | -0.0881 |

`au_dynamic_range_top5` 显示 event 间动态变化里相关性较高的项：

| AU | statistic | r |
|---|---|---:|
| AU04 | mean | -0.2363 |
| AU07 | std | 0.2206 |
| AU14 | std | 0.2035 |
| AU12 | std | 0.1696 |
| AU12 | mean | 0.1412 |

这些 AU 逐项相关性整体不算强，适合作为“AU/gaze 有弱信号、需要聚合和融合”的辅助说明，不适合作为强单变量结论。

## 对最终模型 OpenFace 处理的支撑程度

这组结果可以支持：

```text
OpenFace 不应直接盲目使用全部高维特征；AU 和 gaze 这类语义明确的子集更值得保留。
```

具体证据：

```text
Gaze_only 36维 residual binary acc = 0.8182 > All_features 479维的 0.7273
AU_plus_Gaze / Per_event 399维 direct binary acc = 0.7273 > All_features 479维的 0.6818
AU_all_stats 363维 residual ternary F1 = 0.6500 > All_features 479维的 0.5250
```

但这组结果不能严格证明：

```text
最终模型中的 auvel / e3only / fatigue / auspec 四个模块分别有效。
```

因为这里的评价是本地 binary/residual 分析，不是最终 CodaBench Score，也不是最终四个 OpenFace 模块逐个加入的消融。

## 建议论文表述

稳妥写法：

```text
A semantic subset analysis of OpenFace features showed that AU- and gaze-related subsets provided stronger or more stable signals than the complete high-dimensional OpenFace representation. In particular, the gaze-only subset achieved the best residual binary accuracy, while AU-related statistics improved ternary residual F1. These observations motivated our final OpenFace feature engineering strategy, where AU and gaze streams were summarized through event-level statistics, temporal changes, and spectral descriptors rather than using all raw OpenFace dimensions directly.
```

中文解释：

```text
OpenFace 语义子集分析显示，AU 和 gaze 相关子集比完整高维 OpenFace 表征更有用或更稳定。因此，最终模型围绕 AU/gaze 构建事件统计、时序变化和频谱特征，而不是直接使用全部 OpenFace 维度。
```

## 生成文件

- `openface_feature_subset_results_complete.csv`：完整 10 个 subset 指标。
- `openface_personality_baseline.csv`：personality baseline。
- `openface_au_correlation_long.csv`：完整 AU 相关性长表。
- `openface_au_mean_ranking.csv`：AU mean 相关性排序。
- `openface_dynamic_range_top5.csv`：dynamic range top5。
- `openface_best_methods.csv`：best-method 汇总。
- `01_openface_subset_all_metrics.png`：完整 subset 多指标图。
- `02_openface_key_subset_comparison.png`：关键 subset 对比图。
- `03_openface_au_rankings.png`：AU 排名图。
