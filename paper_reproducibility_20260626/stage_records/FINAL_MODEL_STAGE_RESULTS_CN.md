# 最终模型阶段结果与模块实验速查

整理日期：2026-06-26

用途：这份文件专门记录最终 Young-track 模型的阶段性提升线，以及目前能找到的独立模块/辅助实验结果。它适合写论文、汇报或答辩时快速查阅。

## 结论先行

我们有一条比较可靠的阶段性演进线：

```text
ORIG baseline              0.6701
+ OpenFace AU/gaze         0.7063
+ velocity / multi-ranker  0.7178
+ variance calibration     0.7931
+ AND classification rule  ~0.834
final thresholding         approx 0.87
```

但需要注意：这不是严格的 add-one ablation 表。也就是说，目前没有完整保留如下统一设置下的结果表：

```text
ORIG
ORIG + auvel
ORIG + auvel + e3only
ORIG + auvel + e3only + fatigue
ORIG + auvel + e3only + fatigue + auspec
```

更准确的说法是：

> 我们保留了历史 leaderboard 阶段结果，并保留了最终 consolidated pipeline 的可复现脚本；但早期中间阶段并非每一步都有独立、干净、可一键复现的阶段脚本。

## 一、最终模型阶段线

| 阶段 | 加入/改变了什么 | Score | F1 | CCC | Kappa | 证据类型 | 是否有脚本 |
|---|---|---:|---:|---:|---:|---|---|
| ORIG baseline | IMU 前 3 个加速度通道 + BigFive + OrdinalRidge | 0.6701 | 0.7972 | 0.5083 | 0.7048 | 历史提交记录 + 可复现主干 | 有：`orig_model.py` |
| + OpenFace AU/gaze | 加入 OpenFace AU/gaze 面部统计特征 | 0.7063 | 0.7972 | 0.6170 | 0.7048 | 历史提交记录 | 最终 ranker 脚本能支撑方法，但不是当时精确提交脚本 |
| + velocity / multi-ranker | 加入表情速度特征和多路面部融合 | 0.7178 | 0.7972 | 0.6515 | 0.7048 | 历史提交记录 | 最终 `auvel/e3only/fatigue/auspec` 脚本能支撑方法 |
| + variance calibration | 对融合 PHQ 预测做均值/方差校准 | 0.7931 | 0.7972 | 0.8772 | 0.7048 | 历史提交记录 + 最终代码 | 有：`run_pipeline.py` |
| + AND classification rule | 二分类要求 blend PHQ 和 ORIG 同时支持阳性 | 0.8340-0.8347 | 0.8255-0.8361 | 0.8772 | 0.7887-0.8012 | 历史提交记录 + 最终代码 | 有：`run_pipeline.py` |
| final thresholding | 最终阈值 `t1=4.25, t2=11` | approx 0.87 | 未完整保留 | 0.8772 | 未完整保留 | 最终提交记录 + validated submission | 有：`run_pipeline.py` |

推荐论文写法：

> Starting from the IMU-personality OrdinalRidge baseline, the system was progressively improved by introducing OpenFace AU/gaze dynamics, facial velocity and multi-ranker fusion, variance calibration for CCC, and an AND-based classification rule. The historical leaderboard progression improved from 0.6701 to approximately 0.87, while the final consolidated pipeline is fully reproducible from the released scripts.

中文写法：

> 我们从 IMU + BigFive 的 OrdinalRidge 主干出发，逐步引入 OpenFace AU/gaze 动态特征、表情速度与多路面部 ranker、针对 CCC 的方差校准，以及面部-步态一致性的 AND 分类规则。历史 leaderboard 记录显示分数从 0.6701 提升到约 0.87；最终整合后的 pipeline 已在干净复现脚本中保留。

不建议写法：

> 我们做了严格的五路 add-one 消融，并证明每一路单独提升。

原因：这张严格 add-one 表没有完整保留。

## 二、最终可复现脚本索引

最终 clean package 路径：

```text
01_final_openface_fusion/mpdd_young_repro_final_clean_20260602/
```

最终一键入口：

```text
repro_scripts/run_pipeline.py
```

最终模型组件脚本：

| 组件 | 脚本 | 说明 |
|---|---|---|
| ORIG | `repro_scripts/orig_model.py` | IMU 前 3 个加速度通道 + BigFive + OrdinalRidge |
| auvel | `repro_scripts/build_auvel_ranker.py` | 3 个 event 的 OpenFace mean/std/velocity，387 维 |
| e3only | `repro_scripts/build_e3only_ranker.py` | 只用 event 3 的 OpenFace mean/std/velocity，129 维 |
| fatigue | `repro_scripts/build_fatigue.py` | event 间 OpenFace 均值差，129 维 |
| auspec | `repro_scripts/build_auspec.py` | OpenFace AU/gaze 时序频谱特征，645 维 |
| MLP/CCC 工具 | `repro_scripts/mlp_utils.py` | 50-seed CCC-loss MLP 的公共代码 |
| 最终融合/校准/分类 | `repro_scripts/run_pipeline.py` | 五路融合、方差校准、AND 分类规则、打包 submission |

最终融合公式：

```text
raw = 0.40*ORIG + 0.30*auvel + 0.20*e3only + 0.10*fatigue + 0.05*auspec
phq9 = variance_calibrate(raw)
binary = (phq9 >= t1) AND (ORIG >= 4)
ternary = threshold(phq9, t1, t2), then force ternary=0 if binary=0
```

最终阈值：

```text
t1 = 4.25
t2 = 11
```

复现命令：

```powershell
cd 01_final_openface_fusion/mpdd_young_repro_final_clean_20260602
$env:BLEND_PHQ_THRESHOLD='4.25'
$env:TERNARY_T2='11.0'
$env:OUT_NAME='young_final_t4p25_t2p11'
python repro_scripts/run_pipeline.py
```

## 三、独立模块或辅助实验结果

### 1. ORIG 独立主干

| 模块 | 方法 | 指标 | 结果 | 证据 |
|---|---|---|---:|---|
| ORIG | IMU accel3 + BigFive + OrdinalRidge | CodaBench / historical score | 0.6701-0.6702 | 历史提交记录 + `orig_model.py` |
| ORIG LOO reference | 同上 | LOO CCC | 0.1736 | 本地 LOO 参考 |
| ORIG classification baseline | 同上 | LOO F1 / Kappa | 0.6238 / 0.1962 | 分类规则训练内对照 |

可写结论：ORIG 是稳定主干，负责提供步态/人格侧的低维强基线。

### 2. 四路 OpenFace ranker 的模块设计记录

| ranker | 维度 | 特征构造 | 临床/行为动机 | 是否进入最终模型 | 独立数值证据 |
|---|---:|---|---|---|---|
| auvel | 387 | 每个 event 取 OpenFace mean/std/abs-diff velocity，3 个 event 拼接 | 表情运动速度，反映运动迟缓 | 是 | reference auvel LOO CCC 0.0217；更多作为最终融合组件证明 |
| e3only | 129 | 只用 event 3 的 mean/std/velocity | 任务末段更能暴露疲劳后状态 | 是 | 文档记录：给 auvel 带来约 +0.022 CCC 的正交增量，但未保留成严格 add-one 表 |
| fatigue | 129 | E2-E1、E3-E2、E3-E1 的 OpenFace 均值差 | 跨事件疲劳/状态漂移 | 是 | 作为最终四路 ranker 保留；无单独 CodaBench 表 |
| auspec | 645 | FFT 低/中/高频能量比、谱熵、主频 | 表情动态节律、单调性 | 是 | 作为最终四路 ranker 保留；无单独 CodaBench 表 |

可写结论：四路 ranker 的设计动机和复现脚本都保留了，但不要说每一路都有独立 leaderboard 消融分数。

### 3. OpenFace 子特征/时间特征探索

这些结果能说明我们确实做过模块级探索，但它们多为本地 LOO 或方向性判断，不是最终 CodaBench add-one 消融。

| 实验 | 维度/设置 | 指标 | 结果 | 结论 |
|---|---:|---|---:|---|
| reference auvel | auvel 参考路 | LOO CCC | 0.0217 | 面部 LOO 口径噪声大，只能方向性参考 |
| reference ORIG | ORIG 参考 | LOO CCC | 0.1736 | IMU 主干本地参考 |
| C1 事件内分段轨迹 | 645 维 | LOO CCC | 0.2229 | 有信号，适合作 future work |
| C1 与 auvel 相关 | 645 维 | Pearson corr | 0.414 | 偏正交，但未进最终版 |
| C2 面部 Hjorth 复杂度 | 258 维 | LOO CCC | -0.0181 | 明确无效 |
| C2 与 auvel 相关 | 258 维 | Pearson corr | 0.367 | 没有形成有效互补 |
| e1vel/e2vel | 各 129 维 | 相关性判断 | 与 auvel/e3only 相关性过高 | 冗余，未进最终 |
| headpose | 54 维 ranker | 实测判断 | 无明显增益 | 未进最终 |

可写结论：时间/频谱/跨事件设计是经过筛选的；并不是把所有 OpenFace 派生特征都堆进去。

### 4. OpenFace 子集裁剪实验

这组结果来自 `高低维对比实验/3_OpenFace子集裁剪`，是本地验证实验，不是最终模型 CodaBench 消融，但可以辅助说明 OpenFace 内部哪些子特征更有信号。

| 子集 | 维度 | direct ridge binary acc | residual binary acc | residual ternary F1 | 结论 |
|---|---:|---:|---:|---:|---|
| All features | 479 | 0.6818 | 0.7273 | 0.5250 | 全部特征并非最优 |
| AU + Gaze | 399 | 0.7273 | 0.7273 | 0.5807 | direct binary 最好 |
| AU all stats | 363 | 0.6818 | 0.7273 | 0.6500 | 三分类残差信号最好 |
| Dynamic range | 80 | 0.6364 | 0.7727 | 0.3905 | 二分类残差信号有用 |
| Gaze only | 36 | 0.5909 | 0.8182 | 0.5505 | 残差二分类最好 |

注意：`residual binary acc` 是人格基线 + OpenFace 残差修正后的准确率，不是 OpenFace 单模态准确率。

### 5. 方差校准独立效果

| 对比 | Score | F1 | CCC | Kappa | 结论 |
|---|---:|---:|---:|---:|---|
| velocity / multi-ranker before calibration | 0.7178 | 0.7972 | 0.6515 | 0.7048 | 校准前 CCC 仍受方差失配限制 |
| first calibrated version | 0.7563 | 0.7972 | 0.7668 | 0.7048 | 校准后 CCC 大幅上升 |
| best calibrated CCC version | 0.7931 | 0.7972 | 0.8772 | 0.7048 | 回归头最佳 CCC 版本 |

可写结论：方差校准是最清楚、最能解释的回归提升模块。它不改变样本排序，主要通过匹配预测分布的均值和方差来改善 CCC 分母中的均值偏移和方差失配项。

### 6. AND 分类规则独立效果

历史提交结果：

| 设置 | Score | F1 | CCC | Kappa | 说明 |
|---|---:|---:|---:|---:|---|
| best calibrated CCC version | 0.7931 | 0.7972 | 0.8772 | 0.7048 | 分类仍由普通阈值导出 |
| `young_and_rule` | 0.8340-0.8347 | 0.8255-0.8361 | 0.8772 | 0.7887-0.8012 | AND 规则提升分类指标 |

训练集 LOO 支撑：

| 规则 | LOO F1 | LOO Kappa |
|---|---:|---:|
| ORIG baseline | 0.6238 | 0.1962 |
| AND, t1=4.3 | 0.6347 | 0.2343 |
| AND, t1=4.2 | 0.6238 | 0.2163 |

可写结论：AND 规则有训练集 LOO 支撑，同时在 leaderboard 上显著提升 F1/Kappa。但最终阈值 `t1=4.25, t2=11` 是 leaderboard feedback 选定，需要如实说明。

## 四、哪些不能包装成强消融

| 内容 | 当前证据状态 | 写论文建议 |
|---|---|---|
| Stage 1 权重融合 | 有历史分数，但没有干净阶段脚本 | 弱化为 preliminary weight tuning |
| `ORIG + auvel + e3only + fatigue + auspec` 逐项 add-one | 没有完整保留统一表 | 不要写成严格消融 |
| 单个 face ranker 的独立 CodaBench 分数 | 未完整保留 | 只写设计动机、脚本、辅助 LOO/相关性证据 |
| 最终阈值 `4.25/11` | 有最终脚本和提交，但阈值由 leaderboard 反馈选定 | 可写最终工程选择，不要说是训练集最优 |
| 早期同学高分 0.7477 | 有记录但不属于最终可复现主线 | 不纳入主方法表 |

## 五、推荐论文表格

### 表 1：Final Model Progression

| Step | Main change | Score | F1 | CCC | Kappa |
|---|---|---:|---:|---:|---:|
| ORIG baseline | IMU accel3 + BigFive OrdinalRidge | 0.6701 | 0.7972 | 0.5083 | 0.7048 |
| + OpenFace AU/gaze | facial AU/gaze dynamics | 0.7063 | 0.7972 | 0.6170 | 0.7048 |
| + velocity/multi-ranker | facial velocity and multi-view face fusion | 0.7178 | 0.7972 | 0.6515 | 0.7048 |
| + variance calibration | moment matching for CCC | 0.7931 | 0.7972 | 0.8772 | 0.7048 |
| + classification rule | AND rule for binary/ternary labels | approx 0.834 | approx 0.83 | 0.8772 | approx 0.79-0.80 |
| Final | threshold-tuned final submission | approx 0.87 | not retained | 0.8772 | not retained |

表注建议：

> Intermediate scores are historical leaderboard submission records. The final consolidated pipeline is reproducible, while some intermediate submissions were not preserved as independent clean scripts.

中文表注：

> 中间阶段分数来自历史 leaderboard 提交记录；最终整合 pipeline 可复现，但部分中间阶段没有保留为独立干净脚本。

## 六、证据文件索引

阶段结果：

```text
source_archives/项目完整技术文档(1).md
OTHER_EXPERIMENT_RESULTS.md
```

最终可复现脚本：

```text
01_final_openface_fusion/mpdd_young_repro_final_clean_20260602/repro_scripts/run_pipeline.py
01_final_openface_fusion/mpdd_young_repro_final_clean_20260602/repro_scripts/orig_model.py
01_final_openface_fusion/mpdd_young_repro_final_clean_20260602/repro_scripts/build_auvel_ranker.py
01_final_openface_fusion/mpdd_young_repro_final_clean_20260602/repro_scripts/build_e3only_ranker.py
01_final_openface_fusion/mpdd_young_repro_final_clean_20260602/repro_scripts/build_fatigue.py
01_final_openface_fusion/mpdd_young_repro_final_clean_20260602/repro_scripts/build_auspec.py
01_final_openface_fusion/mpdd_young_repro_final_clean_20260602/repro_scripts/mlp_utils.py
```

OpenFace 子集裁剪辅助实验：

```text
05_high_low_dim_comparison/3_OpenFace子集裁剪/openface_pruned_results.json
05_high_low_dim_comparison/3_OpenFace子集裁剪/openface_pruned_analysis.py
```

分类规则训练内依据：

```text
source_archives/项目完整技术文档(1).md  第 11 节
OTHER_EXPERIMENT_RESULTS.md  Local CV / LOO results
```