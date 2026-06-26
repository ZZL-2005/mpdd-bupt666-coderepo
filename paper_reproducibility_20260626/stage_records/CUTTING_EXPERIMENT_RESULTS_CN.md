# 特征裁剪 / 降维实验结果汇总

整理日期：2026-06-26

用途：这份文件专门整理可以支撑论文中 **特征裁剪、降维、小样本下高维特征容易过拟合** 这一叙事的实验结果，并索引对应复现脚本。

注意：不同实验使用的评价指标不完全一样。写论文时一定要标清楚指标类型，不要把 `CodaBench Score`、`binary accuracy`、`residual binary accuracy`、`CV MAE` 混在同一列里直接比较。

## 口径说明

| 指标名 | 含义 | 是否官方最终指标 | 使用建议 |
|---|---|---|---|
| CodaBench Score | CodaBench 记录的综合分数 | 是 | 可用于主结果或强消融 |
| documented score | 文档或 README 中记录的分数 | 部分是 | 可引用，但应说明来自历史记录 |
| binary acc | 本地验证集二分类准确率 | 否 | 适合作为趋势分析和辅助实验 |
| macro-F1 | 本地验证集二分类 macro-F1 | 否 | 与 binary acc 一起展示更稳妥 |
| residual binary acc | 人格基线 + OpenFace 残差修正后的二分类准确率 | 否 | 不能写成 OpenFace 单模态准确率 |
| CV MAE | 训练集交叉验证 PHQ-9 MAE | 否 | 只能说明训练内验证趋势，不能替代 hidden/test |

## 一、最强支撑实验

### 1. IMU 通道裁剪：12 维 IMU vs 前 3 个加速度通道

这是目前最能支撑 “cutting down 有效” 的一组结果。注意两边都包含 BigFive 人格特征，所以它不是纯 IMU-only 对比，而是：

```text
12 维 IMU + BigFive
vs
前 3 个加速度通道 + BigFive
```

| 设置 | 方法 | 输入通道/原始配置 | 输入特征维度 | 模型实际维度 | 指标 | 结果 | 结论 |
|---|---|---|---:|---:|---|---:|---|
| 高维 IMU | StandardScaler + OrdinalRidge | 12 个 IMU 通道 + BigFive | 约 149 = 12 通道 x 12 手工特征 + 5 | 约 149 | documented score | 0.3633 | 全部 IMU 通道严重过拟合 |
| 紧凑 IMU | StandardScaler + OrdinalRidge | 前 3 个加速度通道 + BigFive | 41 = 36 IMU + 5 BigFive | 41 | CodaBench Score | **0.6662** | 通道级裁剪得到强基线 |
| 紧凑 IMU + 轻微 PCA | StandardScaler + PCA + OrdinalRidge | 前 3 个加速度通道 + BigFive | 41 | 35 | CodaBench Score | **0.6709** | 41 维轻微压缩到 35 维后有小幅额外提升 |

可以写进论文的表述：

> 在同样引入 BigFive 人格特征的条件下，仅保留 IMU 的前三个加速度通道明显优于使用全部 12 个 IMU 通道，CodaBench/documented score 从 0.3633 提升到 0.6662，说明小样本场景下通道级裁剪可以有效降低过拟合。

不要写：

> 3 维 IMU 单独达到了 0.6662。

因为 `0.6662` 对应的是 **3 维 IMU + BigFive**。

证据文件：

```text
04_gp_previous_results/Young_GP/v2_OrdinalRidge_0.6662/REPRODUCTION_GUIDE.md
04_gp_previous_results/Young_GP/v2_OrdinalRidge_0.6662/RESULTS.md
04_gp_previous_results/Young_GP/sota_pca35_0.6709/
05_high_low_dim_comparison/README.md
```

复现脚本：

```text
04_gp_previous_results/Young_GP/v2_OrdinalRidge_0.6662/generate_submission_ordinalridge.py
04_gp_previous_results/Young_GP/sota_pca35_0.6709/gen_sota_pca35.py
05_high_low_dim_comparison/5_IMU_3维vs12维/validate_12dim.py
```

代码仓库注意事项：这些脚本里有服务器硬编码路径，例如 `/data/zilu/mpdd2026/...`。后续放入代码仓库前，建议改成命令行参数或配置文件。

### 2. OpenFace 高维原始特征 vs 低维 PCA

这是同一模态下比较直接的高维/低维对比。实验方法统一为 `StandardScaler + PCA + LogisticRegression`，指标是本地验证集 binary accuracy / macro-F1。

| 设置 | 方法 | 输入维度 | 模型实际维度 | binary acc | macro-F1 | 说明 |
|---|---|---:|---:|---:|---:|---|
| raw OpenFace | raw + LR | 710 | 710 | 0.5000 | 0.3333 | 直接使用高维 OpenFace 表征几乎没有有效信号 |
| OpenFace PCA5 | PCA5 + LR | 710 | 5 | **0.6818** | **0.6812** | 最优低维区间之一 |
| OpenFace PCA15 | PCA15 + LR | 710 | 15 | **0.6818** | **0.6812** | 与 PCA5 持平 |
| OpenFace PCA30 | PCA30 + LR | 710 | 30 | 0.6364 | 0.6333 | 继续增加维度反而下降 |

可以写进论文的表述：

> 直接使用 710 维 OpenFace 特征时，二分类准确率仅为 0.5000；而将其压缩到 5 或 15 个主成分后，准确率提高到 0.6818，说明高维原始表征在 88 个训练样本下并不稳定。

证据文件：

```text
05_high_low_dim_comparison/1_PCA维度扫描_dim_reduction/dim_reduction_analysis.json
05_high_low_dim_comparison/2_全特征基准_raw_vs_pca/all_features_benchmark.json
```

复现脚本：

```text
05_high_low_dim_comparison/1_PCA维度扫描_dim_reduction/dim_reduction_analysis.py
05_high_low_dim_comparison/2_全特征基准_raw_vs_pca/all_features_benchmark.py
```

指标说明：这里是 `binary accuracy / macro-F1`，不是 CodaBench 总分。

### 3. OpenFace 语义子集裁剪

这是 OpenFace 内部不同子集之间的裁剪对比。它比“高维 OpenFace 单模态 vs 低维 OpenFace 融合”更严格，因为它们来自同一个本地分析脚本。

| OpenFace 子集 | 方法 / 评价方式 | 输入维度 | 模型实际维度 | direct ridge binary acc | logistic binary acc | residual binary acc | residual ternary F1 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| All features | Ridge / LR / residual | 479 | 479 | 0.6818 | 0.5455 | 0.7273 | 0.5250 | 全部派生特征并不是最优 |
| AU + Gaze | Ridge / LR / residual | 399 | 399 | **0.7273** | 0.5455 | 0.7273 | 0.5807 | direct binary 最好 |
| AU all stats | Ridge / LR / residual | 363 | 363 | 0.6818 | 0.5455 | 0.7273 | **0.6500** | 三分类残差信号最好 |
| AU intensity stats | Ridge / LR / residual | 255 | 255 | 0.5455 | 0.5909 | 0.5909 | 0.4069 | 信号较弱 |
| AU presence stats | Ridge / LR / residual | 108 | 108 | 0.6364 | 0.5000 | 0.5455 | 0.3952 | 中等偏弱 |
| Dynamic range | Ridge / LR / residual | 80 | 80 | 0.6364 | 0.5000 | 0.7727 | 0.3905 | 二分类残差信号有用，三分类弱 |
| AU intensity mean only | Ridge / LR / residual | 51 | 51 | 0.5455 | 0.5000 | 0.5909 | 0.4720 | 单一统计量不足 |
| AU intensity std only | Ridge / LR / residual | 51 | 51 | 0.5000 | 0.4545 | 0.5909 | 0.5823 | 三分类残差较好但二分类弱 |
| Gaze only | Ridge / LR / residual | 36 | 36 | 0.5909 | 0.5000 | **0.8182** | 0.5505 | 二分类残差信号最好 |

最关键的维度裁剪对比：

| 对比 | 高维/全集 | 低维/裁剪 | 指标 | 变化 |
|---|---|---|---|---:|
| OpenFace residual correction | All features: 479 维，0.7273 | Gaze only: 36 维，0.8182 | residual binary acc | +0.0909 |
| OpenFace direct Ridge | All features: 479 维，0.6818 | AU + Gaze: 399 维，0.7273 | direct ridge binary acc | +0.0455 |
| OpenFace ternary residual | All features: 479 维，0.5250 | AU all stats: 363 维，0.6500 | residual ternary F1 | +0.1250 |

可以写进论文的表述：

> 在 OpenFace 子集分析中，36 维 gaze 子集在人格基线残差修正后的二分类准确率达到 0.8182，高于 399 维 AU+Gaze 和 479 维全特征集合的 0.7273。这说明语义明确的低维面部子集可能比盲目使用全部 OpenFace 派生特征更稳。

不要写：

> Gaze-only 单模态准确率达到 0.8182。

因为 `residual binary acc` 的意思是：

> 先用人格特征建立 baseline，再看 OpenFace 子集作为残差修正时能把二分类准确率提高到多少。

证据文件：

```text
05_high_low_dim_comparison/3_OpenFace子集裁剪/openface_pruned_results.json
```

复现脚本：

```text
05_high_low_dim_comparison/3_OpenFace子集裁剪/openface_pruned_analysis.py
```

## 二、辅助支撑实验

### 1. 多模态 PCA 维度扫描

这组实验是有结果的，属于“跨特征族 PCA 维度扫描”。它不是最终 CodaBench 消融，而是在同一验证划分上比较不同特征族经过 PCA 压缩后的二分类准确率。

实验方法统一为：`StandardScaler + PCA + LogisticRegression`。下面这张表适合直接放进论文或汇报，因为每一行都明确给出方法、输入维度、模型实际使用维度和指标。

| 特征族 | 方法 | 输入维度 | 模型维度 | binary acc | macro-F1 | 说明 |
|---|---|---:|---:|---:|---:|---|
| OpenSmile | raw + LR | 65 | 65 | 0.5000 | 0.3333 | 原始声学特征基准 |
| OpenSmile | PCA6 + LR | 65 | 6 | **0.8182** | **0.8167** | 最优低维结果；PCA8 的 acc 也为 0.8182 |
| OpenFace | raw + LR | 710 | 710 | 0.5000 | 0.3333 | 原始高维 OpenFace 基准 |
| OpenFace | PCA5 + LR | 710 | 5 | **0.6818** | **0.6812** | 最优低维结果；PCA30 降到 0.6364 |
| BERT ASR | PCA5 + LR | 768 | 5 | **0.7273** | **0.7250** | raw 结果未在 benchmark 中保留；PCA15 持平，PCA30 降到 0.5909 |
| emo2vec mean | PCA20 + LR | 768 | 20 | 0.6364 | 0.6364 | PCA30 也为 0.6364，整体信号较弱 |
| IMU hand-crafted | raw + LR | 36/39 | 36/39 | 0.5909 | 0.5686 | 脚本命名为 36d，但 benchmark 记录矩阵为 39 维 |
| IMU hand-crafted | PCA15 + LR | 36 | 15 | **0.6818** | **0.6758** | PCA20 持平，PCA30 降到 0.6364 |
| Personality BigFive | raw + LR | 5 | 5 | 0.8636 | 0.8611 | 低维人格特征本身已有强二分类信号 |
| Personality BigFive | PCA3 + LR | 5 | 3 | **0.9091** | **0.9083** | 在该验证划分上有小幅提升 |

这里的 `36/39` 是因为脚本中该特征族名称写作 `IMU (36d)`，但实际特征矩阵记录为 39 维；论文中建议统一表述为“hand-crafted IMU features”，避免在正文里纠结这个脚本命名差异。

可以写进论文的表述：

> 多个模态的 PCA 扫描结果显示，最佳维度通常集中在 3 到 20 维之间；继续增加维度往往不能带来收益，反而可能降低性能。

证据文件：

```text
05_high_low_dim_comparison/1_PCA维度扫描_dim_reduction/dim_reduction_analysis.json
05_high_low_dim_comparison/2_全特征基准_raw_vs_pca/all_features_benchmark.json
```

复现脚本：

```text
05_high_low_dim_comparison/1_PCA维度扫描_dim_reduction/dim_reduction_analysis.py
05_high_low_dim_comparison/2_全特征基准_raw_vs_pca/all_features_benchmark.py
```

### 2. Elder 特征数裁剪

这不是最终 Young track 主线，但可以作为小样本下特征裁剪有效的辅助证据。结果来自 `高低维对比实验/README.md` 中保留的脚本输出记录。

| 设置 | 方法 | 输入维度 | 模型实际维度 | validation score | Binary F1 | Ternary F1 | 说明 |
|---|---|---:|---:|---:|---:|---:|---|
| full features | Ridge-based baseline | 56 | 56 | 0.4223 | 未保留 | 未保留 | 全量特征基准 |
| selected top-30 | feature selection + model | 56 | 30 | 0.4341 | 0.663 | 0.549 | 小幅高于全量 |
| selected top-35 | feature selection + model | 56 | 35 | 0.4979 | 0.714 | 0.587 | 继续提升 |
| selected top-40 | feature selection + model | 56 | 40 | **0.5654** | **0.766** | **0.628** | 最优裁剪点 |
| selected top-45 | feature selection + model | 56 | 45 | 0.4595 | 0.714 | 0.516 | 维度继续增加后下降 |

可以写进论文的表述：

> 在 Elder 数据上，特征数从 56 裁剪到 40 后 validation score 从 0.4223 提升到 0.5654，进一步说明小样本场景下适当控制特征维度是必要的。

证据文件：

```text
05_high_low_dim_comparison/README.md
```

复现脚本：

```text
05_high_low_dim_comparison/7_Elder_特征数选择_56vs40/elder_exp5_feature_selection.py
```

使用建议：这可以作为辅助证据，不要和 Young 最终 CodaBench 分数直接混表。

## 三、必须交代的反例

### 1. Young G+P：41 维裁到 25 维反而下降

这组结果非常重要，因为它说明 “裁剪不是越多越好”。原始 41 维已经是较强的手工 IMU + BigFive 特征，再做纯 CV 驱动的细粒度筛选会在 hidden/test 上下降。

训练内 CV 选择过程：

| 设置 | 方法 | 输入维度 | 模型实际维度 | CV MAE | CV std | 说明 |
|---|---|---:|---:|---:|---:|---|
| selected top-25 | feature importance selection + OrdinalRidge | 41 | 25 | **3.059** | 0.479 | CV 最优 |
| selected top-30 | feature importance selection + OrdinalRidge | 41 | 30 | 3.092 | 0.251 | CV 次优 |
| selected top-35 | feature importance selection + OrdinalRidge | 41 | 35 | 3.206 | 0.355 | CV 继续变差 |
| selected top-40 | feature importance selection + OrdinalRidge | 41 | 40 | 3.410 | 0.480 | 接近全量 |
| full Young G+P | StandardScaler + OrdinalRidge | 41 | 41 | 3.465 | 未保留 | CV 看起来最差 |

hidden/test 对比：

| 设置 | 方法 | 输入维度 | 模型实际维度 | hidden/test score | Cls_F1 | Cls_CCC | Cls_Kappa | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| full Young G+P | StandardScaler + OrdinalRidge | 41 | 41 | **0.6662** | **0.7972** | **0.4966** | **0.7048** | 测试集表现更好 |
| selected top-25 | feature selection + StandardScaler + OrdinalRidge | 41 | 25 | 0.6011 | 0.7556 | 0.4304 | 0.6175 | CV 更好，但测试集下降 |

可以写进论文的表述：

> 进一步的细粒度特征选择并不总能泛化。虽然从 41 维筛到 25 维使 CV MAE 从 3.465 降到 3.059，但 hidden/test score 从 0.6662 降到 0.6011，说明纯 CV 驱动的特征选择在 88 个样本下可能过拟合。

证据文件：

```text
05_high_low_dim_comparison/4_YoungGP_特征数选择_41vs25/RESULTS.md
05_high_low_dim_comparison/4_YoungGP_特征数选择_41vs25/feature_selection_results.csv
05_high_low_dim_comparison/4_YoungGP_特征数选择_41vs25/selected_features.csv
```

复现脚本：

```text
05_high_low_dim_comparison/4_YoungGP_特征数选择_41vs25/feature_selection_experiment.py
```

### 2. IMU PCA 太激进也会下降

| 设置 | 方法 | 输入维度 | 模型实际维度 | 指标 | 结果 | 结论 |
|---|---|---:|---:|---|---:|---|
| 3 维加速度通道 + BigFive | StandardScaler + OrdinalRidge | 41 | 41 | CodaBench Score | **0.6662** | 不做 PCA 的强基线 |
| 3 维加速度通道 + BigFive，PCA2 | StandardScaler + PCA2 + OrdinalRidge | 41 | 2 | documented score | 0.4911 | 过度压缩，丢失物理意义 |
| 3 维加速度通道 + BigFive，PCA35 | StandardScaler + PCA35 + OrdinalRidge | 41 | 35 | CodaBench Score | **0.6709** | 轻微压缩有小幅收益 |

可以写进论文的表述：

> 轻微降维可以带来小幅提升，但过度压缩会破坏 IMU 手工特征中的物理意义。例如 PCA2 使得分数降到 0.4911，而 PCA35 仅进行轻微压缩，分数为 0.6709。

证据文件：

```text
04_gp_previous_results/Young_GP/v2_OrdinalRidge_0.6662/REPRODUCTION_GUIDE.md
05_high_low_dim_comparison/6_IMU_PCA消融_PC1-4/pca_ablation_study.py
```

复现脚本：

```text
04_gp_previous_results/Young_GP/sota_pca35_0.6709/gen_sota_pca35.py
05_high_low_dim_comparison/6_IMU_PCA消融_PC1-4/pca_ablation_study.py
```

## 四、BigFive 与 IMU 单组件说明

这些结果可以帮助解释模型，但不要过度包装。

| 组件 / 设置 | 方法 | 输入维度 | 模型实际维度 | 指标 | 结果 | 说明 |
|---|---|---:|---:|---|---:|---|
| BigFive-only / P-only | 文档记录 | 5 | 5 | documented score | 0.4232 | 单独人格特征不足以完成完整任务 |
| BigFive raw 5d | raw + LR | 5 | 5 | binary acc / macro-F1 | 0.8636 / 0.8611 | 对二分类有较强信号 |
| BigFive PCA3 | PCA3 + LR | 5 | 3 | binary acc / macro-F1 | 0.9091 / 0.9083 | 本地二分类验证中有小幅提升 |
| 3-axis IMU-only | 本地脚本 | 3 通道 | 未固化 | 未保留 | 未在 README/JSON 中固化分数 | 不建议直接引用固定分数 |
| 3-axis IMU + BigFive | StandardScaler + OrdinalRidge | 41 | 41 | CodaBench Score | 0.6662 | 最强早期 G+P 基线 |
| 3-axis IMU + BigFive + PCA35 | StandardScaler + PCA35 + OrdinalRidge | 41 | 35 | CodaBench Score | 0.6709 | 早期 G+P 最好记录 |

3-axis IMU-only 相关脚本：

```text
05_high_low_dim_comparison/5_IMU_3维vs12维/young_3dim_imu_only.py
```

## 五、推荐论文表格

可以单独做一张表，标题类似：

```text
High-dimensional vs. compact feature representations under small-sample training
```

中文标题可以是：

```text
小样本场景下高维特征与紧凑特征表示的对比
```

| 实验 | 方法 | 高维/全集设置 | 紧凑/裁剪设置 | 指标 | 变化 |
|---|---|---|---|---|---:|
| IMU 通道裁剪 | OrdinalRidge | 约 149 维：12 IMU ch + BigFive，0.3633 | 41 维：3 accel ch + BigFive，0.6662 | documented/CodaBench score | +0.3029 |
| IMU 轻微 PCA | OrdinalRidge | 41 维：0.6662 | 35 维：PCA35，0.6709 | CodaBench Score | +0.0047 |
| OpenFace PCA | LR | 710 维 raw：0.5000 | 5 维 PCA5：0.6818 | binary accuracy | +0.1818 |
| OpenFace 子集裁剪 | residual correction | 479 维 all features：0.7273 | 36 维 gaze only：0.8182 | residual binary accuracy | +0.0909 |
| Elder 特征选择 | feature selection | 56 维 full：0.4223 | 40 维 selected：0.5654 | validation score | +0.1431 |
| Young G+P 反例 | feature selection | 41 维 full：0.6662 | 25 维 selected：0.6011 | hidden/test score | -0.0651 |
| IMU 过度 PCA 反例 | PCA + OrdinalRidge | 41 维 full：0.6662 | 2 维 PCA2：0.4911 | documented score | -0.1751 |

推荐论文段落：

> 实验结果表明，在 88 个训练样本的小样本条件下，高维原始特征往往不稳定。结构化的通道裁剪、低维 PCA 和语义引导的 OpenFace 子集选择能够提升模型鲁棒性；但裁剪并非越多越好，纯 CV 驱动的细粒度特征选择可能过拟合训练分布，并在 hidden/test 集上退化。

## 六、后续整理到代码仓库前的检查清单

1. 把脚本中的硬编码服务器路径 `/data/zilu/mpdd2026/...` 改成命令行参数。
2. 增加统一 `requirements.txt`，至少包含 `numpy`、`pandas`、`scipy`、`scikit-learn`、`mord`。
3. 每个实验目录放一个 README，说明数据路径和运行命令。
4. 建议目录结构分成 `scripts/` 和 `results/`。
5. 明确区分本地 binary accuracy 实验和 CodaBench submission score 实验。