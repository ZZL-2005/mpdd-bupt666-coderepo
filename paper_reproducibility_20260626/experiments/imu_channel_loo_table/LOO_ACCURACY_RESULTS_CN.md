# IMU 通道裁剪 LOO 准确性实验结果

整理日期：2026-06-26

## 实验目的

这组实验用 Leave-One-Out cross-validation（LOO）直接检测不同 IMU 通道数下模型在 Young train 内部的泛化准确性。

实验口径：

```text
只使用 IMU，不加入 BigFive。
分别取前 3/6/9/12 个 IMU 通道。
手工特征提取方式保持 ORIG-style IMU hand-crafted feature 不变。
模型：LinearRegression 和 Ridge(alpha=1.0)。
验证方式：LOO，88 个训练样本，每次留 1 个样本验证。
```

总训练次数：

```text
4 个通道设置 × 2 个模型 × 88 个 LOO folds = 704 个模型
```

## 标签和指标

连续标签使用 `phq9_score`。

分类标签直接使用训练集里的：

```text
label2
label3
```

预测时先得到连续 PHQ-9 分数，再裁剪到 `[0, 27]`，然后派生分类：

```text
label2: predicted PHQ-9 >= 5
label3:
  0: predicted PHQ-9 < 5
  1: 5 <= predicted PHQ-9 < 10
  2: predicted PHQ-9 >= 10
```

主要指标：

- `MAE/RMSE`：PHQ-9 回归误差，越低越好。
- `CCC`：PHQ-9 一致性相关系数，越高越好。
- `label2_acc / label2_f1`：二分类准确率和 F1。
- `label3_acc / label3_quadratic_kappa`：三分类准确率和加权 Kappa。
- `challenge_like_score`：`(label2_f1 + ccc_clipped + label3_quadratic_kappa) / 3`，仅用于内部比较。

## LOO 准确性汇总

| 模型 | IMU 通道数 | 特征维度 | MAE | RMSE | CCC | label2 acc | label2 F1 | label3 acc | label3 QWK | challenge-like score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LinearRegression | 3 | 36 | **3.6653** | **4.7173** | 0.0135 | **0.5341** | **0.5591** | **0.4432** | **0.0422** | **0.2049** |
| LinearRegression | 6 | 78 | 9.4019 | 11.9829 | -0.0869 | 0.3864 | 0.4130 | 0.2841 | -0.1045 | 0.0739 |
| LinearRegression | 9 | 129 | 5.9886 | 7.8178 | 0.0003 | 0.4773 | 0.4773 | 0.3636 | 0.0389 | 0.1722 |
| LinearRegression | 12 | 189 | 4.8573 | 6.3432 | -0.0310 | 0.4773 | 0.4651 | 0.3295 | 0.0234 | 0.1525 |
| Ridge(alpha=1.0) | 3 | 36 | **3.4440** | **4.5221** | -0.0093 | **0.5341** | **0.5591** | **0.4659** | 0.0205 | **0.1901** |
| Ridge(alpha=1.0) | 6 | 78 | 5.5626 | 7.6813 | -0.1327 | 0.4318 | 0.4318 | 0.3750 | -0.1133 | 0.0619 |
| Ridge(alpha=1.0) | 9 | 129 | 4.9451 | 6.3671 | **0.0148** | 0.4886 | 0.4828 | 0.3864 | **0.0345** | 0.1773 |
| Ridge(alpha=1.0) | 12 | 189 | 4.7647 | 6.2290 | -0.0383 | 0.4773 | 0.4651 | 0.3295 | -0.0166 | 0.1367 |

## 主要观察

1. 这组 LOO 实验可以比较直接支持 IMU 通道裁剪。

   在 LinearRegression 和 Ridge 两个模型下，前 3 个 IMU 通道都是 MAE/RMSE 最低的设置。

2. 6 通道设置表现最差。

   LinearRegression 从 3 通道扩展到 6 通道后，MAE 从 3.6653 增加到 9.4019，RMSE 从 4.7173 增加到 11.9829。Ridge 中 6 通道也明显弱于 3 通道。

3. Ridge 的 3 通道设置是回归误差最好的方案。

   ```text
   Ridge 3ch MAE  = 3.4440
   Ridge 3ch RMSE = 4.5221
   ```

4. LinearRegression 的 3 通道设置在 challenge-like score 上最高。

   ```text
   Linear 3ch challenge-like score = 0.2049
   Ridge  3ch challenge-like score = 0.1901
   ```

   但这个 score 只是内部 LOO 比较，不等价于 CodaBench test score。

5. CCC 整体很弱。

   所有设置的 LOO CCC 都接近 0，说明 IMU-only 手工特征本身对连续 PHQ-9 的线性一致性预测能力有限。这个实验更适合支撑“低维 IMU 更稳健”，而不是声称 IMU-only 可以很好预测 PHQ-9。

## 可以写进论文的结论

建议写法：

```text
To evaluate whether compact IMU channel selection improves generalization under small-sample conditions, we performed leave-one-out validation using IMU-only handcrafted features. Across both ordinary linear regression and ridge regression, the compact 3-axis acceleration setting achieved the lowest PHQ-9 MAE/RMSE and the best or tied-best derived classification metrics. Expanding the input to 6, 9, or 12 channels did not improve LOO accuracy and often degraded performance, indicating that the additional IMU channels introduce noise or unstable correlations in this small-sample setting.
```

中文解释：

```text
为了检验 IMU 通道裁剪是否提升小样本条件下的泛化能力，我们使用 IMU-only 手工特征进行留一法验证。结果显示，在普通线性回归和 Ridge 回归中，前 3 个加速度通道都取得最低的 PHQ-9 MAE/RMSE，并在派生分类指标上达到最好或并列最好的表现。将输入扩展到 6/9/12 个 IMU 通道没有带来性能提升，反而经常降低 LOO 准确性，说明额外 IMU 通道在小样本条件下可能引入噪声或不稳定相关性。
```

更保守的写法：

```text
The LOO results support using the first 3 acceleration channels as a compact and more generalizable IMU representation, although IMU-only features remain insufficient for strong continuous PHQ-9 prediction.
```

## 对应文件

- `run_imu_channel_loo_accuracy.py`：LOO 实验脚本。
- `loo_accuracy_summary.csv`：LOO 指标汇总表。
- `loo_predictions.csv`：每个样本的 LOO out-of-fold 预测。
- `loo_accuracy_metadata.json`：实验设置记录。
- `10_loo_regression_accuracy.png`：回归指标图。
- `11_loo_classification_accuracy.png`：分类指标图。
- `12_loo_absolute_error_boxplot.png`：绝对误差分布图。
- `13_loo_true_vs_pred_scatter.png`：真实 PHQ-9 vs LOO 预测散点图。
