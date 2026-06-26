# Young IMU 维度选择结果复核

来源目录：`Young_IMU维度选择`。核心结果来自 `维度选择结果_experiment7.json` 的 `experiment7_imu_dimension_importance`。

## 口径核对

README 里写“单维 + 人格，共 15 特征”是正确的。脚本中每个单独 IMU dim 会提取 10 个 IMU 统计特征，再拼接 5 个 BigFive 人格特征，因此总特征数为 15。JSON 中的 `n_features=10` 指的是该单个 IMU dim 的 IMU 特征数，不包含 BigFive。

增量实验中，基础配置是前 3 个加速度维度 + BigFive，共 41 个特征；之后每新增一个 IMU dim，会增加 10 个该 dim 的统计特征，所以特征数从 41 变成 51、61、71，直到 131。

## 单维度测试

| Dim | 传感器 | IMU特征数 | 总特征数 | LOO Score | Binary F1 | Ternary F1 | MAE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | Accel X | 10 | 15 | 0.3804 | 0.6452 | 0.3736 | 2.8409 |
| 1 | Accel Y | 10 | 15 | 0.3324 | 0.5618 | 0.3274 | 2.8864 |
| 2 | Accel Z | 10 | 15 | 0.3359 | 0.5714 | 0.3221 | 2.9318 |
| 3 | Gyro X | 10 | 15 | 0.3913 | 0.6067 | 0.4193 | 2.6932 |
| 4 | Gyro Y | 10 | 15 | 0.3374 | 0.5773 | 0.3172 | 2.9205 |
| 5 | Gyro Z | 10 | 15 | 0.3817 | 0.6383 | 0.3618 | 2.7614 |
| 6 | Mag X | 10 | 15 | **0.4257** | **0.6813** | 0.4089 | 2.6705 |
| 7 | Mag Y | 10 | 15 | 0.2953 | 0.5169 | 0.2952 | 3.0114 |
| 8 | Mag Z | 10 | 15 | 0.3411 | 0.5934 | 0.3299 | 2.8750 |
| 9 | Other 9 | 10 | 15 | 0.3651 | 0.6316 | 0.3560 | 2.8864 |
| 10 | Other 10 | 10 | 15 | 0.3817 | 0.6154 | 0.3564 | **2.5795** |
| 11 | Other 11 | 10 | 15 | 0.3553 | 0.5934 | 0.3382 | 2.8295 |

## 3维基础上增量加维

| 配置 | 新增维度 | 特征数 | LOO Score | 相对3维变化 |
|---|---|---:|---:|---:|
| base_dims_0_1_2 | - | 41 | **0.3965** | 0.0000 |
| 3dim_plus_dims_3 | 3 | 51 | 0.3701 | -0.0264 |
| 3dim_plus_dims_4 | 3,4 | 61 | 0.3804 | -0.0160 |
| 3dim_plus_dims_5 | 3,4,5 | 71 | 0.3523 | -0.0442 |
| 3dim_plus_dims_6 | 3,4,5,6 | 81 | 0.3208 | -0.0756 |
| 3dim_plus_dims_7 | 3,4,5,6,7 | 91 | 0.3075 | -0.0890 |
| 3dim_plus_dims_8 | 3,4,5,6,7,8 | 101 | 0.2756 | -0.1209 |
| 3dim_plus_dims_9 | 3,4,5,6,7,8,9 | 111 | 0.3351 | -0.0614 |
| 3dim_plus_dims_10 | 3,4,5,6,7,8,9,10 | 121 | 0.3518 | -0.0447 |
| 3dim_plus_dims_11 | 3,4,5,6,7,8,9,10,11 | 131 | 0.3082 | -0.0882 |

## 可写入论文的结论

这组结果比单纯“3维 vs 12维提交分数”更细：单维测试显示 dim6/Mag X 等非加速度通道并非完全无信号；但在 3 维加速度基础上继续加入 dim3-dim11 时，LOO score 从 0.3965 下降到 0.3082，所有增量配置均低于 3维基线。

因此，正确结论不是“dim3-dim11 没有信号”，而是：在 N=88 的小样本下，额外通道带来的特征数增长和伪相关风险会压过其潜在信号，所以最终保留前 3 个加速度通道是更稳妥的低维选择。

建议英文表述：

```text
Although several non-accelerometer IMU dimensions showed non-trivial signal when evaluated individually with BigFive traits, incrementally adding these dimensions to the compact 3-axis acceleration representation consistently degraded leave-one-out performance. This supports the use of a compact 3-axis IMU representation under the small-sample regime, where additional channels increase the risk of spurious correlations and overfitting.
```

## 生成文件

- `single_imu_dimension_with_bigfive.csv`：单维度测试表。
- `incremental_addition_to_3dim.csv`：3维基础上增量加维表。
- `01_single_imu_dimension_importance.png`：单维度信号图。
- `02_incremental_addition_degrades_score.png`：增量加维性能下降图。
