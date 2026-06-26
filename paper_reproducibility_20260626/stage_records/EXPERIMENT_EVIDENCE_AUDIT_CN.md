# MPDD Young 论文实验依据核对记录

整理时间：2026-06-26

目的：区分哪些实验结果可以支撑论文主文，哪些只能作为辅助解释或负面探索。这里记录的是实际查看过的结果文件、脚本位置和实验设置。

## 1. 最终模型与阶段线

### 1.1 最终 clean pipeline

证据位置：

- `mpdd_repro_result_script_collection_20260626/01_final_openface_fusion/mpdd_young_repro_final_clean_20260602/repro_scripts/run_pipeline.py`
- `mpdd_repro_result_script_collection_20260626/01_final_openface_fusion/mpdd_young_repro_final_clean_20260602/repro_scripts/orig_model.py`
- `mpdd_repro_result_script_collection_20260626/01_final_openface_fusion/mpdd_young_repro_final_clean_20260602/repro_scripts/build_auvel_ranker.py`
- `mpdd_repro_result_script_collection_20260626/01_final_openface_fusion/mpdd_young_repro_final_clean_20260602/repro_scripts/build_e3only_ranker.py`
- `mpdd_repro_result_script_collection_20260626/01_final_openface_fusion/mpdd_young_repro_final_clean_20260602/repro_scripts/build_fatigue.py`
- `mpdd_repro_result_script_collection_20260626/01_final_openface_fusion/mpdd_young_repro_final_clean_20260602/repro_scripts/build_auspec.py`
- `mpdd_repro_result_script_collection_20260626/01_final_openface_fusion/mpdd_young_repro_final_clean_20260602/official_baseline/make_submission_forcodabench/young_final_t4p25_t2p11_validated/submission.zip`

核对到的设置：

- 融合：`0.40*ORIG + 0.30*auvel + 0.20*e3only + 0.10*fatigue + 0.05*auspec`
- 校准：`(raw - raw.mean()) * (3.92 / raw.std()) + 4.82`，然后 clip 到 `[0, 27]`
- 分类：`binary = (phq9 >= t1) AND (ORIG >= 4)`，最终环境变量使用 `t1=4.25, t2=11.0`
- 缓存：包内保留 `_orig_phq.npy`, `_auvel50.npy`, `_e3only50.npy`, `_fatigue50.npy`, `_auspec50.npy`
- 重算入口：`SKIP_CACHED=1 python repro_scripts/run_pipeline.py`

论文可用等级：强。最终 pipeline 和最终 submission 产物存在，方法可以复现。

注意：最终 Score 在论文中统一写作 `0.8745`。结果包同时保留最终 submission 产物与 CCC `0.8772`，后续论文表述不再使用模糊分数写法。

### 1.2 历史 leaderboard 阶段线

证据位置：

- `项目完整技术文档(1).md` 第 887--915 行
- `mpdd_repro_result_script_collection_20260626/OTHER_EXPERIMENT_RESULTS.md`
- `mpdd_repro_result_script_collection_20260626/FINAL_MODEL_STAGE_RESULTS_CN.md`

可核对阶段：

| 阶段 | Score | F1 | CCC | Kappa | 证据状态 |
|---|---:|---:|---:|---:|---|
| ORIG check | 0.6701 | 0.7972 | 0.5083 | 0.7048 | 历史 leaderboard + ORIG 脚本 |
| OpenFace AU/gaze | 0.7063 | 0.7972 | 0.6170 | 0.7048 | 历史 leaderboard |
| velocity / multi-ranker | 0.7178 | 0.7972 | 0.6515 | 0.7048 | 历史 leaderboard |
| variance calibration | 0.7931 | 0.7972 | 0.8772 | 0.7048 | 历史 leaderboard + final pipeline contains calibration |
| AND rule | 0.8340--0.8347 | 0.8255--0.8361 | 0.8772 | 0.7887--0.8012 | 历史 leaderboard + final pipeline contains rule |
| final thresholding | 0.8745 | not retained | 0.8772 | not retained | final submission exists; F1/Kappa not locally retained |

论文可用等级：中到强。可以作为 preserved cumulative checkpoints，但不能称为严格 add-one ablation，因为部分中间阶段没有独立 clean 脚本。

## 2. ORIG 主干

证据位置：

- `mpdd_repro_result_script_collection_20260626/01_final_openface_fusion/.../repro_scripts/orig_model.py`
- `mpdd_repro_result_script_collection_20260626/02_asr_prior_imu_key_models/.../young_imu_accel3_ordinalridge_validated/submission_report.json`
- `GP_实验结果汇总/Young_GP/v2_OrdinalRidge_0.6662/RESULTS.md`

核对到的设置：

- 数据：88 train, 22 test
- 特征：前 3 个 IMU 加速度通道手工特征 36 维 + BigFive 5 维，共 41 维
- 标准化：`StandardScaler`
- 模型：`mord.OrdinalRidge(alpha=1.0)`
- PHQ-9：训练集中 14 个 observed PHQ-9 值映射为 ordinal 类别，再映射回 PHQ-9

结果：

- 历史 ORIG check：Score 0.6701 / F1 0.7972 / CCC 0.5083 / Kappa 0.7048
- 早期 v2 记录：Score 0.6662 / F1 0.7972 / CCC 0.4966 / Kappa 0.7048

论文可用等级：强。ORIG 方法和结果都有脚本/报告支撑。

## 3. IMU feature cutting

### 3.1 IMU-only 3/6/9/12 通道 LOO

证据位置：

- `moreexp/imu_channel_seed_stability/run_imu_channel_loo_accuracy.py`
- `moreexp/imu_channel_seed_stability/loo_accuracy_summary.csv`
- `moreexp/imu_channel_seed_stability/loo_accuracy_metadata.json`
- `moreexp/imu_channel_seed_stability/LOO_ACCURACY_RESULTS_CN.md`

实验设置：

- 只用 IMU，不加 BigFive
- 分别使用前 3/6/9/12 个 IMU 通道
- ORIG-style hand-crafted IMU feature
- 模型：`LinearRegression`, `Ridge(alpha=1.0)`
- 验证：Leave-One-Out on 88 train subjects
- 标签：连续 PHQ-9；分类由预测 PHQ-9 阈值派生

关键结果：

| 模型 | 通道 | 特征维度 | clipped MAE | RMSE | CCC | challenge-like |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 3 | 36 | 3.6653 | 4.7173 | 0.0135 | 0.2049 |
| Linear | 6 | 78 | 9.4019 | 11.9829 | -0.0869 | 0.0739 |
| Linear | 9 | 129 | 5.9886 | 7.8178 | 0.0003 | 0.1722 |
| Linear | 12 | 189 | 4.8573 | 6.3432 | -0.0310 | 0.1525 |
| Ridge | 3 | 36 | 3.4440 | 4.5221 | -0.0093 | 0.1901 |
| Ridge | 6 | 78 | 5.5626 | 7.6813 | -0.1327 | 0.0619 |
| Ridge | 9 | 129 | 4.9451 | 6.3671 | 0.0148 | 0.1773 |
| Ridge | 12 | 189 | 4.7647 | 6.2290 | -0.0383 | 0.1367 |

论文可用等级：强，用来支持“IMU 通道裁剪有助于小样本泛化”。注意它是 IMU-only LOO，不是 CodaBench 分数。

### 3.2 带 BigFive 的单维/增量加维分析

证据位置：

- `Young_IMU维度选择/维度选择结果_experiment7.json`
- `Young_IMU维度选择/SPURIOUS_ANALYSIS_完整报告.md`
- `moreexp/young_imu_dimension_selection_review/incremental_addition_to_3dim.csv`
- `moreexp/young_imu_dimension_selection_review/single_imu_dimension_with_bigfive.csv`

实验设置：

- 每个单独 IMU dim 提 10 个统计特征，加 BigFive 5 维，总 15 维
- 增量实验以 3 个加速度通道 + BigFive 为 base，共 41 维；逐步加入 dim3--dim11，每个 dim 增加 10 维
- LOO on 88 train subjects

关键结果：

- 单维 dim6/Mag X 分数最高：LOO Score 0.4257
- 但从 3 维 base 加入 dim3--dim11 后，所有增量配置均低于 base：0.3965 降至最低 0.2756，最终 12 维为 0.3082

论文可用等级：中。适合解释“额外传感器并非完全无信号，但加入 full feature set 后在 N=88 下更容易引入伪相关/过拟合”。不宜作为最终性能主结果，因为同报告也指出 LOO 与 hidden-test 相关性弱。

## 4. OpenFace cutting / PCA

### 4.1 Raw vs PCA 维度扫描

证据位置：

- `高低维对比实验/1_PCA维度扫描_dim_reduction/dim_reduction_analysis.py`
- `高低维对比实验/1_PCA维度扫描_dim_reduction/dim_reduction_analysis.json`
- `高低维对比实验/2_全特征基准_raw_vs_pca/all_features_benchmark.py`
- `高低维对比实验/2_全特征基准_raw_vs_pca/all_features_benchmark.json`
- `moreexp/feature_pca_dimension_scan/feature_pca_dimension_scan_summary.csv`

实验设置：

- `StandardScaler + PCA + LogisticRegression`
- binary accuracy / macro-F1
- 使用 88 train labels 训练，并用脚本中硬编码的 22 test labels 评价

关键结果：

| 特征族 | Raw acc | 最佳 PCA | PCA acc | 说明 |
|---|---:|---|---:|---|
| OpenFace 710d | 0.5000 | PCA5/PCA15 | 0.6818 | raw 接近 chance，低维 PCA 有提升 |
| OpenSmile 65d | 0.5000 | PCA6/PCA8 | 0.8182 | 说明高维声学 raw 不稳 |
| IMU 36/39d | 0.5909 | PCA15/PCA20 | 0.6818 | 轻量 PCA 有提升 |
| Personality BigFive | 0.8636 | PCA3/PCA4 | 0.9091 | BigFive 本身信号强 |

论文可用等级：中。可以支持“高维 raw descriptor 容易失效，低维压缩/裁剪更稳定”的设计动机。但因为评价使用了 22 个 test true labels，论文中不要称为标准 challenge-blind validation。

### 4.2 OpenFace 语义子集

证据位置：

- `高低维对比实验/3_OpenFace子集裁剪/openface_pruned_analysis.py`
- `高低维对比实验/3_OpenFace子集裁剪/openface_pruned_results.json`
- `moreexp/openface_semantic_subset_complete/openface_feature_subset_results_complete.csv`

实验设置：

- OpenFace raw 710 列中使用：
  - gaze: cols 2--7，共 6 维
  - AU intensity: cols 675--691，共 17 维
  - AU presence: cols 692--709，共 18 维
- 每 event 统计：
  - AU intensity: mean/std/max/range/pct_active
  - AU presence: mean/std
  - gaze: mean/std
- 另外有 event 间 dynamic features
- 评价方式包括 direct Ridge, LogisticRegression, personality baseline residual
- 同样使用脚本中硬编码的 22 test labels 评价

关键结果：

| 子集 | 维度 | Direct Ridge bin acc | Residual bin acc | Residual ternary F1 |
|---|---:|---:|---:|---:|
| All_features | 479 | 0.6818 | 0.7273 | 0.5250 |
| AU_plus_Gaze | 399 | 0.7273 | 0.7273 | 0.5807 |
| AU_all_stats | 363 | 0.6818 | 0.7273 | 0.6500 |
| Gaze_only | 36 | 0.5909 | 0.8182 | 0.5505 |

论文可用等级：中。可以支持“最终围绕 AU/gaze 而不是完整 OpenFace 表征构造面部模块”。不能证明四个最终 ranker 分别有效。

## 5. OpenFace 四路 ranker

证据位置：

- `build_auvel_ranker.py`: mean/std/abs diff over 3 events, 387d
- `build_e3only_ranker.py`: event 3 only, 129d
- `build_fatigue.py`: E2-E1, E3-E2, E3-E1 mean drift, 129d
- `build_auspec.py`: FFT band ratios + entropy + dominant frequency, 645d
- `mlp_utils.py`: 43 KEY_COLS, MLP 96->48, dropout 0.3, AdamW, 400 epochs, mixup 2000 samples/epoch, Beta(0.4,0.4), 50 seeds

论文可用等级：方法强、单路消融弱。脚本完整，但没有保留统一条件下的 `ORIG + auvel + ...` 逐项 add-one leaderboard 表。因此论文应写“四路作为最终 pipeline 组件被复现”，不要写“每一路都有严格独立消融提升”。

## 6. 方差校准与分类规则

### 6.1 方差校准

证据位置：

- `项目完整技术文档(1).md` 第 493--508 行
- `run_pipeline.py` 中 `calibrate()`

关键结果：

- calibration 前代表点：Score 0.7178 / CCC 0.6515
- calibration 首版：Score 0.7563 / CCC 0.7668
- best calibrated CCC：Score 0.7931 / CCC 0.8772

论文可用等级：强。机制清楚，脚本存在，阶段结果也清楚。

### 6.2 AND classification rule

证据位置：

- `项目完整技术文档(1).md` 第 744--766 行
- `run_pipeline.py` 中 `classify()`

关键结果：

- leaderboard: Score 0.8340--0.8347, F1 0.8255--0.8361, CCC 0.8772, Kappa 0.7887--0.8012
- train LOO: ORIG baseline F1/Kappa 0.6238/0.1962；AND t1=4.3 为 0.6347/0.2343

论文可用等级：中到强。规则有脚本和训练 LOO 支撑，但最终阈值 `t1=4.25, t2=11` 是 leaderboard feedback 选定，应如实写。

## 7. 不建议放主表的内容

- Broad model comparison: 文档列了 Ridge/SVR/ElasticNet/BayesianRidge/LightGBM/GP/BiLSTM/GRU 等，但没有找到统一可复现的数值表和脚本落盘。适合一句话写 negative exploration，不适合主文表格。
- ASR/audio: ASR V4 有提交报告，Score 0.4578；audio 加入后也有下降记录。由于 final 不使用语音，建议只在 Discussion/negative exploration 中一句话提及，主文不展开。
- OpenFace/PCA 的 22-label local analysis: 有价值，但因为使用 hard-coded test true labels，论文里要降级为 design evidence，不要称为 official validation。

## 8. 对当前论文实验部分的建议

1. Table 2 最好恢复 F1/CCC/Kappa 列。现在只放 Score 和 delta，会显得证据弱；真正强的变化是 CCC 从 0.6515 到 0.8772、F1/Kappa 从 0.7972/0.7048 提升到 0.8255--0.8361 / 0.7887--0.8012。
2. Table 1 可以保留两个 panel，但 caption 应明确：
   - IMU panel 是 train-only LOO；
   - OpenFace panel 是 local/post-hoc design analysis，不是 official CodaBench。
3. `0.8745` 作为最终提交 Score 写入论文；同时在可复现材料清单中保留 final submission zip、`run_pipeline.py` 和阶段结果文档作为来源索引。
4. 不要在主文声称“每个 face ranker 都有独立消融”。可以写“cumulative checkpoints show gains from introducing AU/gaze, velocity/multi-ranker features, calibration, and classification rule.”
