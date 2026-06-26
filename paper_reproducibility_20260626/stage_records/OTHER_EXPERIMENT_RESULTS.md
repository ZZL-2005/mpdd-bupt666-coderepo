# Other experiment results

这个文件记录“有结果记录，但不一定有完整复现脚本”的实验。它们主要来自：

- `source_archives/项目完整技术文档(1).md`
- `C:/Users/ZZL/Desktop/report/make_submission_forcodabench.zip`

这些结果适合用于汇报“探索过程、消融、负面结果、工程迭代”，但论文中需要区分：

- clean package 可复现的最终主线；
- 历史 leaderboard 提交；
- 本地 CV / LOO 结果；
- 只有提交产物、缺少生成脚本的历史目录。

## 1. Leaderboard stage results

| stage | representative submission/result | Score | F1 | CCC | Kappa | note |
|---|---|---:|---:|---:|---:|---|
| early external/classmate result | early high score | 0.7477 | 0.7911 | 0.7380 | 0.7141 | 不是最终可复现主线，报告标注为早期/不稳 |
| stage 1 | initial reproduction/mix | 0.6615 | 0.7972 | 0.4825 | 0.7048 | 复现/混合起点 |
| ORIG check | IMU accel3 + BigFive OrdinalRidge | 0.6701 | 0.7972 | 0.5083 | 0.7048 | 对应 ORIG 主干 |
| stage 1 best | 25% blend | 0.6798 | 0.7972 | 0.5374 | 0.7048 | 早期融合最佳 |
| audio added | audio/opensmile blend | 0.6662 | 0.7972 | 0.4968 | 0.7048 | 低于 0.6798，负结果 |
| MLP blend | `mlp2_o65_i15_m20` | 0.6838 | 0.7972 | 0.5494 | 0.7048 | stage 1 later |
| face AU breakthrough | face AU variants | 0.6953 | 0.7972 | 0.5840 | 0.7048 | OpenFace AU 开始起作用 |
| face AU | `au_au43_o60_i10_m30` | 0.7057 | 0.7972 | 0.6151 | 0.7048 | AU/gaze 统计特征 |
| face AU | `au43s_o55_i10_m35` | 0.7063 | 0.7972 | 0.6170 | 0.7048 | stage 2 best |
| four-way | four-way face blend | 0.7077 | 0.7972 | 0.6212 | 0.7048 | 多路融合 |
| four-way | `4w_o55_i8_a18_e19` | 0.7084 | 0.7972 | 0.6232 | 0.7048 | 多路融合 |
| velocity | `vel3w` | 0.7145 | 0.7972 | 0.6414 | 0.7048 | 速度特征加入 |
| velocity | `vel3ws_o55_a25_e20` | 0.7153 | 0.7972 | 0.6438 | 0.7048 | 速度特征 |
| velocity best | `vel3ws_o55_a45_e0` | 0.7178 | 0.7972 | 0.6515 | 0.7048 | stage 3 best |
| variance calibration | first calibrated version | 0.7563 | 0.7972 | 0.7668 | 0.7048 | CCC 大幅上升 |
| variance calibration | intermediate | 0.7752 | 0.7972 | 0.8236 | 0.7048 | CCC 提升 |
| variance calibration | intermediate | 0.7824 | 0.7972 | 0.8453 | 0.7048 | CCC 提升 |
| variance calibration | intermediate | 0.7867 | 0.7972 | 0.8582 | 0.7048 | CCC 提升 |
| variance calibration | intermediate | 0.7908 | 0.7972 | 0.8706 | 0.7048 | CCC 提升 |
| best CCC frozen | calibrated best CCC | 0.7931 | 0.7972 | 0.8772 | 0.7048 | 回归头最佳 CCC 版本 |
| classification head | `young_and_rule` / classification rule | 0.8340-0.8347 | 0.8255-0.8361 | 0.8772 | 0.7887-0.8012 | 文档正文和台账记录略有差异，应写成约 0.834 |
| final | `young_final_t4p25_t2p11` | approx 0.87 | not retained | 0.8772 | not retained | 最终提交 |
| ASR V4 | `young_asr_v4_submission_clean` | 0.4578 | 0.5833 | 0.5084 | 0.2815 | ASR 独立探索，负/辅助结果 |

## 2. Local CV / LOO results

这些结果不是 hidden test leaderboard 分数，但可用于解释为什么选择或放弃某类方法。

| experiment | local result | interpretation |
|---|---:|---|
| ORIG LOO classification baseline | F1 0.6238, Kappa 0.1962 | 训练集上分类规则基线 |
| AND rule, t1=4.3 | F1 0.6347, Kappa 0.2343 | 双模态一致性规则有训练集依据 |
| AND rule, t1=4.2 | F1 0.6238, Kappa 0.2163 | 不如 4.3 |
| C1 event-internal segment trajectory | LOO CCC 0.2229, corr with auvel 0.414 | 有信号但未进最终版，可写 future work |
| C2 face Hjorth complexity | LOO CCC -0.0181, corr with auvel 0.367 | 明确负结果 |
| reference auvel LOO | LOO CCC 0.0217 | 单一路面部 ranker 的 LOO 很低，leaderboard/CV 方差大 |
| reference ORIG LOO | LOO CCC 0.1736 | IMU 主干本地参考 |
| RidgePHQ vs OrdinalRidge | RidgePHQ local CV 0.3804 vs OrdinalRidge approx 0.342 | RidgePHQ 本地更好，但 hidden test 不如 OrdinalRidge 稳 |
| opensmile + IMU + personality | local approx score 0.385 vs IMU+P 0.325 | 本地提升没有在 leaderboard 兑现 |
| ASR V4 train-side raw sum | PHQ MAE 0.966, label2 acc 0.875, label3 macro-F1 0.836 | 训练侧可解释强，但 hidden test 弱 |
| ASR V4 isotonic | PHQ MAE 1.077, label2 acc 0.855, label3 macro-F1 0.780 | 比 raw sum 差 |
| opensmile + GRU | binary F1 0.5856, ternary Kappa 0.0421 | 深度语音序列失败 |
| wav2vec2 + GRU | binary F1 0.5227, ternary Kappa 0.1758 | 深度语音序列失败 |
| BiLSTM raw IMU | binary F1 approx 0.59 | 端到端 IMU 时序失败 |

## 3. Ranker / feature exploration families

这些大多有历史目录或缓存预测，但并不都在 clean package 中保留完整生成脚本。

| family | examples | outcome |
|---|---|---|
| final face rankers | `auvel`, `e3only`, `fatigue`, `auspec` | 进入最终版 |
| event velocity variants | `e1vel`, `e2vel`, `vel3w`, `vel3ws`, `velvel`, `_e1vel50.npy`, `_e2vel50.npy`, `_velfat50.npy` | 部分有帮助，最终凝练成 auvel/fatigue 等 |
| richer AU statistics | `au_rich`, `_aurich50.npy` | 边际，没有进入最终 clean 主线 |
| AU depression subset | `audep`, `audepMLP` | RidgeCV 易收缩，MLP 才有 spread，未进最终 |
| AU/audio joint | `au_audio`, `auaud`, `auaudio` | 边际或不稳 |
| speech fatigue / e2speech | `_e2speech50.npy`, `_speech_fat50.npy` | 有临床动机，但没有稳定提升 |
| head pose | `_headpose50.npy`, `young_hp_*` | 实测没用或边际 |
| landmarks | `_lmk3d_fat50.npy` | 未进最终 |
| sub-feature ablation | `_sub_auint50.npy`, `_sub_aupres50.npy`, `_sub_gaze50.npy` | 可用于说明 AU intensity / presence / gaze 子集探索 |
| IMU body channels | `imubody` | 与 ORIG 有差异但增益小 |
| video embeddings | `vid`, `VDense`, `VRes`, DenseNet/ResNet PCA | 边际 |
| description embeddings | `desc`, `desc_emb`, 1024-d text embeddings PCA | 边际，未替代 BigFive |
| KNN / local structure | `knn`, `tt_*` | 融合候选，非主力 |

## 4. Model comparison families

| family | examples | conclusion |
|---|---|---|
| linear/nonlinear regressors | `young_ccc_ridge`, `young_ccc_svr`, `young_ccc_elasticnet`, `young_ccc_bayesian` | 单纯换回归器不是主要杠杆 |
| stacking | `young_ccc_stacking_blend10..50` | 有增益但复杂、未作为最终主线 |
| ordinal-continuous hybrids | `young_ccc_ordcont_blend10/30/50` | 边际 |
| LightGBM | `young_lgbm_*`, `young_lgbm4_*` | 小样本过拟合 |
| Gaussian Process | `young_gp_orig_std392` | 边际 |
| deep sequence models | `young_bilstm_full`, `young_hybrid_bilstm`, GRU scripts | 失败，F1 0.52-0.59 |
| direct classifiers | `young_cls_lr_*`, `young_cls_svm_*`, `young_cls_rf*`, `young_cls_gbt`, `young_cls_vote*` | 直接分类不稳定，最终仍用连续 PHQ + 规则 |

## 5. Threshold, calibration, and competition-engineering results

这些记录可以解释最终分数如何从 0.79 到 0.83/0.87，但论文中要诚实标注它们是 leaderboard feedback / threshold tuning，不是新的模型结构。

| family | examples | note |
|---|---|---|
| variance calibration sweeps | `young_calib_std30`, `young_calib_std32`, `young_calib_std349`, `young_fat_f10_std*` | CCC 提升关键 |
| threshold sweeps | `young_thresh_*`, `young_blendthr_*`, `young_tnt2_*`, `young_blend_t5_t2_*` | 调 binary / ternary 阈值 |
| AND / consistency rules | `young_and_rule`, `young_consistent_blendthresh` | 分类头提升 |
| sample flip/hybrid probes | `young_flip_*`, `young_hybrid_*`, `young_v2_*` | 多为反馈驱动探测，不宜包装成通用方法 |
| final threshold | `t1=4.25`, `t2=11.0` | leaderboard 选定；训练 LOO 较支持 t1 approx 4.3 |

## 6. What can be safely reported

推荐写进论文/汇报主线：

- ORIG: IMU accel3 + BigFive + OrdinalRidge, Score approx 0.6702.
- OpenFace AU/gaze 多路 ranker：AU/速度/事件三/疲劳/频谱带来 0.706 -> 0.718 的阶段提升。
- 方差校准：CCC 从 0.6515 到 0.8772，是最清楚的回归提升机制。
- AND 分类规则：分类指标从固定阈值的 F1 0.7972/Kappa 0.7048 提到约 F1 0.83/Kappa 0.79-0.80。
- ASR/audio 负结果：ASR V4 Score 0.4578；opensmile/audio 加入后 0.6662 < 0.6798；GRU F1 0.52-0.59。
- 小样本结论：端到端深度时序模型过拟合，手工时序特征 + 小模型 + 集成 + 校准更稳。

不建议当作强方法贡献写：

- 逐阈值/逐权重的 leaderboard tuning。
- `flip_*` 这类疑似逐样本反馈探测。
- 没有脚本的历史 submission 目录。
- 早期外部/同学高分结果，除非明确作为对照或背景。

