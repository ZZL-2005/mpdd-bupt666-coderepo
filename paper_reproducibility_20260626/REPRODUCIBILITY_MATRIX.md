# Reproducibility Matrix

This matrix links paper claims, method components, tables, and preserved
scores to concrete scripts and result files.

## Method Components

| Paper section | Component | Script(s) | Notes |
|---|---|---|---|
| 3.1 Feature Cutting | IMU channel pruning to accel3 | `experiments/imu_channel_loo_table/run_imu_channel_loo_accuracy.py`; `experiments/imu_dimension_selection/scripts/spurious_gp_analysis.py` | First script supports Table 1A; second supports the more detailed IMU dimension review. |
| 3.1 Feature Cutting | Big-Five parsing | `final_pipeline/repro_scripts/orig_model.py`; `experiments/orig_backbone/v2_OrdinalRidge_0.6662/generate_submission_ordinalridge.py` | Regex parsing from description text to five numeric traits. |
| 3.1 Feature Cutting | OpenFace AU/gaze semantic subset | `experiments/openface_pca_and_cutting/3_openface_semantic_subset/openface_pruned_analysis.py`; `experiments/openface_pca_and_cutting/4_openface_semantic_subset_complete/build_openface_subset_complete.py` | Supports AU/gaze subset discussion and Table 1B semantic subset rows. |
| 3.2 ORIG Backbone | IMU handcrafted features + StandardScaler + OrdinalRidge | `final_pipeline/repro_scripts/orig_model.py` | Final implementation used by the submitted pipeline. |
| 3.3 OpenFace Block | auvel | `final_pipeline/repro_scripts/build_auvel_ranker.py` | 387-d all-event mean/std/frame-change view. |
| 3.3 OpenFace Block | e3only | `final_pipeline/repro_scripts/build_e3only_ranker.py` | 129-d event-3 view. |
| 3.3 OpenFace Block | cross-event drift / fatigue | `final_pipeline/repro_scripts/build_fatigue.py` | 129-d event-mean-difference view. |
| 3.3 OpenFace Block | auspec | `final_pipeline/repro_scripts/build_auspec.py` | 645-d FFT spectral view. |
| 3.3 OpenFace Block | CCC-loss MLP, mixup, 50-seed averaging | `final_pipeline/repro_scripts/mlp_utils.py`; each OpenFace builder above | Shared training code. |
| 3.4 Fusion / Calibration / Classification | weighted blend, batch moment alignment, clipping, AND rule | `final_pipeline/repro_scripts/run_pipeline.py` | Produces the final submission zip. |

## Experiment Tables

| Paper table / result | Exact values in paper | Script(s) | Result file(s) | Evidence status |
|---|---|---|---|---|
| Table 1A, IMU 3/6/9/12 channels | 36d MAE 3.444; 78d MAE 5.563; 129d MAE 4.945; 189d MAE 4.765 | `experiments/imu_channel_loo_table/run_imu_channel_loo_accuracy.py` | `experiments/imu_channel_loo_table/loo_accuracy_summary.csv` | Directly supported. |
| Table 1B, raw OpenFace vs PCA5 | raw 710d acc 0.500; PCA5 acc 0.682 | `experiments/openface_pca_and_cutting/1_PCA_dimension_scan/dim_reduction_analysis.py`; `2_raw_vs_pca_benchmark/all_features_benchmark.py` | `dim_reduction_analysis.json`; `all_features_benchmark.json` | Directly supported. |
| Table 1B, OpenFace semantic subset | all semantic stats 479d acc 0.682; AU+gaze stats 399d acc 0.727 | `experiments/openface_pca_and_cutting/3_openface_semantic_subset/openface_pruned_analysis.py` | `openface_pruned_results.json` | Directly supported. |
| Table 2, ORIG backbone | Score 0.6701, F1 0.7972, CCC 0.5083, Kappa 0.7048 | `final_pipeline/repro_scripts/orig_model.py`; `experiments/orig_backbone/v2_OrdinalRidge_0.6662/generate_submission_ordinalridge.py` | `stage_records/FINAL_MODEL_STAGE_RESULTS_CN.md`; `experiments/orig_backbone/v2_OrdinalRidge_0.6662/RESULTS.md` | Historical leaderboard + reproducible backbone script. |
| Table 2, OpenFace AU/gaze | Score 0.7063, F1 0.7972, CCC 0.6170, Kappa 0.7048 | Final OpenFace builder scripts | `stage_records/FINAL_MODEL_STAGE_RESULTS_CN.md` | Historical leaderboard checkpoint; exact intermediate script not preserved. |
| Table 2, frame-change / multi-view | Score 0.7178, F1 0.7972, CCC 0.6515, Kappa 0.7048 | Final OpenFace builder scripts | `stage_records/FINAL_MODEL_STAGE_RESULTS_CN.md` | Historical leaderboard checkpoint; exact intermediate script not preserved. |
| Table 2, batch moment alignment | Score 0.7931, F1 0.7972, CCC 0.8772, Kappa 0.7048 | `final_pipeline/repro_scripts/run_pipeline.py` | `stage_records/FINAL_MODEL_STAGE_RESULTS_CN.md` | Historical checkpoint and final code support calibration logic. |
| Table 2, AND rule | Score 0.8347, F1 0.8361, CCC 0.8772, Kappa 0.8012 | `final_pipeline/repro_scripts/run_pipeline.py` | `stage_records/FINAL_MODEL_STAGE_RESULTS_CN.md` | Historical checkpoint and final code support rule logic. |
| Table 2, final thresholding | Score 0.8745, CCC 0.8772; F1/Kappa not retained | `final_pipeline/repro_scripts/run_pipeline.py` | `final_pipeline/official_baseline/make_submission_forcodabench/young_final_t4p25_t2p11_validated/submission.zip`; `stage_records/EXPERIMENT_EVIDENCE_AUDIT_CN.md` | Final submission preserved; final F1/Kappa unavailable locally. |

## Non-Final Routes

| Paper statement | Script(s) | Result file(s) | Notes |
|---|---|---|---|
| ASR route reached public Score 0.457751 | `experiments/asr_negative_route/scripts/extract_young_asr_phq9_evidence_features.py`; `run_young_strong_prior_router_9x3.py`; `build_young_asr_v4_test_submission.py` | `experiments/asr_negative_route/RESULTS.md`; `experiments/asr_negative_route/results/young_asr_v4_submission_clean_validated/submission.zip` | ASR transcript JSONL files are intentionally omitted from this public package. |
| Deep sequence / speech routes were not used in final ensemble | preserved summaries in `stage_records/OTHER_EXPERIMENT_RESULTS.md` and ASR package records | `stage_records/OTHER_EXPERIMENT_RESULTS.md` | Summary evidence only; not part of final system. |

## Raw Data Requirement

The scripts assume local access to the official MPDD data. The package does not
include raw audio, video, IMU, OpenFace frame files, labels, or participant
descriptions.

