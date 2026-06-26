# Paper Reproducibility Package

This directory maps the ACM paper
`Semantics-Guided Feature Cutting and Batch Calibration for Low-Resource
Multimodal Depression Assessment` to the scripts and preserved result files
available in this repository.

The package is a paper-level evidence index. It does not redistribute the raw
MPDD dataset. Scripts that recompute features from scratch expect the local
MPDD Young data layout described in the corresponding script README files.

## Directory Layout

```text
paper_reproducibility_20260626/
  README.md
  REPRODUCIBILITY_MATRIX.md
  final_pipeline/
    repro_scripts/
    official_baseline/make_submission_forcodabench/
  experiments/
    orig_backbone/
    imu_channel_loo_table/
    imu_dimension_selection/
    openface_pca_and_cutting/
    asr_negative_route/
  stage_records/
  paper_snapshot/
```

## What Is Covered

| Paper item | Primary script(s) | Result/evidence files | Status |
|---|---|---|---|
| Final submitted system, Score 0.8745 | `final_pipeline/repro_scripts/run_pipeline.py` | `final_pipeline/official_baseline/make_submission_forcodabench/young_final_t4p25_t2p11_validated/submission.zip` | Reproducible with cached predictions; from-scratch run requires MPDD data |
| ORIG backbone: IMU accel3 + BigFive + OrdinalRidge | `final_pipeline/repro_scripts/orig_model.py`; `experiments/orig_backbone/v2_OrdinalRidge_0.6662/generate_submission_ordinalridge.py` | `experiments/orig_backbone/v2_OrdinalRidge_0.6662/RESULTS.md` | Script and historical result preserved |
| OpenFace auvel/e3only/fatigue/auspec blocks | `final_pipeline/repro_scripts/build_auvel_ranker.py`, `build_e3only_ranker.py`, `build_fatigue.py`, `build_auspec.py`, `mlp_utils.py` | cached `_auvel50.npy`, `_e3only50.npy`, `_fatigue50.npy`, `_auspec50.npy` in `final_pipeline/official_baseline/...` | Script and cached prediction vectors preserved |
| Linear fusion, batch moment alignment, AND classification rule | `final_pipeline/repro_scripts/run_pipeline.py` | final validated submission zip and `stage_records/FINAL_MODEL_STAGE_RESULTS_CN.md` | Reproducible for final submission |
| Table 1A IMU 3/6/9/12-channel LOO MAE | `experiments/imu_channel_loo_table/run_imu_channel_loo_accuracy.py` | `experiments/imu_channel_loo_table/loo_accuracy_summary.csv` | Script and result table preserved |
| Table 1B OpenFace raw/PCA and semantic subset cutting | `experiments/openface_pca_and_cutting/1_PCA_dimension_scan/dim_reduction_analysis.py`; `3_openface_semantic_subset/openface_pruned_analysis.py` | `dim_reduction_analysis.json`; `openface_pruned_results.json` | Script and result JSON preserved |
| Table 2 final model progression | final pipeline scripts plus `stage_records/FINAL_MODEL_STAGE_RESULTS_CN.md` | `paper_snapshot/fig/finalmodel.tex`; `stage_records/EXPERIMENT_EVIDENCE_AUDIT_CN.md` | Historical public-leaderboard trajectory, not strict add-one ablation |
| ASR/audio/deep-sequence non-final exploration | `experiments/asr_negative_route/scripts/*.py` | `experiments/asr_negative_route/RESULTS.md`; `results/young_asr_v4_submission_clean_validated/submission.zip` | Negative/non-final route preserved; ASR transcripts are not redistributed here |
| Current paper source, references, and figures | `paper_snapshot/sigconf.tex`; `paper_snapshot/references.bib`; `paper_snapshot/fig/*` | `paper_snapshot/sigconf.pdf`; exported Figure 1 PDF/PNG | Snapshot for traceability |

## Fast Reproduction of the Final Submission

Run from this directory:

```powershell
cd paper_reproducibility_20260626\final_pipeline
$env:BLEND_PHQ_THRESHOLD='4.25'
$env:TERNARY_T2='11.0'
$env:OUT_NAME='young_final_t4p25_t2p11'
python repro_scripts\run_pipeline.py
```

This uses cached prediction vectors in:

```text
final_pipeline/official_baseline/make_submission_forcodabench/
```

To recompute the OpenFace and ORIG predictions from raw data, also set:

```powershell
$env:SKIP_CACHED='1'
```

That mode requires the local MPDD Young data directory.

## Evidence Boundaries

- The final consolidated pipeline is reproducible from scripts and cached
  prediction vectors.
- The final public leaderboard Score is `0.8745`; the separate final F1 and
  Kappa values were not retained locally.
- Table 2 is a preserved historical public-leaderboard trajectory. It should not
  be described as a strict leave-one-component-out ablation.
- The ASR route is included as a non-final exploration. The public package keeps
  scripts, metrics, and submission artifacts, but omits ASR transcript JSONL
  files because they may contain derived participant text.
- All full from-scratch runs require the official MPDD data, which is not
  redistributed in this package.
