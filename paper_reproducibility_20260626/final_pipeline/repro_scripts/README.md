# Reproduction Instructions

Run all commands from the project root:

```powershell
cd "D:\aaa大三下作业\语音信息处理\asr_prior_imu_key_models_20260506"
```

## Install Dependencies

```powershell
pip install -r repro_scripts\requirements.txt
```

## Fast Reproduction

This uses the provided cached prediction files in
`official_baseline/make_submission_forcodabench/`.

Required files in `official_baseline/make_submission_forcodabench/`:

```text
make_submission_sample.py
binary_sample.csv
ternary_sample.csv
_orig_phq.npy
_auvel50.npy
_e3only50.npy
_fatigue50.npy
_auspec50.npy
```

```powershell
$env:BLEND_PHQ_THRESHOLD='4.25'
$env:TERNARY_T2='11.0'
$env:OUT_NAME='young_final_t4p25_t2p11'
python repro_scripts\run_pipeline.py
```

The generated submission is:

```text
official_baseline/make_submission_forcodabench/young_final_t4p25_t2p11_validated/submission.zip
```

## Train/Recompute From Scratch

This recomputes the prediction files instead of using cached `.npy` files.

```powershell
$env:BLEND_PHQ_THRESHOLD='4.25'
$env:TERNARY_T2='11.0'
$env:OUT_NAME='young_final_t4p25_t2p11'
$env:SKIP_CACHED='1'
python repro_scripts\run_pipeline.py
```

Expected data layout:

```text
Train-MPDD-Young/Young/
Test-MPDD-Young/Young/
official_baseline/make_submission_forcodabench/
repro_scripts/
```
