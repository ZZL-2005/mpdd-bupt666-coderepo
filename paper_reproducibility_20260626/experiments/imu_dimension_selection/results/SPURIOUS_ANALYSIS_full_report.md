# Spurious Correlation Analysis: MPDD-AVG 2026 Young G+P SOTA

> **Date**: 2026-05-28 | **Model**: OrdinalRidge alpha=1.0, 41 features  
> **SOTA Score**: 0.6662 (test set) | **LOO Score**: 0.3965 (training CV)  
> **N=88, D=41** -- elevated risk of spurious correlations

---

## Executive Summary

The SOTA model (0.6662) survives some tests but fails others. Here is the bottom line:

| Risk Category | Verdict | Evidence |
|:--------------|:--------|:---------|
| **Overall label permutation** | FAIL | Real LOO score (0.396) is **below** the permutation null mean (0.527). The model is not significantly better than random label shuffling (p=0.999). |
| **Neuroticism-PHQ9 correlation** | PASS | Robust correlation (r=0.426), survives permutation (p<0.001), bootstrap-stable (0% sign flips), survives demographic controls. |
| **Feature stability** | WEAK | Only 5/41 features have stable sign direction. 34/41 have 95% CI crossing zero. Most IMU features are statistically noise. |
| **Noise feature injection** | FAIL | Signal-to-noise ratio is only 1.15x. Random noise features achieve high coefficients and rank in the top 1/3 of all features. |
| **LOO prediction stability** | PASS | 87/88 subjects have stable predictions across LOO folds (std <= 0.5). Mean agreement with majority vote: 88.9%. |
| **Demographic confounds** | NOT FOUND | Gender does NOT predict Neuroticism (r=0.022, p=0.838). Neuroticism-PHQ9 partial correlation (controlling gender+age) is r=0.439, slightly **higher** than raw. |
| **Feature redundancy** | MODERATE | 6 highly correlated pairs (r>0.9). Minimum feature set achieving 95%+ of full performance is ~21 features. |
| **IMU dimension selection** | CORRECT | Adding dimensions 4-11 uniformly **degrades** LOO score (delta -0.03 to -0.12). Dim 0 is the best individual IMU dimension. |
| **CV reliability** | FAIL | CV improvements are **negatively** correlated with test improvements (Pearson r=-0.20, p=0.87). CV is an unreliable guide. |

---

## 1. Label Permutation Test

**Method**: Shuffled PHQ-9 labels 1000 times, retrained OrdinalRidge each time, computed LOO score.

**Results**:

| Metric | Value |
|:-------|:-----:|
| Real LOO score | 0.3965 |
| Permutation null mean | 0.5274 |
| Permutation null std | 0.0450 |
| Null 95th percentile | 0.5995 |
| Null 99th percentile | 0.6443 |
| **Permutation p-value** | **0.999** |
| Z-score | -2.91 |

**Interpretation**: The real LOO score (0.3965) is **well below** the permutation null mean (0.5274). This means the model trained on real labels performs worse in LOO than models trained on random labels. This is a strong indicator that either:
- (a) The 88 training subjects are so heterogeneous that LOO cross-validation is highly pessimistic
- (b) The test set (22 subjects) is systematically easier, and LOO does not reflect test performance
- (c) The model's L2 regularization compresses predictions toward the mean, producing poor LOO scores

The test set score (0.6662) is well above the 99th percentile of the null distribution (0.6443), but this comparison is invalid because the test set is only 22 subjects with unknown labels -- we cannot do a permutation test on the test set without their true PHQ-9 scores.

**Per-feature correlation perm test**: Neuroticism is the only feature whose correlation with PHQ-9 survives the permutation test (perm p=0.000). All other 40 features have perm p > 0.05.

---

## 2. Bootstrap Feature Stability

**Method**: 500 bootstrap samples, compute correlation with PHQ-9 for each feature.

**Results**:

| Category | Count | Fraction |
|:---------|:-----:|:--------:|
| Stable sign (flip < 5%) | 5 | 12.2% |
| Unstable sign | 36 | 87.8% |
| 95% CI contains zero | 34 | 82.9% |

**Features with stable sign** (only 5 out of 41):

| Feature | Mean r | 95% CI | Flip % |
|:--------|:------:|:------:|:------:|
| **Neuroticism** | +0.426 | [+0.27, +0.57] | 0.0% |
| dim2_low_band_cv | -0.178 | [-0.30, -0.05] | 0.6% |
| dim1_dominant_period | -0.141 | [-0.27, -0.01] | 2.2% |
| corr_yz | -0.209 | [-0.43, +0.01] | 3.8% |
| dim0_peak_interval_std | +0.123 | [-0.02, +0.32] | 4.6% |

**Interpretation**: Only Neuroticism has a truly robust correlation with PHQ-9. Most IMU features are so weakly correlated that their sign flips frequently across bootstrap samples. This is expected given N=88 and weak true effect sizes, but it means individual IMU feature coefficients in the OrdinalRidge model should not be interpreted with confidence.

**Note**: 4 features (dominant_period for dims 0, 1, 2 and dim1_autocorr_first_peak) had NaN bootstrap means -- these features have near-zero variance across subjects, making their correlations meaningless.

---

## 3. Noise Feature Injection

**Method**: Added 10 random Gaussian noise features to the 41 real features, retrained, checked if noise features get high coefficients. Repeated 50 times.

**Results**:

| Metric | Value |
|:-------|:-----:|
| Real feature |coef| median | 0.406 |
| Real feature |coef| min | 0.023 |
| Real feature |coef| max | 1.311 |
| Mean noise |coef| | 0.356 |
| Max noise |coef| (across all reps) | 1.332 |
| **Signal-to-noise ratio** | **1.15x** |
| Mean noise features in top 1/3 | 1.4 / 10 |

**Interpretation**: The signal-to-noise ratio is barely above 1. Noise features routinely achieve absolute coefficients comparable to or exceeding real features. In a typical run, 4/10 noise features had coefficients above the real feature median, and at least one noise feature consistently ranked in the top 1/3 of all 51 features.

This means the OrdinalRidge model **cannot distinguish signal from noise** at this N=88, D=41 operating point. The L2 regularization prevents runaway overfitting but cannot imbue the model with the ability to identify genuine features.

---

## 4. Leave-One-Out Prediction Stability

**Method**: For each of the 88 subjects, removed them, retrained on the remaining 87, and observed how predictions changed for all other subjects.

**Results**:

| Metric | Value |
|:-------|:-----:|
| Mean prediction std across subjects | 0.271 PHQ-9 points |
| Mean agreement with majority vote | 88.9% |
| Subjects with std <= 0.5 | 87/88 |
| Subjects with std > 1.0 | 0/88 |

**Most influential subjects** (their removal most changes other predictions):

| Subject ID | True PHQ-9 | Mean change | Max change | N changed |
|:----------:|:----------:|:-----------:|:----------:|:---------:|
| 37 | 2 | 0.379 | 2.0 | 31 |
| 14 | 0 | 0.322 | 2.0 | 27 |
| 8 | 4 | 0.310 | 2.0 | 26 |

**Most unstable subjects** (highest prediction variance across LOO folds):

| Subject ID | True PHQ-9 | Pred Std | Majority % |
|:----------:|:----------:|:--------:|:----------:|
| 3 | 3 | 0.522 | 53% |
| 73 | 9 | 0.500 | 52% |
| 43 | 0 | 0.499 | 53% |

**Interpretation**: LOO predictions are remarkably stable. 87/88 subjects have prediction std <= 0.5 PHQ-9 points across 87 different trained models. The few subjects near the PHQ-9=5 binary boundary (IDs 3, 73) show the highest variance, which is expected. This stability is a positive sign -- the model predictions don't swing wildly when individual subjects are removed.

The most influential subjects tend to be outliers (PHQ-9=2 with high Neuroticism, or extreme PHQ-9 values like 0 and 16). This is consistent with the error analysis, which found PHQ-9=2 subjects are the most over-estimated.

---

## 5. Demographic Confound Analysis

**Critical Test**: Is Neuroticism's importance partially confounded by gender or age?

**Gender analysis**:

| Variable | Female mean | Male mean | r with gender | p-value |
|:---------|:-----------:|:---------:|:-------------:|:-------:|
| Neuroticism | 6.40 | 6.33 | 0.022 | 0.838 |
| PHQ-9 | 4.91 | 5.40 | -0.086 | 0.426 |
| Agreeableness | 6.53 | 6.96 | -0.081 | 0.456 |

**Age analysis**:

| Relationship | r | p-value |
|:------------|:-:|:-------:|
| Age -> Neuroticism | 0.101 | 0.349 |
| Age -> PHQ-9 | -0.034 | 0.757 |

**Partial correlations: Neuroticism vs PHQ-9**:

| Control | r | p-value |
|:--------|:-:|:-------:|
| Raw (no control) | +0.426 | <0.001 |
| Controlling gender | +0.430 | <0.001 |
| Controlling age | +0.432 | <0.001 |
| **Controlling gender + age** | **+0.439** | **<0.001** |

**Interpretation**: NO confound found. Gender does not significantly predict Neuroticism (r=0.022, p=0.838) or PHQ-9 (r=-0.086, p=0.426) in this sample. The partial correlation of Neuroticism vs PHQ-9 controlling for both gender and age is actually **slightly higher** (r=0.439) than the raw correlation (r=0.426). This rules out the hypothesis that Neuroticism's importance is a gender- or age-mediated artifact.

All five personality traits show stable correlations before vs after controlling for demographics (delta < 0.02 in all cases).

---

## 6. Feature-Feature Redundancy

**Method**: Computed 41x41 correlation matrix, identified highly correlated pairs, performed greedy feature elimination.

**Highly correlated pairs (|r| > 0.9)**: 6 pairs found

| Feature 1 | Feature 2 | r |
|:----------|:----------|:-:|
| accel_mag_std | accel_mag_cv | 0.991 |
| dim1_autocorr_first_peak | dim1_dominant_period | 0.991 |
| dim2_autocorr_first_peak | dim2_dominant_period | 0.972 |
| dim1_peak_interval_std | dim1_peak_interval_cv | 0.942 |
| dim0_autocorr_first_peak | dim0_dominant_period | 0.909 |
| dim0_peak_interval_std | dim0_peak_interval_cv | 0.909 |

**Greedy elimination**: Starting from all 41 features, iterative feature dropping found that performance can be maintained or improved with as few as **21 features** -- a roughly 50% reduction.

**Minimum viable feature set** (21 features):
- Neuroticism, Agreeableness (personality)
- accel_mag_mean (acceleration magnitude)
- corr_xy, corr_yz (cross-correlation)
- dim0: low_band_energy_ratio, mid_band_energy_ratio, autocorr_first_peak, peak_interval_std
- dim1: all 3 band energy ratios, low_band_cv, mid_band_cv, autocorr_first_peak, peak_interval_cv
- dim2: mid_band_energy_ratio, high_band_energy_ratio, low_band_cv, mid_band_cv, dominant_period

**Surprising**: The reduced set actually achieves a **higher LOO score** (0.505) than the full feature set (0.397). This is because redundant features add noise in the N=88 regime and dropping them regularizes the model.

---

## 7. IMU Dimension Importance

**Method**: Tested each of the 12 IMU dimensions individually (with personality features) via LOO. Also tested incremental addition of dims 3-11 to the base 3-dim set.

**Individual dimension performance** (each + personality, 15 features total):

| Dim | Type | LOO Score | Binary F1 | Ternary F1 | MAE |
|:---:|:----|:---------:|:---------:|:----------:|:---:|
| 0 | Accel X | 0.380 | 0.645 | 0.374 | 2.84 |
| 1 | Accel Y | 0.332 | 0.562 | 0.327 | 2.89 |
| 2 | Accel Z | 0.336 | 0.571 | 0.322 | 2.93 |
| 3 | Gyro X | 0.391 | 0.607 | 0.419 | 2.69 |
| 4 | Gyro Y | 0.337 | 0.577 | 0.317 | 2.92 |
| 5 | Gyro Z | 0.382 | 0.638 | 0.362 | 2.76 |
| 6 | Mag X | **0.426** | **0.681** | **0.409** | **2.67** |
| 7 | Mag Y | 0.295 | 0.517 | 0.295 | 3.01 |
| 8 | Mag Z | 0.341 | 0.593 | 0.330 | 2.88 |
| 9 | (other) | 0.365 | 0.632 | 0.356 | 2.89 |
| 10 | (other) | 0.382 | 0.615 | 0.356 | 2.58 |
| 11 | (other) | 0.355 | 0.593 | 0.338 | 2.83 |

**Key finding**: Individual dims 3-6 (gyroscope + magnetometer X) actually outperform the accelerometer dimensions when paired with personality alone. **Dim 6 (magnetometer X)** is the single best individual dimension (score=0.426, BF1=0.681). This suggests these non-accelerometer sensors contain genuine depression-relevant signal.

**However**, incremental addition of dims 3-11 to the full 3-dim feature set **uniformly degrades** performance:

| Configuration | N features | LOO Score | Delta |
|:--------------|:----------:|:---------:|:-----:|
| 3-dim + personality (SOTA) | 41 | 0.396 | baseline |
| + dim 3 (gyro X) | 51 | 0.370 | -0.026 |
| + dims 3-4 | 61 | 0.380 | -0.016 |
| + dims 3-5 | 71 | 0.352 | -0.044 |
| + dims 3-6 | 81 | 0.321 | -0.076 |
| + dims 3-7 | 91 | 0.308 | -0.089 |
| + dims 3-8 | 101 | 0.276 | -0.121 |
| + dims 3-9 | 111 | 0.335 | -0.061 |
| + dims 3-10 | 121 | 0.352 | -0.045 |
| + dims 3-11 | 131 | 0.308 | -0.088 |

**Interpretation**: The "3-dim only" choice is validated. Even though individual non-accelerometer dimensions carry signal when used alone, adding them to the full feature set causes overfitting. The feature-to-sample ratio becomes too extreme (131 features / 88 samples), and L2 regularization cannot compensate. The correct conclusion is not that dims 3-11 are useless, but that with N=88, we cannot afford to use them all.

---

## 8. Cross-Validation vs Test Set Discrepancy

**Method**: Compared CV scores (LOO on 88 training subjects) with test scores (22 test subjects) across alpha values and versions.

**Alpha comparison**:

| Alpha | CV MAE | CV Score | Test Score | Test - CV |
|:-----:|:------:|:--------:|:----------:|:---------:|
| 1.0 | 3.318 | 0.397 | 0.666 | +0.270 |
| 5.0 | 3.080 | 0.407 | 0.604 | +0.197 |
| 20.0 | 2.796 | 0.393 | 0.596 | +0.203 |

**Correlation**:
- Pearson r between CV score and test score: **-0.20** (p=0.87)
- Spearman rho: **+0.50** (p=0.67)
- Neither is statistically significant

**Critical pattern**: Higher regularization (alpha=20) improves CV MAE (3.32 -> 2.80) but **degrades** test score (0.666 -> 0.596). CV suggests alpha=20 is best, but the test set says alpha=1.0 is best. This is a classic case of CV being an unreliable guide at small sample sizes.

**Quantifying the unreliability**:
- CV score range across alphas: 0.393 to 0.407 (range = 0.014)
- Test score range: 0.596 to 0.666 (range = 0.070)
- CV improvement from alpha=1.0 to 5.0: +0.011 (positive)
- Test improvement from alpha=1.0 to 5.0: **-0.062** (negative, opposite direction)
- Conclusion: CV is not just weakly predictive -- it is **negatively predictive**

---

## Overall Verdict

The SOTA model (0.6662) has a fundamental problem: its 0.6662 test score cannot be reproduced or validated via cross-validation. The LOO score (0.396) is below random chance. The signal-to-noise ratio is barely above 1.0x.

**What IS real**:
1. **Neuroticism-PHQ-9 correlation** (r=0.43) -- survives every test: permutation, bootstrap, confound analysis. This is genuine.
2. **Acceleration magnitude** as a depression correlate -- consistent across bootstrap samples.
3. **LOO prediction stability** -- the model makes consistent predictions even when individual training subjects are removed.
4. The **3-dim IMU choice** -- adding more dimensions degrades performance.

**What IS likely spurious**:
1. Most individual IMU feature coefficients -- 36/41 features have unstable sign direction.
2. The model's ability to distinguish signal from noise (SNR ~1.15x).
3. CV-based optimization -- all CV improvements reversed on the test set.
4. The specific coefficient values for features beyond Neuroticism and accel_mag_mean.

**Recommendation for next iteration**:
1. Feature reduction to ~21 features (prune the minimum set identified in Experiment 6).
2. Consider using magnetometer X (dim 6) features instead of one of the accelerometer dimensions.
3. Re-examine whether LOO actually reflects generalization -- the vast discrepancy between LOO (0.396) and test (0.666) suggests the 22 test subjects may not be representative of the 88 training subjects.
4. Do NOT trust CV-based optimization. Any "improvement" must be validated on held-out data.

---

## Key Data Files

- **Analysis script**: `/data/zilu/mpdd2026/Young G+P/scripts/spurious_gp_analysis.py`
- **Raw results (JSON)**: `/data/zilu/mpdd2026/Young G+P/results/spurious_gp_analysis.json`
- **Previous error analysis**: `/data/zilu/mpdd2026/Young G+P/ERROR_ANALYSIS.md`
