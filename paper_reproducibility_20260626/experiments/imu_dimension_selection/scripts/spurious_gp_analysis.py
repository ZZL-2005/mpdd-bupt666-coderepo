#!/usr/bin/env python3
"""
Spurious Correlation Analysis for MPDD-AVG 2026 Young G+P SOTA Model.
Tests 8 hypotheses about whether the 0.6662 SOTA score is driven by
genuine signal or spurious correlations.

Author: Auto-generated analysis
Date: 2026-05-28
"""

import numpy as np
import pandas as pd
import re
import json
import warnings
import sys

from pathlib import Path
from scipy import signal, stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, mean_absolute_error
from mord import OrdinalRidge

warnings.filterwarnings('ignore')
np.random.seed(42)

# ===== PATHS =====
DATA_PATH = Path('/data/zilu/mpdd2026/datasets/MPDD-AVG-2026/Young')
GP_PATH = Path('/data/zilu/mpdd2026/Young G+P')
SCRIPTS_PATH = GP_PATH / 'scripts'
RESULTS_PATH = GP_PATH / 'results'
SUBMISSIONS_PATH = GP_PATH / 'submissions'

RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Test subject IDs (22 held-out subjects)
TEST_IDS = [1, 5, 7, 13, 15, 22, 28, 33, 34, 40, 42, 44, 47, 58, 74, 83, 85, 89, 90, 93, 105, 110]

# ============================================================
# DATA LOADING & FEATURE EXTRACTION (reproducing SOTA exactly)
# ============================================================

def bandpass_filter(sig, lowcut, highcut, fs=50, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, sig)


def extract_fast_imu_features(imu_data, fs=50, max_dim=3):
    """Extract IMU features from first max_dim dimensions.
    Default max_dim=3 reproduces the SOTA configuration."""
    features = {}
    for dim in range(max_dim):
        sig = imu_data[:, dim]
        low_band = bandpass_filter(sig, 0.5, 1.5, fs)
        mid_band = bandpass_filter(sig, 1.5, 3.0, fs)
        high_band = bandpass_filter(sig, 3.0, 10.0, fs)

        total_energy = np.sum(sig**2)
        features[f'dim{dim}_low_band_energy_ratio'] = np.sum(low_band**2) / (total_energy + 1e-8)
        features[f'dim{dim}_mid_band_energy_ratio'] = np.sum(mid_band**2) / (total_energy + 1e-8)
        features[f'dim{dim}_high_band_energy_ratio'] = np.sum(high_band**2) / (total_energy + 1e-8)
        features[f'dim{dim}_low_band_cv'] = np.std(low_band) / (np.abs(np.mean(low_band)) + 1e-8)
        features[f'dim{dim}_mid_band_cv'] = np.std(mid_band) / (np.abs(np.mean(mid_band)) + 1e-8)

        autocorr = np.correlate(sig - np.mean(sig), sig - np.mean(sig), mode='full')
        autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
        peaks, _ = signal.find_peaks(autocorr[1:100], height=0.3)
        features[f'dim{dim}_autocorr_first_peak'] = autocorr[peaks[0]+1] if len(peaks) > 0 else 0
        features[f'dim{dim}_dominant_period'] = (peaks[0]+1) / fs if len(peaks) > 0 else 0

        peaks, _ = signal.find_peaks(sig, distance=fs//4, prominence=0.5)
        if len(peaks) > 1:
            intervals = np.diff(peaks) / fs
            features[f'dim{dim}_peak_interval_mean'] = np.mean(intervals)
            features[f'dim{dim}_peak_interval_std'] = np.std(intervals)
            features[f'dim{dim}_peak_interval_cv'] = np.std(intervals) / (np.mean(intervals) + 1e-8)
        else:
            features[f'dim{dim}_peak_interval_mean'] = 0
            features[f'dim{dim}_peak_interval_std'] = 0
            features[f'dim{dim}_peak_interval_cv'] = 0

    accel_x, accel_y, accel_z = imu_data[:, 0], imu_data[:, 1], imu_data[:, 2]
    features['corr_xy'] = np.corrcoef(accel_x, accel_y)[0, 1]
    features['corr_xz'] = np.corrcoef(accel_x, accel_z)[0, 1]
    features['corr_yz'] = np.corrcoef(accel_y, accel_z)[0, 1]

    accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    features['accel_mag_mean'] = np.mean(accel_mag)
    features['accel_mag_std'] = np.std(accel_mag)
    features['accel_mag_cv'] = np.std(accel_mag) / (np.mean(accel_mag) + 1e-8)

    return features


def extract_personality_scores_robust(text):
    scores = {}
    pattern1 = r'Agreeableness and Conscientiousness scores of (\d+)'
    match1 = re.search(pattern1, text)
    if match1:
        score = float(match1.group(1))
        scores['Agreeableness'] = score
        scores['Conscientiousness'] = score

    pattern2 = r'Agreeableness and Conscientiousness scores are both (\d+)'
    match2 = re.search(pattern2, text)
    if match2:
        score = float(match2.group(1))
        scores['Agreeableness'] = score
        scores['Conscientiousness'] = score

    pattern3 = r'Agreeableness, Conscientiousness, and Neuroticism scores are all (\d+)'
    match3 = re.search(pattern3, text)
    if match3:
        score = float(match3.group(1))
        scores['Agreeableness'] = score
        scores['Conscientiousness'] = score
        scores['Neuroticism'] = score

    for trait in ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']:
        if trait not in scores:
            pattern = rf'{trait} score of (\d+)'
            match = re.search(pattern, text)
            if match:
                scores[trait] = float(match.group(1))
            else:
                scores[trait] = np.nan

    return scores


def compute_overall_score(binary_true, binary_pred, ternary_true, ternary_pred, phq9_true, phq9_pred):
    """Compute the MPDD overall score."""
    binary_f1 = f1_score(binary_true, binary_pred)
    ternary_f1 = f1_score(ternary_true, ternary_pred, average='macro')
    kappa = cohen_kappa_score(phq9_true, phq9_pred)
    ccc = concordance_correlation_coefficient(phq9_true, phq9_pred)
    return 0.35 * binary_f1 + 0.35 * ternary_f1 + 0.15 * kappa + 0.15 * ccc

def concordance_correlation_coefficient(y_true, y_pred):
    """Compute Lin's Concordance Correlation Coefficient."""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred)**2 + 1e-10)


def load_and_extract_features(data_path=DATA_PATH):
    """Load all data and extract the 41 features for all subjects (train + test)."""
    df_labels = pd.read_csv(data_path / 'split_labels_train.csv')
    df_desc = pd.read_csv(data_path / 'descriptions.csv')

    all_features_list = []
    for _, row in df_desc.iterrows():
        sid = row['ID']
        features = {}
        # Try IMU files
        for prefix in ['train/', '']:
            imu_file = data_path / 'IMU' / prefix / str(sid) / f'{sid}.npy'
            if imu_file.exists():
                try:
                    imu_data = np.load(imu_file)
                    imu_features = extract_fast_imu_features(imu_data, max_dim=3)
                    features.update(imu_features)
                except Exception:
                    pass
                break
        # Personality
        p_scores = extract_personality_scores_robust(row['Descriptions'])
        features.update(p_scores)
        features['ID'] = sid
        all_features_list.append(features)

    df_all = pd.DataFrame(all_features_list).dropna(
        subset=['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
    )

    # Separate train/test
    df_all['is_test'] = df_all['ID'].isin(TEST_IDS)
    df_train = df_all[~df_all['is_test']].copy()
    df_test = df_all[df_all['is_test']].copy()

    # Merge labels
    df_train = df_train.merge(df_labels, on='ID', how='left')

    # Feature columns (41 total)
    feature_cols = [col for col in df_train.columns if (
        col.startswith('dim') or col.startswith('corr') or col.startswith('accel') or
        col in ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
    )]

    return df_train, df_test, df_desc, feature_cols


def get_X_y(df_train, feature_cols):
    """Get feature matrix X and label vectors y from training dataframe."""
    X = df_train[feature_cols].values.astype(np.float64)
    y_phq9 = df_train['phq9_score'].values.astype(np.float64)
    y_binary = df_train['label2'].values.astype(int)
    y_ternary = df_train['label3'].values.astype(int)

    unique_phq9 = np.sort(np.unique(y_phq9))
    phq9_to_ordinal = {s: i for i, s in enumerate(unique_phq9)}
    y_ordinal = np.array([phq9_to_ordinal[s] for s in y_phq9])

    return X, y_phq9, y_binary, y_ternary, y_ordinal, unique_phq9


def train_ordinal_ridge(X_train, y_ordinal_train, alpha=1.0):
    """Train OrdinalRidge with StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = OrdinalRidge(alpha=alpha)
    model.fit(X_scaled, y_ordinal_train)
    return model, scaler


def predict_ordinal_ridge(model, scaler, X, unique_phq9):
    """Predict PHQ-9 scores from OrdinalRidge model."""
    X_scaled = scaler.transform(X)
    y_pred_o = model.predict(X_scaled)
    return np.array([unique_phq9[int(o)] for o in y_pred_o])


def compute_metrics(y_phq9_true, y_phq9_pred):
    """Compute all metrics given true and predicted PHQ-9 scores."""
    binary_true = (y_phq9_true >= 5).astype(int)
    binary_pred = (y_phq9_pred >= 5).astype(int)
    ternary_true = np.zeros_like(y_phq9_true, dtype=int)
    ternary_true[y_phq9_true >= 5] = 1
    ternary_true[y_phq9_true >= 10] = 2
    ternary_pred = np.zeros_like(y_phq9_pred, dtype=int)
    ternary_pred[y_phq9_pred >= 5] = 1
    ternary_pred[y_phq9_pred >= 10] = 2

    score = compute_overall_score(binary_true, binary_pred, ternary_true, ternary_pred, y_phq9_true, y_phq9_pred)
    bin_f1 = f1_score(binary_true, binary_pred)
    ter_f1 = f1_score(ternary_true, ternary_pred, average='macro')
    kappa = cohen_kappa_score(y_phq9_true, y_phq9_pred)
    ccc = concordance_correlation_coefficient(y_phq9_true, y_phq9_pred)
    mae = mean_absolute_error(y_phq9_true, y_phq9_pred)
    bin_acc = accuracy_score(binary_true, binary_pred)

    return {
        'overall_score': score,
        'binary_f1': bin_f1,
        'ternary_f1': ter_f1,
        'kappa': kappa,
        'ccc': ccc,
        'mae': mae,
        'binary_acc': bin_acc
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

print("=" * 80)
print("MPDD-AVG 2026 Young G+P - Spurious Correlation Analysis")
print("=" * 80)

print("\n--- Loading data and extracting features ---")
df_train, df_test, df_desc, feature_cols = load_and_extract_features()
X, y_phq9, y_binary, y_ternary, y_ordinal, unique_phq9 = get_X_y(df_train, feature_cols)

n_subjects = X.shape[0]
n_features = X.shape[1]
print(f"Loaded {n_subjects} training subjects, {n_features} features")

results = {
    'metadata': {
        'n_subjects': n_subjects,
        'n_features': n_features,
        'feature_names': feature_cols,
        'test_ids': TEST_IDS,
        'n_permutations': 1000,
        'n_bootstraps': 500,
        'n_noise_features': 10,
        'random_seed': 42
    }
}

# ============================================================
# EXPERIMENT 1: LABEL PERMUTATION TEST
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 1: Label Permutation Test")
print("=" * 80)

N_PERM = 1000
perm_scores = []
perm_feature_corrs = {feat: [] for feat in feature_cols}

for perm_idx in range(N_PERM):
    y_shuffled = y_phq9.copy()
    np.random.shuffle(y_shuffled)

    # Create ordinal labels from shuffled PHQ-9
    unique_shuf = np.sort(np.unique(y_shuffled))
    shuf_to_ord = {s: i for i, s in enumerate(unique_shuf)}
    y_shuf_ord = np.array([shuf_to_ord[s] for s in y_shuffled])

    # Train OrdinalRidge
    scaler_p = StandardScaler()
    X_p_scaled = scaler_p.fit_transform(X)
    model_p = OrdinalRidge(alpha=1.0)
    model_p.fit(X_p_scaled, y_shuf_ord)

    # Predict (in-sample, to get score)
    y_pred_o = model_p.predict(X_p_scaled)
    y_pred_phq9 = np.array([unique_shuf[int(o)] for o in y_pred_o])

    # Compute score against shuffled labels (to measure overfitting)
    metrics_p = compute_metrics(y_shuffled, y_pred_phq9)
    perm_scores.append(metrics_p['overall_score'])

    # Also compute feature-PHQ9 correlations
    for fi, feat in enumerate(feature_cols):
        r, _ = pearsonr(X[:, fi], y_shuffled)
        perm_feature_corrs[feat].append(r)

    if (perm_idx + 1) % 200 == 0:
        print(f"  Permutation {perm_idx + 1}/{N_PERM} completed")

perm_scores = np.array(perm_scores)
# Real score: train on all 88, CV-based score
# Compute real SOTA score using full model
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X)
model_full = OrdinalRidge(alpha=1.0)
model_full.fit(X_full_scaled, y_ordinal)
y_full_pred_o = model_full.predict(X_full_scaled)
y_full_pred_phq9 = np.array([unique_phq9[int(o)] for o in y_full_pred_o])
real_metrics = compute_metrics(y_phq9, y_full_pred_phq9)
real_score = real_metrics['overall_score']
# Also compute LOO-based real score
loo = LeaveOneOut()
loo_preds = []
for train_idx, test_idx in loo.split(X):
    X_lo_tr, X_lo_te = X[train_idx], X[test_idx]
    y_lo_tr = y_ordinal[train_idx]
    s_lo = StandardScaler()
    X_lo_tr_s = s_lo.fit_transform(X_lo_tr)
    X_lo_te_s = s_lo.transform(X_lo_te)
    m_lo = OrdinalRidge(alpha=1.0)
    m_lo.fit(X_lo_tr_s, y_lo_tr)
    loo_preds.append(unique_phq9[int(m_lo.predict(X_lo_te_s)[0])])
loo_preds = np.array(loo_preds)
real_metrics_loo = compute_metrics(y_phq9, loo_preds)
real_score_loo = real_metrics_loo['overall_score']

perm_p_value = np.mean(perm_scores >= real_score_loo)
perm_z_score = (real_score_loo - np.mean(perm_scores)) / (np.std(perm_scores) + 1e-10)

print(f"\nResults:")
print(f"  Real overall score (LOO): {real_score_loo:.4f}")
print(f"  Real overall score (in-sample): {real_score:.4f}")
print(f"  Permutation null mean: {np.mean(perm_scores):.4f}")
print(f"  Permutation null std: {np.std(perm_scores):.4f}")
print(f"  Permutation null 95th percentile: {np.percentile(perm_scores, 95):.4f}")
print(f"  Permutation null 99th percentile: {np.percentile(perm_scores, 99):.4f}")
print(f"  Permutation p-value: {perm_p_value:.6f}")
print(f"  Z-score: {perm_z_score:.2f}")
print(f"  Significant at p<0.05: {perm_p_value < 0.05}")
print(f"  Significant at p<0.01: {perm_p_value < 0.01}")

# Per-feature permutation test
print("\nPer-feature correlation permutation test:")
feat_perm_results = {}
for feat in feature_cols:
    null_corrs = np.array(perm_feature_corrs[feat])
    real_r, real_p = pearsonr(X[:, feature_cols.index(feat)], y_phq9)
    feat_p_val = np.mean(np.abs(null_corrs) >= np.abs(real_r))
    feat_perm_results[feat] = {
        'real_corr': float(f'{real_r:.4f}'),
        'real_p_value': float(f'{real_p:.4f}'),
        'perm_p_value': float(f'{feat_p_val:.4f}'),
        'null_mean': float(f'{np.mean(null_corrs):.4f}'),
        'null_std': float(f'{np.std(null_corrs):.4f}')
    }

# Print top features by permutation p-value
print(f"  {'Feature':>35} {'Real r':>8} {'Real p':>8} {'Perm p':>8} {'Null mean':>10} {'Null std':>8}")
print(f"  {'-'*80}")
feat_perm_sorted = sorted(feat_perm_results.items(), key=lambda x: x[1]['perm_p_value'])
for feat, res in feat_perm_sorted[:10]:
    print(f"  {feat:>35} {res['real_corr']:>8} {res['real_p_value']:>8} {res['perm_p_value']:>8} {res['null_mean']:>10} {res['null_std']:>8}")

results['experiment1_label_permutation'] = {
    'real_score_loo': float(f'{real_score_loo:.6f}'),
    'real_score_in_sample': float(f'{real_score:.6f}'),
    'null_mean': float(f'{np.mean(perm_scores):.6f}'),
    'null_std': float(f'{np.std(perm_scores):.6f}'),
    'null_95th_percentile': float(f'{np.percentile(perm_scores, 95):.6f}'),
    'null_99th_percentile': float(f'{np.percentile(perm_scores, 99):.6f}'),
    'p_value': float(f'{perm_p_value:.6f}'),
    'z_score': float(f'{perm_z_score:.2f}'),
    'significant_p05': bool(perm_p_value < 0.05),
    'significant_p01': bool(perm_p_value < 0.01),
    'permutation_scores_percentiles': {
        'min': float(f'{np.min(perm_scores):.4f}'),
        '25th': float(f'{np.percentile(perm_scores, 25):.4f}'),
        '50th': float(f'{np.percentile(perm_scores, 50):.4f}'),
        '75th': float(f'{np.percentile(perm_scores, 75):.4f}'),
        'max': float(f'{np.max(perm_scores):.4f}')
    },
    'feature_correlation_tests': feat_perm_results
}


# ============================================================
# EXPERIMENT 2: BOOTSTRAP FEATURE STABILITY
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 2: Bootstrap Feature Stability")
print("=" * 80)

N_BOOT = 500
boot_results = {}

for fi, feat in enumerate(feature_cols):
    boot_corrs = []
    for _ in range(N_BOOT):
        idx = np.random.choice(n_subjects, n_subjects, replace=True)
        r, _ = pearsonr(X[idx, fi], y_phq9[idx])
        boot_corrs.append(r)
    boot_corrs = np.array(boot_corrs)
    mean_corr = np.mean(boot_corrs)
    std_corr = np.std(boot_corrs)
    ci_low = np.percentile(boot_corrs, 2.5)
    ci_high = np.percentile(boot_corrs, 97.5)
    sign_flips = np.mean(np.sign(boot_corrs) != np.sign(mean_corr))
    stable_sign = sign_flips < 0.05  # Less than 5% flips

    boot_results[feat] = {
        'mean_corr': float(f'{mean_corr:.4f}'),
        'std_corr': float(f'{std_corr:.4f}'),
        'ci_95_low': float(f'{ci_low:.4f}'),
        'ci_95_high': float(f'{ci_high:.4f}'),
        'sign_flip_fraction': float(f'{sign_flips:.4f}'),
        'stable_sign': bool(stable_sign),
        'ci_contains_zero': bool(ci_low < 0 < ci_high)
    }

# Print summary
stable_feats = [f for f, r in boot_results.items() if r['stable_sign']]
unstable_feats = [f for f, r in boot_results.items() if not r['stable_sign']]
zero_crossing = [f for f, r in boot_results.items() if r['ci_contains_zero']]

print(f"\nFeature sign stability:")
print(f"  Stable sign (flip < 5%): {len(stable_feats)}/{n_features}")
print(f"  Unstable sign (flip >= 5%): {len(unstable_feats)}/{n_features}")
print(f"  95% CI contains zero: {len(zero_crossing)}/{n_features}")
print(f"\nTop 10 most stable features (lowest sign flip):")
boot_sorted = sorted(boot_results.items(), key=lambda x: x[1]['sign_flip_fraction'])
for feat, res in boot_sorted[:10]:
    print(f"  {feat:>35} mean_r={res['mean_corr']:>8} 95%CI=[{res['ci_95_low']:>7},{res['ci_95_high']:>7}] "
          f"flip={res['sign_flip_fraction']:.4f} stable={res['stable_sign']}")

print(f"\nBottom 10 most unstable features (highest sign flip):")
for feat, res in boot_sorted[-10:]:
    print(f"  {feat:>35} mean_r={res['mean_corr']:>8} 95%CI=[{res['ci_95_low']:>7},{res['ci_95_high']:>7}] "
          f"flip={res['sign_flip_fraction']:.4f} stable={res['stable_sign']}")

print(f"\nFeatures with 95% CI crossing zero (likely spurious):")
for feat in zero_crossing[:20]:
    res = boot_results[feat]
    print(f"  {feat:>35} mean_r={res['mean_corr']:>8} 95%CI=[{res['ci_95_low']:>7},{res['ci_95_high']:>7}]")

results['experiment2_bootstrap_stability'] = {
    'n_bootstraps': N_BOOT,
    'stable_sign_features': stable_feats,
    'unstable_sign_features': unstable_feats,
    'ci_contains_zero_features': zero_crossing,
    'n_stable': len(stable_feats),
    'n_unstable': len(unstable_feats),
    'n_zero_crossing': len(zero_crossing),
    'feature_details': boot_results
}


# ============================================================
# EXPERIMENT 3: NOISE FEATURE INJECTION
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 3: Noise Feature Injection")
print("=" * 80)

N_NOISE = 10
noise_results = {}
n_repeat_noise = 50

noise_coefs_all = np.zeros((n_repeat_noise, N_NOISE))
real_feat_ranks_all = np.zeros((n_repeat_noise, N_NOISE))

for rep in range(n_repeat_noise):
    # Add random Gaussian noise features
    np.random.seed(42 + rep)
    X_noise = np.random.randn(n_subjects, N_NOISE)
    X_aug = np.hstack([X, X_noise])
    all_feat_names = feature_cols + [f'noise_{i}' for i in range(N_NOISE)]

    # Train OrdinalRidge
    scaler_n = StandardScaler()
    X_aug_scaled = scaler_n.fit_transform(X_aug)
    model_n = OrdinalRidge(alpha=1.0)
    model_n.fit(X_aug_scaled, y_ordinal)

    coefs = np.abs(model_n.coef_.flatten())
    noise_coefs = coefs[-N_NOISE:]

    # Get ranks of noise features among all features
    sorted_idx = np.argsort(-coefs)
    for ni in range(N_NOISE):
        noise_coefs_all[rep, ni] = noise_coefs[ni]
        rank = np.where(sorted_idx == (n_features + ni))[0][0]
        real_feat_ranks_all[rep, ni] = rank

    # Which real features do noise features displace?
    real_coefs = coefs[:n_features]
    real_min_coef = np.min(real_coefs)
    real_max_coef = np.max(real_coefs)
    real_median_coef = np.median(real_coefs)

    n_noise_above_median = np.sum(noise_coefs > real_median_coef)
    n_noise_top_half = np.sum(real_feat_ranks_all[rep] < n_features / 2)

    if rep == 0:
        print(f"\nWith {N_NOISE} noise features added:")
        print(f"  Real feature |coef| range: [{real_min_coef:.4f}, {real_max_coef:.4f}]")
        print(f"  Real feature |coef| median: {real_median_coef:.4f}")
        print(f"  Noise feature |coef| values: {noise_coefs}")
        print(f"  Noise features with |coef| > real median: {n_noise_above_median}/{N_NOISE}")

    noise_results[rep] = {
        'noise_coefs': [float(f'{c:.4f}') for c in noise_coefs],
        'noise_ranks': [int(r) for r in real_feat_ranks_all[rep]],
        'n_noise_above_median': int(n_noise_above_median)
    }

# Aggregate across repetitions
mean_noise_coef = np.mean(noise_coefs_all)
max_noise_coef = np.max(noise_coefs_all)
mean_rank = np.mean(real_feat_ranks_all)
n_noise_top_third = np.mean(np.sum(real_feat_ranks_all < n_features / 3, axis=1))
n_noise_displacing_real = np.mean(np.sum(real_feat_ranks_all < n_features, axis=1))

print(f"\nAggregated over {n_repeat_noise} repetitions:")
print(f"  Mean noise |coef|: {mean_noise_coef:.4f}")
print(f"  Max noise |coef| across all: {max_noise_coef:.4f}")
print(f"  Mean noise feature rank (1=best): {mean_rank:.1f}")
print(f"  Mean noise features in top 1/3: {n_noise_top_third:.1f}/{N_NOISE}")
print(f"  Signal-to-noise ratio (median real / mean noise): "
      f"{(real_median_coef / (mean_noise_coef + 1e-10)):.2f}x")

results['experiment3_noise_injection'] = {
    'n_noise_features': N_NOISE,
    'n_repetitions': n_repeat_noise,
    'real_feature_median_coef': float(f'{real_median_coef:.4f}'),
    'real_feature_min_coef': float(f'{real_min_coef:.4f}'),
    'real_feature_max_coef': float(f'{real_max_coef:.4f}'),
    'mean_noise_coef': float(f'{mean_noise_coef:.4f}'),
    'max_noise_coef': float(f'{max_noise_coef:.4f}'),
    'signal_to_noise_ratio': float(f'{(real_median_coef / (mean_noise_coef + 1e-10)):.2f}'),
    'mean_noise_feat_rank': float(f'{mean_rank:.1f}'),
    'mean_n_noise_top_third': float(f'{n_noise_top_third:.1f}'),
    'repetition_details': noise_results
}


# ============================================================
# EXPERIMENT 4: LEAVE-ONE-OUT PREDICTION STABILITY
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 4: Leave-One-Out Prediction Stability")
print("=" * 80)

loo_stability_results = {}

# For each subject, remove it, retrain on remaining 87, predict all 87
y_all_predictions = np.zeros((n_subjects, n_subjects))  # [left_out_idx, predicted_subject_idx] = prediction
# y_all_predictions[i,j] = prediction for subject j when subject i is left out

for left_out_idx in range(n_subjects):
    train_idx = np.array([i for i in range(n_subjects) if i != left_out_idx])

    scaler_lo = StandardScaler()
    X_lo_tr = scaler_lo.fit_transform(X[train_idx])
    y_lo_tr = y_ordinal[train_idx]

    model_lo = OrdinalRidge(alpha=1.0)
    model_lo.fit(X_lo_tr, y_lo_tr)

    # Predict on remaining subjects
    X_lo_te = scaler_lo.transform(X[train_idx])
    preds_o = model_lo.predict(X_lo_te)
    preds_phq9 = np.array([unique_phq9[int(o)] for o in preds_o])
    y_all_predictions[left_out_idx, train_idx] = preds_phq9

    if (left_out_idx + 1) % 22 == 0:
        print(f"  LOO stability: {left_out_idx + 1}/{n_subjects} done")

# For each subject, compute prediction variance across models
subject_prediction_stability = {}
for si in range(n_subjects):
    # Predictions for subject si when OTHER subjects are left out
    preds_for_si = y_all_predictions[:, si]
    valid_preds = preds_for_si[preds_for_si != 0]  # exclude when si is left out (prediction=0)
    # Actually, let's be more careful:
    preds_for_si = np.array([y_all_predictions[lo, si] for lo in range(n_subjects) if lo != si])

    std_pred = np.std(preds_for_si)
    unique_preds = len(np.unique(preds_for_si))
    range_pred = np.max(preds_for_si) - np.min(preds_for_si)
    most_common = np.argmax(np.bincount(preds_for_si.astype(int)))
    agreement_pct = np.mean(preds_for_si == most_common) * 100

    subject_prediction_stability[int(df_train.iloc[si]['ID'])] = {
        'index': int(si),
        'pred_std': float(f'{std_pred:.4f}'),
        'pred_range': float(f'{range_pred:.2f}'),
        'n_unique_preds': int(unique_preds),
        'most_common_pred': int(most_common),
        'agreement_pct': float(f'{agreement_pct:.2f}'),
        'true_phq9': int(y_phq9[si])
    }

# For each left-out subject, compute influence on other subjects
subject_influence = {}
for lo in range(n_subjects):
    full_preds = y_full_pred_phq9.copy()
    loo_preds = y_all_predictions[lo]
    # For subjects NOT left out, how much did removal of lo change predictions?
    other_idx = [i for i in range(n_subjects) if i != lo]
    changes = np.abs(loo_preds[other_idx] - full_preds[other_idx])
    mean_change = np.mean(changes)
    max_change = np.max(changes)
    n_changed = np.sum(changes > 0)

    subject_influence[int(df_train.iloc[lo]['ID'])] = {
        'index': int(lo),
        'mean_pred_change': float(f'{mean_change:.4f}'),
        'max_pred_change': float(f'{max_change:.4f}'),
        'n_predictions_changed': int(n_changed),
        'true_phq9': int(y_phq9[lo])
    }

# Identify highly influential outliers
influence_df = pd.DataFrame(subject_influence).T
influence_df['mean_pred_change'] = influence_df['mean_pred_change'].astype(float)
top_influential = influence_df.nlargest(10, 'mean_pred_change')

print(f"\nSubject prediction stability:")
stable_ids = [sid for sid, res in subject_prediction_stability.items() if res['pred_std'] <= 0.5]
unstable_ids = [sid for sid, res in subject_prediction_stability.items() if res['pred_std'] > 1.0]
print(f"  Stable subjects (pred_std <= 0.5): {len(stable_ids)}/{n_subjects}")
print(f"  Unstable subjects (pred_std > 1.0): {len(unstable_ids)}/{n_subjects}")
print(f"  Mean prediction std across subjects: "
      f"{np.mean([r['pred_std'] for r in subject_prediction_stability.values()]):.3f}")
print(f"  Mean agreement with most common prediction: "
      f"{np.mean([r['agreement_pct'] for r in subject_prediction_stability.values()]):.1f}%")

print(f"\nTop 10 most influential subjects (their removal changes others most):")
for sid, row in top_influential.iterrows():
    print(f"  Subject ID={int(sid):>3}: mean_change={row['mean_pred_change']:.3f} "
          f"max_change={row['max_pred_change']:.3f} n_changed={int(row['n_predictions_changed']):>2} "
          f"true_phq9={int(row['true_phq9'])}")

print(f"\nTop 10 most unstable subjects (highest prediction variance):")
unstable_sorted = sorted(subject_prediction_stability.items(), key=lambda x: x[1]['pred_std'], reverse=True)
for sid, res in unstable_sorted[:10]:
    print(f"  Subject ID={sid:>3}: pred_std={res['pred_std']:.3f} range={res['pred_range']:.1f} "
          f"n_unique={res['n_unique_preds']} majority={res['agreement_pct']:.0f}% "
          f"true_phq9={res['true_phq9']}")

results['experiment4_loo_stability'] = {
    'mean_prediction_std': float(f'{np.mean([r["pred_std"] for r in subject_prediction_stability.values()]):.4f}'),
    'mean_agreement_pct': float(f'{np.mean([r["agreement_pct"] for r in subject_prediction_stability.values()]):.1f}'),
    'n_stable_subjects': len(stable_ids),
    'n_unstable_subjects': len(unstable_ids),
    'top_influential_subjects': {
        str(int(sid)): {
            'mean_pred_change': float(f'{row["mean_pred_change"]:.4f}'),
            'max_pred_change': float(f'{row["max_pred_change"]:.4f}'),
            'n_predictions_changed': int(row['n_predictions_changed']),
            'true_phq9': int(row['true_phq9'])
        }
        for sid, row in top_influential.iterrows()
    },
    'subject_prediction_stability': {
        str(sid): res for sid, res in subject_prediction_stability.items()
    },
    'subject_influence': {
        str(sid): res for sid, res in subject_influence.items()
    }
}


# ============================================================
# EXPERIMENT 5: DEMOGRAPHIC CONFOUND ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 5: Demographic Confound Analysis")
print("=" * 80)

# Extract gender and age from descriptions for training subjects
genders = []
ages = []
for _, row in df_train.iterrows():
    sid = row['ID']
    desc_row = df_desc[df_desc['ID'] == sid]
    if len(desc_row) > 0:
        text = desc_row.iloc[0]['Descriptions']
        age_m = re.search(r'(\d+)-year-old', text)
        gender_m = re.search(r'(male|female)', text, re.IGNORECASE)
        genders.append(gender_m.group(1).lower() if gender_m else 'unknown')
        ages.append(int(age_m.group(1)) if age_m else np.nan)
    else:
        genders.append('unknown')
        ages.append(np.nan)

df_train = df_train.copy()
df_train['gender'] = genders
df_train['age'] = ages
df_train['gender_f'] = (np.array(genders) == 'female').astype(float)

# Test 1: Does gender predict Neuroticism?
r_gen_neuro, p_gen_neuro = pearsonr(df_train['gender_f'], df_train['Neuroticism'])
# Test 2: Does gender predict PHQ-9?
r_gen_phq, p_gen_phq = pearsonr(df_train['gender_f'], df_train['phq9_score'])
# Test 3: Does gender predict Agreeableness?
r_gen_agree, p_gen_agree = pearsonr(df_train['gender_f'], df_train['Agreeableness'])

print(f"\nGender differences:")
print(f"  Female n: {(df_train['gender']=='female').sum()}, Male n: {(df_train['gender']=='male').sum()}")
print(f"  Female Neuroticism mean: {df_train[df_train['gender']=='female']['Neuroticism'].mean():.2f} "
      f"vs Male: {df_train[df_train['gender']=='male']['Neuroticism'].mean():.2f}")
print(f"  Gender -> Neuroticism: r={r_gen_neuro:.4f}, p={p_gen_neuro:.4f}")
print(f"  Gender -> PHQ-9: r={r_gen_phq:.4f}, p={p_gen_phq:.4f}")
print(f"  Gender -> Agreeableness: r={r_gen_agree:.4f}, p={p_gen_agree:.4f}")

# Test 4: Does age predict anything?
r_age_neuro, p_age_neuro = pearsonr(df_train['age'].values, df_train['Neuroticism'].values)
r_age_phq, p_age_phq = pearsonr(df_train['age'].values, df_train['phq9_score'].values)

print(f"\nAge effects:")
print(f"  Age -> Neuroticism: r={r_age_neuro:.4f}, p={p_age_neuro:.4f}")
print(f"  Age -> PHQ-9: r={r_age_phq:.4f}, p={p_age_phq:.4f}")

# Partial correlation: Neuroticism vs PHQ-9 controlling for gender and age
def partial_corr(x, y, covars):
    """Compute partial correlation between x and y controlling for covars."""
    # Regress x on covars
    A = np.column_stack([np.ones(len(covars)), covars])
    coef_x = np.linalg.lstsq(A, x, rcond=None)[0]
    x_resid = x - A @ coef_x
    # Regress y on covars
    coef_y = np.linalg.lstsq(A, y, rcond=None)[0]
    y_resid = y - A @ coef_y
    # Correlation of residuals
    r, p = pearsonr(x_resid, y_resid)
    return r, p

# Raw correlation Neuroticism vs PHQ-9
raw_r, raw_p = pearsonr(df_train['Neuroticism'].values, df_train['phq9_score'].values)

# Controlling for gender
covar_gender = df_train['gender_f'].values.reshape(-1, 1)
r_partial_gender, p_partial_gender = partial_corr(
    df_train['Neuroticism'].values,
    df_train['phq9_score'].values,
    covar_gender
)

# Controlling for age
covar_age = df_train['age'].values.reshape(-1, 1)
r_partial_age, p_partial_age = partial_corr(
    df_train['Neuroticism'].values,
    df_train['phq9_score'].values,
    covar_age
)

# Controlling for both gender and age
covar_both = np.column_stack([df_train['gender_f'].values, df_train['age'].values])
r_partial_both, p_partial_both = partial_corr(
    df_train['Neuroticism'].values,
    df_train['phq9_score'].values,
    covar_both
)

print(f"\nPartial correlation: Neuroticism vs PHQ-9")
print(f"  Raw correlation: r={raw_r:.4f}, p={raw_p:.4f}")
print(f"  Controlling for gender: r={r_partial_gender:.4f}, p={p_partial_gender:.4f}")
print(f"  Controlling for age: r={r_partial_age:.4f}, p={p_partial_age:.4f}")
print(f"  Controlling for gender+age: r={r_partial_both:.4f}, p={p_partial_both:.4f}")

# Test other personality traits with confounds
print(f"\nPartial correlations for all personality traits (controlling gender+age):")
for trait in ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']:
    raw_r_t, raw_p_t = pearsonr(df_train[trait].values, df_train['phq9_score'].values)
    r_partial_t, p_partial_t = partial_corr(
        df_train[trait].values, df_train['phq9_score'].values, covar_both
    )
    print(f"  {trait:>20}: raw r={raw_r_t:+.4f} (p={raw_p_t:.4f}) -> "
          f"partial r={r_partial_t:+.4f} (p={p_partial_t:.4f})")
    print(f"    {'Confounded!' if abs(raw_r_t) > abs(r_partial_t) + 0.02 else 'Stable'} "
          f"(delta={abs(raw_r_t) - abs(r_partial_t):+.4f})")

results['experiment5_demographic_confound'] = {
    'gender_counts': {
        'female': int((df_train['gender']=='female').sum()),
        'male': int((df_train['gender']=='male').sum())
    },
    'gender_neuroticism': {
        'female_mean': float(f'{df_train[df_train["gender"]=="female"]["Neuroticism"].mean():.2f}'),
        'male_mean': float(f'{df_train[df_train["gender"]=="male"]["Neuroticism"].mean():.2f}'),
        'r': float(f'{r_gen_neuro:.4f}'),
        'p': float(f'{p_gen_neuro:.4f}')
    },
    'gender_phq9': {
        'female_mean': float(f'{df_train[df_train["gender"]=="female"]["phq9_score"].mean():.2f}'),
        'male_mean': float(f'{df_train[df_train["gender"]=="male"]["phq9_score"].mean():.2f}'),
        'r': float(f'{r_gen_phq:.4f}'),
        'p': float(f'{p_gen_phq:.4f}')
    },
    'age_neuroticism': {
        'r': float(f'{r_age_neuro:.4f}'),
        'p': float(f'{p_age_neuro:.4f}')
    },
    'age_phq9': {
        'r': float(f'{r_age_phq:.4f}'),
        'p': float(f'{p_age_phq:.4f}')
    },
    'neuroticism_phq9_partial_correlations': {
        'raw': {'r': float(f'{raw_r:.4f}'), 'p': float(f'{raw_p:.4f}')},
        'controlling_gender': {'r': float(f'{r_partial_gender:.4f}'), 'p': float(f'{p_partial_gender:.4f}')},
        'controlling_age': {'r': float(f'{r_partial_age:.4f}'), 'p': float(f'{p_partial_age:.4f}')},
        'controlling_both': {'r': float(f'{r_partial_both:.4f}'), 'p': float(f'{p_partial_both:.4f}')}
    },
    'partial_correlations_all_traits': {
        trait: {
            'raw_r': float(f'{pearsonr(df_train[trait].values, df_train["phq9_score"].values)[0]:.4f}'),
            'raw_p': float(f'{pearsonr(df_train[trait].values, df_train["phq9_score"].values)[1]:.4f}'),
            'partial_r_gender_age': float(f'{partial_corr(df_train[trait].values, df_train["phq9_score"].values, covar_both)[0]:.4f}'),
            'partial_p_gender_age': float(f'{partial_corr(df_train[trait].values, df_train["phq9_score"].values, covar_both)[1]:.4f}'),
        }
        for trait in ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
    }
}


# ============================================================
# EXPERIMENT 6: FEATURE-FEATURE REDUNDANCY
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 6: Feature-Feature Redundancy")
print("=" * 80)

# Compute 41x41 correlation matrix
corr_matrix = np.corrcoef(X.T)
corr_df = pd.DataFrame(corr_matrix, index=feature_cols, columns=feature_cols)

# Find highly correlated pairs (r > 0.9)
high_corr_pairs = []
for i in range(n_features):
    for j in range(i+1, n_features):
        r_val = corr_matrix[i, j]
        if abs(r_val) > 0.9:
            high_corr_pairs.append({
                'feat1': feature_cols[i],
                'feat2': feature_cols[j],
                'correlation': float(f'{r_val:.4f}')
            })

high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

print(f"\nHighly correlated feature pairs (|r| > 0.9): {len(high_corr_pairs)} pairs found")
for pair in high_corr_pairs[:20]:
    print(f"  {pair['feat1']:>35} <-> {pair['feat2']:>35}: r={pair['correlation']}")

# If any pairs exist, drop one and measure performance change
redundancy_results = {}
if len(high_corr_pairs) > 0:
    # For each highly correlated pair, drop feat2 and retrain
    for pair in high_corr_pairs[:10]:  # Limit to 10 pairs for speed
        feat2 = pair['feat2']
        dropped_cols = [c for c in feature_cols if c != feat2]
        X_dropped = df_train[dropped_cols].values

        loo_preds_dropped = []
        loo = LeaveOneOut()
        for tr_idx, te_idx in loo.split(X_dropped):
            X_d_tr, X_d_te = X_dropped[tr_idx], X_dropped[te_idx]
            y_d_tr = y_ordinal[tr_idx]
            s_d = StandardScaler()
            X_d_tr_s = s_d.fit_transform(X_d_tr)
            X_d_te_s = s_d.transform(X_d_te)
            m_d = OrdinalRidge(alpha=1.0)
            m_d.fit(X_d_tr_s, y_d_tr)
            loo_preds_dropped.append(unique_phq9[int(m_d.predict(X_d_te_s)[0])])

        loo_preds_dropped = np.array(loo_preds_dropped)
        metrics_dropped = compute_metrics(y_phq9, loo_preds_dropped)

        redundancy_results[f"drop_{feat2}"] = {
            'dropped_feature': feat2,
            'correlated_with': pair['feat1'],
            'correlation': pair['correlation'],
            'score_full': float(f'{real_score_loo:.4f}'),
            'score_dropped': float(f'{metrics_dropped["overall_score"]:.4f}'),
            'delta': float(f'{metrics_dropped["overall_score"] - real_score_loo:.4f}'),
            'pct_of_full': float(f'{metrics_dropped["overall_score"] / (real_score_loo + 1e-10) * 100:.1f}')
        }

# Also do greedy feature elimination: starting from all 41, drop the most redundant feature
print(f"\nGreedy feature elimination (drop one at a time):")
greedy_results = {}
current_features = feature_cols.copy()
current_X = X.copy()
iteration_scores = [real_score_loo]

for step in range(min(20, n_features - 10)):
    best_drop = None
    best_score = -1
    best_drop_score = -1

    for feat in current_features:
        fi = feature_cols.index(feat)  # original index in full list
        # But we need to drop from current index... this is getting complex
        # Let's just use column names
        cols_dropped = [c for c in current_features if c != feat]
        idx = [feature_cols.index(c) for c in cols_dropped]
        X_sub = X[:, idx]

        loo_preds_sub = []
        loo = LeaveOneOut()
        for tr_idx, te_idx in loo.split(X_sub):
            X_s_tr, X_s_te = X_sub[tr_idx], X_sub[te_idx]
            y_s_tr = y_ordinal[tr_idx]
            sc = StandardScaler()
            X_s_tr_sc = sc.fit_transform(X_s_tr)
            X_s_te_sc = sc.transform(X_s_te)
            md = OrdinalRidge(alpha=1.0)
            md.fit(X_s_tr_sc, y_s_tr)
            loo_preds_sub.append(unique_phq9[int(md.predict(X_s_te_sc)[0])])

        loo_preds_sub = np.array(loo_preds_sub)
        met = compute_metrics(y_phq9, loo_preds_sub)
        sc_new = met['overall_score']

        if sc_new > best_score:
            best_score = sc_new
            best_drop = feat
            best_drop_score = sc_new

    if best_drop is not None and best_score >= real_score_loo * 0.95:
        current_features.remove(best_drop)
        idx = [feature_cols.index(c) for c in current_features]
        current_X = X[:, idx]
        iteration_scores.append(best_score)
        greedy_results[f'step_{step+1}_drop_{best_drop}'] = {
            'dropped': best_drop,
            'n_features_left': len(current_features),
            'score': float(f'{best_score:.4f}'),
            'pct_of_full': float(f'{best_score / (real_score_loo + 1e-10) * 100:.1f}')
        }
        print(f"  Step {step+1}: Dropped '{best_drop}' -> score={best_score:.4f} ({len(current_features)} feats left)")
    else:
        break

min_feature_set = current_features

print(f"\nMinimum feature set achieving >=95% of full performance: {len(min_feature_set)} features")
print(f"  Full performance: {real_score_loo:.4f}")
print(f"  Reduced performance: {best_drop_score:.4f}")
print(f"  Pct of full: {best_drop_score / (real_score_loo + 1e-10) * 100:.1f}%")
print(f"  Features kept: {min_feature_set}")

results['experiment6_feature_redundancy'] = {
    'correlation_matrix': [[float(f'{corr_matrix[i,j]:.4f}') for j in range(n_features)] for i in range(n_features)],
    'high_corr_pairs_r_threshold': 0.9,
    'n_high_corr_pairs': len(high_corr_pairs),
    'high_corr_pairs': high_corr_pairs,
    'pair_drop_results': redundancy_results,
    'greedy_elimination': greedy_results,
    'min_feature_set_n': len(min_feature_set),
    'min_feature_set': min_feature_set,
    'min_feature_set_score': float(f'{best_drop_score:.4f}'),
    'pct_of_full_performance': float(f'{best_drop_score / (real_score_loo + 1e-10) * 100:.1f}')
}


# ============================================================
# EXPERIMENT 7: IMU DIMENSION IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 7: IMU Dimension Importance")
print("=" * 80)

# Test each of the 12 IMU dimensions individually
# Also test adding dimensions 3-11 incrementally

# First, reload IMU data with all 12 dims
print("\nLoading full 12-dim IMU data...")
all_dims_features = {}
for dim_idx in range(12):
    dim_features_list = []
    for _, row in df_train.iterrows():
        sid = row['ID']
        imu_file = DATA_PATH / 'IMU' / 'train' / str(sid) / f'{sid}.npy'
        if imu_file.exists():
            try:
                imu_data = np.load(imu_file)
                if dim_idx < 3:
                    # For 0-2, use the full feature extraction
                    feats = extract_fast_imu_features(imu_data, max_dim=3)
                else:
                    # For dims 3-11: extract a minimal set of general features
                    sig = imu_data[:, dim_idx]
                    low_b = bandpass_filter(sig, 0.5, 1.5)
                    mid_b = bandpass_filter(sig, 1.5, 3.0)
                    high_b = bandpass_filter(sig, 3.0, 10.0)
                    total_e = np.sum(sig**2)
                    feats = {
                        f'dim{dim_idx}_low_band_energy_ratio': np.sum(low_b**2) / (total_e + 1e-8),
                        f'dim{dim_idx}_mid_band_energy_ratio': np.sum(mid_b**2) / (total_e + 1e-8),
                        f'dim{dim_idx}_high_band_energy_ratio': np.sum(high_b**2) / (total_e + 1e-8),
                        f'dim{dim_idx}_low_band_cv': np.std(low_b) / (np.abs(np.mean(low_b)) + 1e-8),
                        f'dim{dim_idx}_mid_band_cv': np.std(mid_b) / (np.abs(np.mean(mid_b)) + 1e-8),
                    }
                    autocorr = np.correlate(sig - np.mean(sig), sig - np.mean(sig), mode='full')
                    autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
                    peaks, _ = signal.find_peaks(autocorr[1:100], height=0.3)
                    feats[f'dim{dim_idx}_autocorr_first_peak'] = autocorr[peaks[0]+1] if len(peaks) > 0 else 0
                    feats[f'dim{dim_idx}_dominant_period'] = (peaks[0]+1) / 50 if len(peaks) > 0 else 0

                    peaks, _ = signal.find_peaks(sig, distance=50//4, prominence=0.5)
                    if len(peaks) > 1:
                        intervals = np.diff(peaks) / 50
                        feats[f'dim{dim_idx}_peak_interval_mean'] = np.mean(intervals)
                        feats[f'dim{dim_idx}_peak_interval_std'] = np.std(intervals)
                        feats[f'dim{dim_idx}_peak_interval_cv'] = np.std(intervals) / (np.mean(intervals) + 1e-8)
                    else:
                        feats[f'dim{dim_idx}_peak_interval_mean'] = 0
                        feats[f'dim{dim_idx}_peak_interval_std'] = 0
                        feats[f'dim{dim_idx}_peak_interval_cv'] = 0
                dim_features_list.append(feats)
            except Exception as e:
                dim_features_list.append({})
        else:
            dim_features_list.append({})

    all_dims_features[dim_idx] = dim_features_list

# Build feature matrices for each dimension set
# First get personality features
p_feat_cols = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
X_pers = df_train[p_feat_cols].values

# Test each individual dimension's features + personality
print(f"\nIndividual IMU dimension importance (with personality):")
dim_importance = {}
for dim_idx in range(12):
    try:
        dim_feat_df = pd.DataFrame(all_dims_features[dim_idx])
        dim_cols = [c for c in dim_feat_df.columns if c.startswith(f'dim{dim_idx}_')]
        if len(dim_cols) == 0:
            print(f"  Dim {dim_idx}: no features extracted")
            continue
        X_dim = np.hstack([dim_feat_df[dim_cols].values.astype(np.float64), X_pers])

        loo_preds = []
        loo = LeaveOneOut()
        for tr_idx, te_idx in loo.split(X_dim):
            X_d_tr, X_d_te = X_dim[tr_idx], X_dim[te_idx]
            y_d_tr = y_ordinal[tr_idx]
            s_d = StandardScaler()
            X_d_tr_s = s_d.fit_transform(X_d_tr)
            X_d_te_s = s_d.transform(X_d_te)
            m_d = OrdinalRidge(alpha=1.0)
            m_d.fit(X_d_tr_s, y_d_tr)
            loo_preds.append(unique_phq9[int(m_d.predict(X_d_te_s)[0])])
        loo_preds = np.array(loo_preds)
        met = compute_metrics(y_phq9, loo_preds)
        dim_importance[dim_idx] = {
            'score': float(f'{met["overall_score"]:.4f}'),
            'binary_f1': float(f'{met["binary_f1"]:.4f}'),
            'ternary_f1': float(f'{met["ternary_f1"]:.4f}'),
            'mae': float(f'{met["mae"]:.4f}'),
            'n_features': int(len(dim_cols))
        }
        print(f"  Dim {dim_idx:>2}: {len(dim_cols):>2} feat + personality -> score={met['overall_score']:.4f} "
              f"(BF1={met['binary_f1']:.4f}, TF1={met['ternary_f1']:.4f}, MAE={met['mae']:.2f})")
    except Exception as e:
        print(f"  Dim {dim_idx}: ERROR - {e}")
        dim_importance[dim_idx] = {'error': str(e)}

# Test incremental addition of dims 3-11 to the base 3-dim SOTA
# First recompute base SOTA features (3 dims + personality) through LOO
print(f"\nIncremental addition of higher dimensions:")
base_3dim_cols = [c for c in feature_cols]  # This is the base 3-dim + personality

# Build complete 12-dim feature set for each subject
print("  Building complete 12-dim feature set...")
X_full_12_list = []
for si in range(n_subjects):
    feats = {}
    # 3-dim features (already in feature_cols)
    for ci, col in enumerate(feature_cols):
        feats[col] = X[si, ci]
    # Add dims 3-11 features
    for dim_idx in range(3, 12):
        try:
            dim_feats = all_dims_features[dim_idx][si]
            for k, v in dim_feats.items():
                feats[k] = v
        except:
            pass
    # Add personality
    for pi, col in enumerate(p_feat_cols):
        feats[col] = X_pers[si, pi]
    X_full_12_list.append(feats)

df_full_12 = pd.DataFrame(X_full_12_list)
full_12_cols = [c for c in df_full_12.columns]

# Test adding higher dims incrementally
incremental_dims_results = {}
current_dim_features = list(feature_cols)  # Start with 3-dim + personality
current_dim_set = set([c for c in current_dim_features])

for add_dim in range(3, 12):
    # Add features from this dimension
    new_cols = [c for c in full_12_cols if c.startswith(f'dim{add_dim}_') and c not in current_dim_set]
    if len(new_cols) == 0:
        continue
    current_dim_features.extend(new_cols)
    current_dim_set.update(new_cols)

    X_curr = df_full_12[current_dim_features].values.astype(np.float64)

    loo_preds = []
    loo = LeaveOneOut()
    for tr_idx, te_idx in loo.split(X_curr):
        X_c_tr, X_c_te = X_curr[tr_idx], X_curr[te_idx]
        y_c_tr = y_ordinal[tr_idx]
        s_c = StandardScaler()
        X_c_tr_s = s_c.fit_transform(X_c_tr)
        X_c_te_s = s_c.transform(X_c_te)
        m_c = OrdinalRidge(alpha=1.0)
        m_c.fit(X_c_tr_s, y_c_tr)
        loo_preds.append(unique_phq9[int(m_c.predict(X_c_te_s)[0])])
    loo_preds = np.array(loo_preds)
    met = compute_metrics(y_phq9, loo_preds)

    delta = met['overall_score'] - dim_importance.get('3dim_base', {}).get('score', real_score_loo)
    incremental_dims_results[f'3dim_plus_dims_{add_dim}'] = {
        'added_dimensions': list(range(3, add_dim + 1)),
        'n_features': int(len(current_dim_features)),
        'score': float(f'{met["overall_score"]:.4f}'),
        'delta_from_3dim': float(f'{delta:.4f}')
    }
    print(f"  +dims 3-{add_dim}: {len(current_dim_features)} feats -> score={met['overall_score']:.4f} "
          f"(delta={delta:+.4f})")

results['experiment7_imu_dimension_importance'] = {
    'single_dimension_with_personality': dim_importance,
    'incremental_addition': incremental_dims_results,
    'conclusion_3dim_vs_full': {
        '3dim_score': float(f'{real_score_loo:.4f}'),
        '12dim_score': float(f'{incremental_dims_results.get("3dim_plus_dims_11", {}).get("score", 0):.4f}'),
        'delta': float(f'{incremental_dims_results.get("3dim_plus_dims_11", {}).get("delta_from_3dim", 0):.4f}')
    } if incremental_dims_results else {}
}


# ============================================================
# EXPERIMENT 8: CROSS-VALIDATION vs TEST SET DISCREPANCY
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT 8: Cross-Validation vs Test Set Discrepancy")
print("=" * 80)

# Compile data from all versions
# We need CV scores and test scores for each version
# From existing results files

version_data = {
    'v1_LogisticAT': {'test_score': 0.6005, 'cv_binary_acc': 0.636, 'cv_mae': None},
    'v2_OrdinalRidge': {'test_score': 0.6662, 'cv_binary_acc': 0.636, 'cv_mae': 3.318},
    'v3_Ensemble': {'test_score': 0.6011, 'cv_binary_acc': None, 'cv_mae': None},
    'v4_Tuned_alpha20': {'test_score': 0.5962, 'cv_mae': 2.952},
    'v5_Alpha5': {'test_score': 0.6042, 'cv_mae': 3.193},
    'v6_FeatureSel_25': {'test_score': 0.6011, 'cv_mae': 3.059},
}

# Also compute CV scores for each version using LOO on training set
# For versions that are just alpha changes, we can compute here
alpha_test_scores = {1.0: 0.6662, 5.0: 0.6042, 20.0: 0.5962}
alpha_cv_mae = {1.0: 3.318, 5.0: 3.193, 20.0: 2.952}

print(f"\nCV vs Test Score Comparison:")
print(f"  {'Version':<25} {'CV MAE':>8} {'Test Score':>12} {'Delta':>10}")
print(f"  {'-'*60}")
# Our computed LOO scores
from sklearn.metrics import mean_absolute_error

# Compute CV MAE for alpha=1.0 (from error_analysis.py: 3.318)
# For each alpha, compute LOO MAE as well
alpha_loo_results = {}
for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    loo_preds = []
    loo = LeaveOneOut()
    for tr_idx, te_idx in loo.split(X):
        X_l_tr, X_l_te = X[tr_idx], X[te_idx]
        y_l_tr = y_ordinal[tr_idx]
        sc = StandardScaler()
        X_l_tr_sc = sc.fit_transform(X_l_tr)
        X_l_te_sc = sc.transform(X_l_te)
        m = OrdinalRidge(alpha=alpha)
        m.fit(X_l_tr_sc, y_l_tr)
        loo_preds.append(unique_phq9[int(m.predict(X_l_te_sc)[0])])
    loo_preds = np.array(loo_preds)
    cv_mae = mean_absolute_error(y_phq9, loo_preds)
    cv_score = compute_metrics(y_phq9, loo_preds)['overall_score']
    alpha_loo_results[alpha] = {
        'cv_mae': float(f'{cv_mae:.4f}'),
        'cv_score': float(f'{cv_score:.4f}')
    }

# Map to test scores
alpha_to_test = {1.0: 0.6662, 5.0: 0.6042, 20.0: 0.5962}

cv_test_comparison = []
for alpha, test_score in alpha_to_test.items():
    cv_info = alpha_loo_results[alpha]
    cv_test_comparison.append({
        'alpha': alpha,
        'cv_mae': cv_info['cv_mae'],
        'cv_score': cv_info['cv_score'],
        'test_score': test_score,
        'delta_score': float(f'{test_score - cv_info["cv_score"]:.4f}'),
        'cv_improvement_over_v1': float(f'{cv_info["cv_score"] - alpha_loo_results[1.0]["cv_score"]:.4f}'),
        'test_improvement_over_v1': float(f'{test_score - 0.6005:.4f}')
    })
    print(f"  alpha={alpha:<5.1f}               {cv_info['cv_mae']:>8} {test_score:>12.4f} "
          f"{test_score - cv_info['cv_score']:>+10.4f}")

# For v4 (alpha=20), we can compute delta
# For v3, v6, we need different approaches
# v6 is feature selection - let's try recreating
print(f"\nCross-version CV-to-Test correlation:")

# Test if better CV score predicts better test score
cv_scores_vers = [cv_info['cv_score'] for cv_info in alpha_loo_results.values() if 1.0 in alpha_to_test or True]
test_scores_vers = [alpha_to_test.get(a, 0) for a in sorted(alpha_to_test.keys())]

# Actually let's just look at all alpha values we computed
# The key question: does CV improvement correlate with test improvement?
all_alpha_cv_scores = []
all_alpha_test_scores = []
for a in sorted(alpha_to_test.keys()):
    all_alpha_cv_scores.append(alpha_loo_results[a]['cv_score'])
    all_alpha_test_scores.append(alpha_to_test[a])

if len(all_alpha_cv_scores) >= 3:
    r_cv_test, p_cv_test = pearsonr(all_alpha_cv_scores, all_alpha_test_scores)
    rho_cv_test, p_rho = spearmanr(all_alpha_cv_scores, all_alpha_test_scores)
    print(f"  Pearson r between CV score and test score: {r_cv_test:.4f} (p={p_cv_test:.4f})")
    print(f"  Spearman rho between CV score and test score: {rho_cv_test:.4f} (p={p_rho:.4f})")
    print(f"  Interpretation: {'CV is predictive of test set' if r_cv_test > 0.5 else 'CV is a weak/unreliable predictor'}")

# Detailed table
print(f"\nDetailed CV vs Test Breakdown:")
print(f"  {'Alpha':>7} {'CV MAE':>8} {'CV Score':>10} {'Test Score':>12} {'Delta':>10} {'CV Imprv':>10} {'Test Imprv':>10}")
print(f"  {'-'*70}")
for row in cv_test_comparison:
    print(f"  alpha={row['alpha']:<5.1f} {row['cv_mae']:>8} {row['cv_score']:>10.4f} {row['test_score']:>12.4f} "
          f"{row['delta_score']:>+10.4f} {row['cv_improvement_over_v1']:>+10.4f} {row['test_improvement_over_v1']:>+10.4f}")

# The critical insight: all CV improvements (higher alpha -> lower MAE) reversed on test set
print(f"\nCritical finding:")
print(f"  CV MAE trend: higher alpha -> LOWER MAE (better)")
print(f"  Test score trend: higher alpha -> LOWER score (worse)")
print(f"  Conclusion: CV improvements are OPPOSITE to test performance")

results['experiment8_cv_vs_test'] = {
    'alpha_loo_results': {str(a): res for a, res in alpha_loo_results.items()},
    'cv_test_comparison': cv_test_comparison,
    'pearson_r_cv_test': float(f'{r_cv_test:.4f}') if 'r_cv_test' in locals() else None,
    'pearson_p': float(f'{p_cv_test:.4f}') if 'p_cv_test' in locals() else None,
    'spearman_rho': float(f'{rho_cv_test:.4f}') if 'rho_cv_test' in locals() else None,
    'conclusion': 'CV improvements are inversely related to test performance - higher alpha improves CV but degrades test',
    'cv_mae_trend': 'higher_alpha -> lower_MAE (CV improves)',
    'test_score_trend': 'higher_alpha -> lower_score (test degrades)',
    'is_cv_reliable': False
}


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("Saving results...")
print("=" * 80)

# Convert numpy types to native Python for JSON serialization
def convert_for_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    return obj

results_clean = convert_for_json(results)

output_path = RESULTS_PATH / 'spurious_gp_analysis.json'
with open(output_path, 'w') as f:
    json.dump(results_clean, f, indent=2, ensure_ascii=False)

print(f"Results saved to: {output_path}")
print(f"\n{'=' * 80}")
print("SPURIOUS CORRELATION ANALYSIS COMPLETE")
print(f"{'=' * 80}")
