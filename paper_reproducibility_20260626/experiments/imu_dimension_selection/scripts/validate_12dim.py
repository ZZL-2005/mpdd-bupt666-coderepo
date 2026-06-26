#!/usr/bin/env python3
"""验证12维IMU特征的性能"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, cohen_kappa_score
from mord import OrdinalRidge
import re

def compute_ccc(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred)**2 + 1e-8)

def bandpass_filter(sig, lowcut, highcut, fs=50, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, sig)

def extract_imu_features(imu_data, fs=50):
    features = {}
    for dim in range(12):
        sig = imu_data[:, dim]
        low_band = bandpass_filter(sig, 0.5, 1.5, fs)
        mid_band = bandpass_filter(sig, 1.5, 3.0, fs)
        high_band = bandpass_filter(sig, 3.0, 10.0, fs)

        total_energy = np.sum(sig**2)
        features[f'dim{dim}_low_energy_ratio'] = np.sum(low_band**2) / (total_energy + 1e-8)
        features[f'dim{dim}_mid_energy_ratio'] = np.sum(mid_band**2) / (total_energy + 1e-8)
        features[f'dim{dim}_high_energy_ratio'] = np.sum(high_band**2) / (total_energy + 1e-8)
        features[f'dim{dim}_low_cv'] = np.std(low_band) / (np.abs(np.mean(low_band)) + 1e-8)
        features[f'dim{dim}_mid_cv'] = np.std(mid_band) / (np.abs(np.mean(mid_band)) + 1e-8)

        autocorr = np.correlate(sig - np.mean(sig), sig - np.mean(sig), mode='full')
        autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
        peaks, _ = signal.find_peaks(autocorr[1:100], height=0.3)
        features[f'dim{dim}_autocorr_peak'] = autocorr[peaks[0]+1] if len(peaks) > 0 else 0
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
    features['mag_mean'] = np.mean(accel_mag)
    features['mag_std'] = np.std(accel_mag)
    features['mag_cv'] = np.std(accel_mag) / (np.mean(accel_mag) + 1e-8)

    return features

def extract_personality(text):
    scores = {}
    traits = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']

    for trait in traits:
        patterns = [
            rf'{trait}[:\s]+(\d+)',
            rf'(\d+)\s+(?:on|for)\s+{trait}',
            rf'{trait}\s+score\s+(?:of|is)\s+(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                scores[trait] = float(match.group(1))
                break

    return scores if len(scores) == 5 else None

print("Young G+P 12维特征验证...")
data_path = Path('/data/zilu/mpdd2026/datasets/MPDD-AVG-2026/Young')

print("\n1. 加载训练数据...")
df_labels = pd.read_csv(data_path / 'split_labels_train.csv')
df_desc = pd.read_csv(data_path / 'descriptions.csv')

train_features = []
for _, row in df_labels.iterrows():
    sample_id = row['ID']
    imu_file = data_path / 'IMU' / 'train' / str(sample_id) / f'{sample_id}.npy'

    if imu_file.exists():
        imu_data = np.load(imu_file)
        features = extract_imu_features(imu_data)

        desc_row = df_desc[df_desc['ID'] == sample_id]
        if not desc_row.empty:
            p_scores = extract_personality(desc_row.iloc[0]['Descriptions'])
            if p_scores:
                features.update(p_scores)
                features['ID'] = sample_id
                features['phq9_score'] = row['phq9_score']
                features['label2'] = row['label2']
                features['label3'] = row['label3']
                train_features.append(features)

df_train = pd.DataFrame(train_features)
print(f"✓ 训练样本: {len(df_train)}")

feature_cols = [col for col in df_train.columns if col not in ['ID', 'phq9_score', 'label2', 'label3']]
print(f"✓ 特征数: {len(feature_cols)}")

X = df_train[feature_cols].values
y_phq9 = df_train['phq9_score'].values
y_binary = df_train['label2'].values
y_ternary = df_train['label3'].values

unique_phq9 = np.sort(np.unique(y_phq9))
phq9_to_ordinal = {score: i for i, score in enumerate(unique_phq9)}
ordinal_to_phq9 = {i: score for i, score in enumerate(unique_phq9)}
y_ordinal = np.array([phq9_to_ordinal[score] for score in y_phq9])

print("\n2. 划分训练集和验证集...")
X_train, X_val, y_ordinal_train, y_ordinal_val = train_test_split(
    X, y_ordinal, test_size=0.2, random_state=42, stratify=y_binary
)
_, _, y_phq9_train, y_phq9_val = train_test_split(
    X, y_phq9, test_size=0.2, random_state=42, stratify=y_binary
)
_, _, y_binary_train, y_binary_val = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
_, _, y_ternary_train, y_ternary_val = train_test_split(
    X, y_ternary, test_size=0.2, random_state=42, stratify=y_binary
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}")

print("\n3. 训练OrdinalRidge...")
model = OrdinalRidge(alpha=1.0)
model.fit(X_train_scaled, y_ordinal_train)

print("\n4. 评估...")
y_ordinal_pred = model.predict(X_val_scaled)
y_phq9_pred = np.array([ordinal_to_phq9[int(o)] for o in y_ordinal_pred])

y_binary_pred = (y_phq9_pred >= 5).astype(int)
y_ternary_pred = np.zeros_like(y_phq9_pred, dtype=int)
y_ternary_pred[y_phq9_pred < 5] = 0
y_ternary_pred[(y_phq9_pred >= 5) & (y_phq9_pred < 10)] = 1
y_ternary_pred[y_phq9_pred >= 10] = 2

binary_f1 = f1_score(y_binary_val, y_binary_pred, average='macro')
binary_ccc = compute_ccc(y_binary_val, y_binary_pred)
binary_kappa = cohen_kappa_score(y_binary_val, y_binary_pred)

ternary_f1 = f1_score(y_ternary_val, y_ternary_pred, average='macro')
ternary_ccc = compute_ccc(y_ternary_val, y_ternary_pred)
ternary_kappa = cohen_kappa_score(y_ternary_val, y_ternary_pred)

cls_f1 = (binary_f1 + ternary_f1) / 2
cls_ccc = (binary_ccc + ternary_ccc) / 2
cls_kappa = (binary_kappa + ternary_kappa) / 2
score = (cls_f1 + cls_ccc + cls_kappa) / 3

print("\n" + "="*80)
print("Young G+P 12维特征验证结果")
print("="*80)
print(f"\nBinary分类:")
print(f"  F1:    {binary_f1:.4f}")
print(f"  CCC:   {binary_ccc:.4f}")
print(f"  Kappa: {binary_kappa:.4f}")

print(f"\nTernary分类:")
print(f"  F1:    {ternary_f1:.4f}")
print(f"  CCC:   {ternary_ccc:.4f}")
print(f"  Kappa: {ternary_kappa:.4f}")

print(f"\n总体:")
print(f"  Cls_F1:    {cls_f1:.4f}")
print(f"  Cls_CCC:   {cls_ccc:.4f}")
print(f"  Cls_Kappa: {cls_kappa:.4f}")
print(f"  Score:     {score:.4f}")

print("\n对比:")
print(f"  3维特征 (测试集): 0.6662")
print(f"  12维特征 (验证集): {score:.4f}")
