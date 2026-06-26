#!/usr/bin/env python3
"""
使用OrdinalRidge生成提交文件
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from scipy import signal
from sklearn.preprocessing import StandardScaler
from mord import OrdinalRidge

def bandpass_filter(sig, lowcut, highcut, fs=50, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, sig)

def extract_fast_imu_features(imu_data, fs=50):
    features = {}
    for dim in range(3):  # 只使用前3维（加速度计x,y,z）- 这是0.6662的配置
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

print("使用OrdinalRidge生成提交...")
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
        features = extract_fast_imu_features(imu_data)

        desc_row = df_desc[df_desc['ID'] == sample_id]
        if len(desc_row) > 0:
            p_scores = extract_personality_scores_robust(desc_row.iloc[0]['Descriptions'])
            features.update(p_scores)
            features['ID'] = sample_id
            features['phq9_score'] = row['phq9_score']
            train_features.append(features)

df_train = pd.DataFrame(train_features).dropna()
print(f"✓ 训练样本: {len(df_train)}")

feature_cols = [col for col in df_train.columns if col not in ['ID', 'phq9_score']]
X_train = df_train[feature_cols].values
y_phq9_train = df_train['phq9_score'].values

unique_phq9 = np.sort(np.unique(y_phq9_train))
phq9_to_ordinal = {score: i for i, score in enumerate(unique_phq9)}
ordinal_to_phq9 = {i: score for i, score in enumerate(unique_phq9)}
y_ordinal_train = np.array([phq9_to_ordinal[score] for score in y_phq9_train])

print("\n2. 训练OrdinalRidge模型...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

ordinal_model = OrdinalRidge(alpha=1.0)
ordinal_model.fit(X_train_scaled, y_ordinal_train)
print("✓ 模型训练完成")

print("\n3. 加载测试数据...")
test_ids = [1, 5, 7, 13, 15, 22, 28, 33, 34, 40, 42, 44, 47, 58, 74, 83, 85, 89, 90, 93, 105, 110]

test_features = []
for sample_id in test_ids:
    imu_file = data_path / 'IMU' / str(sample_id) / f'{sample_id}.npy'

    if imu_file.exists():
        imu_data = np.load(imu_file)
        features = extract_fast_imu_features(imu_data)

        desc_row = df_desc[df_desc['ID'] == sample_id]
        if len(desc_row) > 0:
            p_scores = extract_personality_scores_robust(desc_row.iloc[0]['Descriptions'])
            features.update(p_scores)
            features['ID'] = sample_id
            test_features.append(features)

df_test = pd.DataFrame(test_features).dropna()
print(f"✓ 测试样本: {len(df_test)}")

print("\n4. 生成预测...")
X_test = df_test[feature_cols].values
X_test_scaled = scaler.transform(X_test)

y_ordinal_pred = ordinal_model.predict(X_test_scaled)
y_phq9_pred = np.array([ordinal_to_phq9[int(o)] for o in y_ordinal_pred])

binary_pred = (y_phq9_pred >= 5).astype(int)
ternary_pred = np.zeros_like(y_phq9_pred, dtype=int)
ternary_pred[y_phq9_pred < 5] = 0
ternary_pred[(y_phq9_pred >= 5) & (y_phq9_pred < 10)] = 1
ternary_pred[y_phq9_pred >= 10] = 2

print("\n5. 生成提交文件...")
output_dir = Path('/data/zilu/mpdd2026/observation/outputs/submission')

df_binary = pd.DataFrame({
    'id': df_test['ID'].astype(int),
    'binary_pred': binary_pred.astype(int),
    'phq9_pred': y_phq9_pred
})

df_ternary = pd.DataFrame({
    'id': df_test['ID'].astype(int),
    'ternary_pred': ternary_pred.astype(int),
    'phq9_pred': y_phq9_pred
})

df_binary.to_csv(output_dir / 'binary.csv', index=False)
df_ternary.to_csv(output_dir / 'ternary.csv', index=False)

print(f"✓ 已生成: binary.csv")
print(f"✓ 已生成: ternary.csv")
print(f"\n✅ OrdinalRidge提交文件生成完成！")
