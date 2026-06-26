#!/usr/bin/env python3
"""
基于3维SOTA配置的超参数搜索
目标：突破0.6662分数
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, f1_score, cohen_kappa_score
from mord import OrdinalRidge

def bandpass_filter(sig, lowcut, highcut, fs=50, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, sig)

def extract_fast_imu_features(imu_data, fs=50):
    features = {}
    for dim in range(3):  # SOTA配置：只用3维
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

print("=" * 70)
print("基于3维SOTA配置的超参数搜索")
print("=" * 70)

data_path = Path('/data/zilu/mpdd2026/datasets/MPDD-AVG-2026/Young')

print("\n[1/5] 加载训练数据...")
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

print(f"✓ 特征维度: {len(feature_cols)} (36 IMU + 5 Personality)")
print(f"✓ PHQ-9范围: {int(min(y_phq9_train))}-{int(max(y_phq9_train))}")

print("\n[2/5] 超参数网格搜索 (5折交叉验证)...")
# 扩展alpha搜索范围
alphas = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

best_alpha = None
best_mae = float('inf')
results = []

for alpha in alphas:
    maes = []
    for train_idx, val_idx in kf.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_ordinal_train[train_idx], y_ordinal_train[val_idx]

        model = OrdinalRidge(alpha=alpha)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        y_val_phq9 = np.array([ordinal_to_phq9[int(o)] for o in y_val])
        y_pred_phq9 = np.array([ordinal_to_phq9[int(o)] for o in y_pred])
        mae = mean_absolute_error(y_val_phq9, y_pred_phq9)
        maes.append(mae)

    mean_mae = np.mean(maes)
    std_mae = np.std(maes)
    results.append({'alpha': alpha, 'mae': mean_mae, 'std': std_mae})
    print(f"  alpha={alpha:6.2f}: MAE={mean_mae:.3f} ± {std_mae:.3f}")

    if mean_mae < best_mae:
        best_mae = mean_mae
        best_alpha = alpha

print(f"\n✓ 最优alpha: {best_alpha} (MAE={best_mae:.3f})")
print(f"✓ SOTA配置(alpha=1.0)的MAE: {[r for r in results if r['alpha']==1.0][0]['mae']:.3f}")

df_results = pd.DataFrame(results)
output_dir = Path('/data/zilu/mpdd2026/Young G+P/submissions/hyperparameter_search')
output_dir.mkdir(parents=True, exist_ok=True)
df_results.to_csv(output_dir / 'cv_results.csv', index=False)
print(f"✓ 交叉验证结果已保存")

print("\n[3/5] 生成多个版本的提交文件...")
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
X_test = df_test[feature_cols].values
X_test_scaled = scaler.transform(X_test)

# 生成多个版本
versions_to_test = [
    ('v_sota', 1.0, 'SOTA配置'),
    ('v_best', best_alpha, f'最优alpha={best_alpha}'),
    ('v_low_reg', 0.1, '低正则化'),
    ('v_high_reg', 5.0, '高正则化'),
]

print(f"\n[4/5] 训练并生成{len(versions_to_test)}个版本...")
for version_name, alpha, desc in versions_to_test:
    print(f"\n  {version_name} ({desc}):")

    model = OrdinalRidge(alpha=alpha)
    model.fit(X_train_scaled, y_ordinal_train)

    y_ordinal_pred = model.predict(X_test_scaled)
    y_phq9_pred = np.array([ordinal_to_phq9[int(o)] for o in y_ordinal_pred])

    binary_pred = (y_phq9_pred >= 5).astype(int)
    ternary_pred = np.zeros_like(y_phq9_pred, dtype=int)
    ternary_pred[y_phq9_pred < 5] = 0
    ternary_pred[(y_phq9_pred >= 5) & (y_phq9_pred < 10)] = 1
    ternary_pred[y_phq9_pred >= 10] = 2

    print(f"    PHQ-9范围: {int(min(y_phq9_pred))}-{int(max(y_phq9_pred))}")
    print(f"    Binary分布: Normal={np.sum(binary_pred==0)}, Depressed={np.sum(binary_pred==1)}")
    print(f"    Ternary分布: Normal={np.sum(ternary_pred==0)}, Mild={np.sum(ternary_pred==1)}, Severe={np.sum(ternary_pred==2)}")

    version_dir = output_dir / version_name
    version_dir.mkdir(exist_ok=True)

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

    df_binary.to_csv(version_dir / 'binary.csv', index=False)
    df_ternary.to_csv(version_dir / 'ternary.csv', index=False)

    with open(version_dir / 'config.txt', 'w') as f:
        f.write(f"Version: {version_name}\n")
        f.write(f"Description: {desc}\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Features: 41 (36 IMU + 5 Personality)\n")
        f.write(f"IMU Dimensions: 3 (x, y, z)\n")
        f.write(f"CV MAE: {[r for r in results if r['alpha']==alpha][0]['mae']:.3f}\n")

print(f"\n[5/5] 生成总结报告...")
summary = f"""# 超参数搜索总结

## 搜索配置
- 基础配置: 3维IMU SOTA配置
- 特征维度: 41 (36 IMU + 5 Personality)
- Alpha范围: {min(alphas)} - {max(alphas)}
- 交叉验证: 5折

## 最优结果
- 最优alpha: {best_alpha}
- 最优MAE: {best_mae:.3f}
- SOTA(alpha=1.0) MAE: {[r for r in results if r['alpha']==1.0][0]['mae']:.3f}
- 改进: {([r for r in results if r['alpha']==1.0][0]['mae'] - best_mae):.3f}

## 生成的版本
"""

for version_name, alpha, desc in versions_to_test:
    cv_mae = [r for r in results if r['alpha']==alpha][0]['mae']
    summary += f"\n### {version_name}\n"
    summary += f"- 描述: {desc}\n"
    summary += f"- Alpha: {alpha}\n"
    summary += f"- CV MAE: {cv_mae:.3f}\n"
    summary += f"- 位置: {output_dir / version_name}\n"

summary += f"\n## 下一步\n"
summary += f"1. 测试所有版本，找出实际最优配置\n"
summary += f"2. 如果最优alpha≠1.0，说明SOTA可以改进\n"
summary += f"3. 考虑ensemble多个alpha的模型\n"

with open(output_dir / 'SUMMARY.md', 'w') as f:
    f.write(summary)

print("\n" + "=" * 70)
print("✅ 超参数搜索完成！")
print(f"   生成了{len(versions_to_test)}个版本")
print(f"   输出目录: {output_dir}")
print(f"   最优alpha: {best_alpha} (CV MAE={best_mae:.3f})")
print("=" * 70)
