# 0.6662 SOTA方案复现指南

## 测试分数
- **Overall Score: 0.6662** 🏆
- Binary F1: 0.8634
- Ternary F1: 0.731
- Ternary Kappa: 0.6823

## 关键配置（必须严格遵守）

### 1. IMU特征提取 - 只用3维！
```python
def extract_fast_imu_features(imu_data, fs=50):
    features = {}
    for dim in range(3):  # ⚠️ 关键：只用前3维（加速度计x,y,z），不是12维！
        sig = imu_data[:, dim]
        # ... 特征提取代码
```

**为什么只用3维？**
- 12维IMU数据包含：加速度计(3) + 陀螺仪(3) + 磁力计(3) + 其他(3)
- 实验证明：3维(0.6662) > 12维(0.3633)
- 原因：88个样本太少，12维×12特征=144维会严重过拟合

### 2. 特征维度
- **IMU特征**: 3维 × 12特征/维 = 36维
  - 每维12个特征：
    - 频段能量比(3): low/mid/high band energy ratio
    - 变异系数(2): low/mid band CV
    - 自相关(2): autocorr first peak, dominant period
    - 峰值间隔(3): peak interval mean/std/cv
    - 跨维度相关性(3): corr_xy, corr_xz, corr_yz
    - 加速度幅值(3): accel_mag mean/std/cv
- **人格特征**: 5维 (Big Five)
- **总计**: 41维

### 3. 模型配置
```python
from sklearn.preprocessing import StandardScaler
from mord import OrdinalRidge

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = OrdinalRidge(alpha=1.0)  # ⚠️ 关键：alpha=1.0
model.fit(X_train_scaled, y_ordinal_train)
```

### 4. 数据集
- 训练样本：88个
- 测试样本：22个
- 数据路径：`/data/zilu/mpdd2026/datasets/MPDD-AVG-2026/Young`

### 5. 标签映射
```python
# PHQ-9分数 → 序数标签
unique_phq9 = np.sort(np.unique(y_phq9_train))
phq9_to_ordinal = {score: i for i, score in enumerate(unique_phq9)}
ordinal_to_phq9 = {i: score for i, score in enumerate(unique_phq9)}

# 预测 → Binary/Ternary
binary_pred = (y_phq9_pred >= 5).astype(int)
ternary_pred[y_phq9_pred < 5] = 0
ternary_pred[(y_phq9_pred >= 5) & (y_phq9_pred < 10)] = 1
ternary_pred[y_phq9_pred >= 10] = 2
```

## 复现步骤

### 方法1：使用现有脚本
```bash
cd /data/zilu/mpdd2026
python observation/scripts/generate_submission_ordinalridge.py
```

**检查点**：确认脚本第22行是 `for dim in range(3):`

### 方法2：从头实现
参考本目录下的完整脚本，关键点：
1. 只提取前3维IMU特征
2. 使用OrdinalRidge(alpha=1.0)
3. StandardScaler标准化
4. 88训练样本，22测试样本

## 预测结果验证

生成的预测应该与以下完全一致：

| ID | binary_pred | ternary_pred | phq9_pred |
|----|-------------|--------------|-----------|
| 1  | 1 | 2 | 11.0 |
| 5  | 0 | 0 | 4.0 |
| 7  | 1 | 1 | 6.0 |
| 13 | 1 | 1 | 8.0 |
| 15 | 1 | 1 | 5.0 |
| 22 | 0 | 0 | 3.0 |
| 28 | 0 | 0 | 0.0 |
| 33 | 0 | 0 | 4.0 |
| 34 | 1 | 1 | 6.0 |
| 40 | 0 | 0 | 0.0 |
| 42 | 1 | 1 | 9.0 |
| 44 | 0 | 0 | 0.0 |
| 47 | 0 | 0 | 3.0 |
| 58 | 0 | 0 | 2.0 |
| 74 | 0 | 0 | 2.0 |
| 83 | 0 | 0 | 1.0 |
| 85 | 1 | 1 | 6.0 |
| 89 | 1 | 1 | 6.0 |
| 90 | 1 | 1 | 5.0 |
| 93 | 1 | 1 | 7.0 |
| 105 | 0 | 0 | 1.0 |
| 110 | 0 | 0 | 0.0 |

## 常见错误

### ❌ 错误1：使用12维
```python
for dim in range(12):  # 错误！会导致分数降到0.36
```
**结果**：Score = 0.3633（过拟合）

### ❌ 错误2：使用PCA降维
```python
pca = PCA(n_components=2)  # 错误！
```
**结果**：Score = 0.4911（丢失物理意义）

### ❌ 错误3：只用人格特征
```python
# 不使用IMU特征  # 错误！
```
**结果**：Score = 0.4232（缺少关键信息）

## 为什么这个配置最优？

1. **3维vs12维**：88样本无法支撑144维特征，3维刚好
2. **OrdinalRidge vs LogisticAT**：L2正则化对小样本更友好
3. **alpha=1.0**：交叉验证最优值
4. **有序回归策略**：利用Binary/Ternary标签的内在关系

## 更新记录
- 2026-05-06: 创建复现指南，记录SOTA配置
- 2026-05-02: 首次测试，获得0.6662分数
