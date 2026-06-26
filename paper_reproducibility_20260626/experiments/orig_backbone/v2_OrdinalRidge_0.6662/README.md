# Version 2: OrdinalRidge有序回归

## 算法说明

### 核心思路
使用OrdinalRidge（岭回归的序数版本）学习PHQ-9分数，通过正则化提升模型稳定性。

### 方法详解

**1. 与Version 1的区别**
- Version 1: LogisticAT（基于逻辑回归）
- Version 2: OrdinalRidge（基于岭回归）
- 核心差异：正则化方式和优化目标不同

**2. 特征工程**
- 与Version 1完全相同
- IMU特征 (36维) + Personality特征 (5维) = 41维

**3. 模型选择**
- 算法：OrdinalRidge
- 参数：alpha=1.0（L2正则化强度）
- 优势：
  - 交叉验证表现最佳（MAE=2.82）
  - 岭回归的正则化可能提升泛化能力
  - 对特征共线性更鲁棒

**4. 预测流程**
```
IMU+Personality特征 → 标准化 → OrdinalRidge → 序数预测 → PHQ-9分数
                                                    ↓
                                            阈值映射(5, 10)
                                                    ↓
                                        Binary预测 + Ternary预测
```

### 性能表现

**交叉验证结果**:
- MAE: 2.82 ± 0.54（5折交叉验证）
- 在所有测试的序数回归模型中表现最佳

**预期**:
- 可能略优于LogisticAT
- 更稳定的泛化性能

### 优势
1. ✅ 交叉验证最佳模型
2. ✅ L2正则化提升稳定性
3. ✅ 对特征共线性更鲁棒
4. ✅ 保持与Version 1相同的建模思路

### 代码
生成脚本：`generate_submission_ordinalridge.py`

### 状态
🔬 **待测试，理论上可能优于LogisticAT**
