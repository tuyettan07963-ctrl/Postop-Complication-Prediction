import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. 读取与清洗数据
# ==========================================
file_path = "修正后的ppcs数据集.xlsx"
print(f"正在读取文件: {file_path}...")

try:
    df = pd.read_excel(file_path, engine='openpyxl')
except Exception as e:
    print(f"读取失败: {e}")
    exit()

# 剔除泄露和无关列
cols_to_drop = ['序号', 'cens', '术后24h因低氧未拔管']
df_corrected = df.drop(columns=cols_to_drop, errors='ignore')

# 准备数据
X = df_corrected.drop(columns=['PPCS'], errors='ignore')
y = df_corrected['PPCS'].map({'PPCS异常组': 1, 'PPCS正常组': 0})

# ==========================================
# 2. 数据划分与平衡
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 训练集过采样
train_data = pd.concat([X_train, y_train], axis=1)
majority = train_data[train_data['PPCS'] == 0]
minority = train_data[train_data['PPCS'] == 1]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
train_balanced = pd.concat([majority, minority_upsampled])

X_train_bal = train_balanced.drop('PPCS', axis=1)
y_train_bal = train_balanced['PPCS']

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # 原始训练集分布
X_test_scaled = scaler.transform(X_test)
X_train_bal_scaled = scaler.transform(X_train_bal) # 平衡后的训练数据

# ==========================================
# 3. 训练模型 (使用表现最好的 Gradient Boosting)
# ==========================================
model = GradientBoostingClassifier(random_state=42, max_depth=2, learning_rate=0.05, n_estimators=100)
model.fit(X_train_bal_scaled, y_train_bal)

# ==========================================
# 4. 定义 DCA 和 CIC 计算函数
# ==========================================
def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefits = []
    for thresh in thresholds:
        if thresh == 0:
            net_benefit = np.mean(y_true)
        elif thresh == 1:
            net_benefit = 0
        else:
            y_pred = (y_prob >= thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            n = len(y_true)
            net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefits.append(net_benefit)
    return net_benefits

def calculate_clinical_impact(y_true, y_prob, thresholds, population_size=1000):
    num_high_risk = []
    num_true_positives = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        factor = population_size / len(y_true)
        n_high_risk = np.sum(y_pred == 1) * factor
        n_true_pos = np.sum((y_pred == 1) & (y_true == 1)) * factor
        num_high_risk.append(n_high_risk)
        num_true_positives.append(n_true_pos)
    return num_high_risk, num_true_positives

# ==========================================
# 5. 计算绘图数据
# ==========================================
thresholds = np.linspace(0, 1, 101)

# 获取概率
y_train_prob = model.predict_proba(X_train_scaled)[:, 1] # 使用原始比例的训练集评估
y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

# 计算 Train 指标
nb_train = calculate_net_benefit(y_train, y_train_prob, thresholds)
cic_hr_train, cic_tp_train = calculate_clinical_impact(y_train, y_train_prob, thresholds)
nb_all_train = [np.mean(y_train) - (1 - np.mean(y_train)) * (t / (1 - t)) for t in thresholds]

# 计算 Validation 指标
nb_test = calculate_net_benefit(y_test, y_test_prob, thresholds)
cic_hr_test, cic_tp_test = calculate_clinical_impact(y_test, y_test_prob, thresholds)
nb_all_test = [np.mean(y_test) - (1 - np.mean(y_test)) * (t / (1 - t)) for t in thresholds]

nb_none = [0] * len(thresholds)

# ==========================================
# 6. 绘图 (高清 + 全英文)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=300)

# A. DCA - Train
axes[0, 0].plot(thresholds, nb_train, color='red', lw=2, label='Model (Gradient Boosting)')
axes[0, 0].plot(thresholds, nb_all_train, color='gray', linestyle='--', label='Treat All')
axes[0, 0].plot(thresholds, nb_none, color='black', linestyle='-', label='Treat None')
axes[0, 0].set_ylim([-0.05, 0.4])
axes[0, 0].set_xlabel('Threshold Probability')
axes[0, 0].set_ylabel('Net Benefit')
axes[0, 0].set_title('A. DCA - Training Set')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# B. DCA - Validation
axes[0, 1].plot(thresholds, nb_test, color='blue', lw=2, label='Model (Gradient Boosting)')
axes[0, 1].plot(thresholds, nb_all_test, color='gray', linestyle='--', label='Treat All')
axes[0, 1].plot(thresholds, nb_none, color='black', linestyle='-', label='Treat None')
axes[0, 1].set_ylim([-0.05, 0.4])
axes[0, 1].set_xlabel('Threshold Probability')
axes[0, 1].set_ylabel('Net Benefit')
axes[0, 1].set_title('B. DCA - Validation Set')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# C. CIC - Train
axes[1, 0].plot(thresholds, cic_hr_train, color='red', lw=2, label='Number High Risk')
axes[1, 0].plot(thresholds, cic_tp_train, color='blue', linestyle='--', lw=2, label='Number True Positives')
axes[1, 0].set_xlabel('Threshold Probability')
axes[1, 0].set_ylabel('Number per 1000 Patients')
axes[1, 0].set_title('C. Clinical Impact Curve - Training Set')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# D. CIC - Validation
axes[1, 1].plot(thresholds, cic_hr_test, color='red', lw=2, label='Number High Risk')
axes[1, 1].plot(thresholds, cic_tp_test, color='blue', linestyle='--', lw=2, label='Number True Positives')
axes[1, 1].set_xlabel('Threshold Probability')
axes[1, 1].set_ylabel('Number per 1000 Patients')
axes[1, 1].set_title('D. Clinical Impact Curve - Validation Set')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_file = 'DCA_CIC_Analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"绘图完成！图片已保存为: {output_file}")
plt.show()