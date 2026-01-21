import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# ==========================================
# 1. 基础设置
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 兼顾英文和中文显示

# 文件路径
train_file = "修正后的ppcs数据集.xlsx"
ext_file = "外部验证.xlsx"  # 您的新文件


# ==========================================
# 2. 数据加载与预处理函数
# ==========================================
def load_and_clean(file_path, is_train=True):
    if not os.path.exists(file_path):
        print(f"[错误] 找不到文件: {file_path}")
        return None

    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        print(f"[错误] 读取失败 {file_path}: {e}")
        return None

    # 剔除泄露列
    cols_to_drop = ['序号', 'cens', '术后24h因低氧未拔管']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # 统一列名 (如果有表头不一致的情况，需在此处理)
    # 假设外部数据第一列也是 PPCS，如果列名不同请修改
    # 这里假设列名已经一致或需要重命名
    # df.rename(columns={'旧列名': '新列名'}, inplace=True)

    return df


# 加载数据
print("1. 加载训练集...")
df_train = load_and_clean(train_file, is_train=True)
print("2. 加载外部验证集...")
df_ext = load_and_clean(ext_file, is_train=False)

if df_train is None or df_ext is None:
    exit()


# ==========================================
# 3. 数据对齐与映射
# ==========================================
# 定义映射规则 (确保训练集和验证集一致)
# 假设训练集 '性别' 是 0/1，外部集是 '男性'/'女性'，需要统一
# 如果训练集本身就是 0/1，则无需映射训练集，只需映射外部集
# 这里为了保险，检查数据类型

def preprocess_data(df):
    # 映射目标变量
    if 'PPCS' in df.columns:
        # 如果是字符串，映射为数字
        if df['PPCS'].dtype == 'object':
            df['PPCS'] = df['PPCS'].map({'PPCS异常组': 1, 'PPCS正常组': 0})

    # 映射性别 (示例，根据您的实际数据调整)
    if '性别' in df.columns and df['性别'].dtype == 'object':
        df['性别'] = df['性别'].map({'男性': 1, '女性': 0})

    # 处理其他可能存在的分类变量映射...

    # 剔除空值
    df = df.dropna()

    X = df.drop(columns=['PPCS'], errors='ignore')
    y = df['PPCS']

    return X, y


print("3. 数据预处理与对齐...")
X_train_raw, y_train = preprocess_data(df_train)
X_ext_raw, y_ext = preprocess_data(df_ext)

# 确保列顺序一致
common_cols = X_train_raw.columns.intersection(X_ext_raw.columns)
X_train_raw = X_train_raw[common_cols]
X_ext_raw = X_ext_raw[common_cols]

print(f"训练集特征数: {X_train_raw.shape[1]}, 样本数: {len(X_train_raw)}")
print(f"外部集特征数: {X_ext_raw.shape[1]}, 样本数: {len(X_ext_raw)}")

# ==========================================
# 4. 训练模型 (Full Training Strategy)
# ==========================================
# 数据平衡 (只对训练集做)
train_data = pd.concat([X_train_raw, y_train], axis=1)
majority = train_data[train_data['PPCS'] == 0]
minority = train_data[train_data['PPCS'] == 1]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
train_balanced = pd.concat([majority, minority_upsampled])

X_train_bal = train_balanced.drop('PPCS', axis=1)
y_train_bal = train_balanced['PPCS']

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)  # 在平衡后的训练集上拟合
X_ext_scaled = scaler.transform(X_ext_raw)  # 用训练集的参数转换外部集

print("4. 训练 Gradient Boosting 模型...")
model = GradientBoostingClassifier(random_state=42, max_depth=2, learning_rate=0.05, n_estimators=100)
model.fit(X_train_scaled, y_train_bal)

# ==========================================
# 5. 外部验证与绘图
# ==========================================
# 预测
y_pred_ext = model.predict(X_ext_scaled)
y_prob_ext = model.predict_proba(X_ext_scaled)[:, 1]

# 计算指标
acc = accuracy_score(y_ext, y_pred_ext)
auc_score = roc_auc_score(y_ext, y_prob_ext)
print(f"\n[外部验证结果] Accuracy: {acc:.4f}, AUC: {auc_score:.4f}")

# --- 绘图 ---
fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=300)

# A. ROC Curve
fpr, tpr, _ = roc_curve(y_ext, y_prob_ext)
axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'External Validation (AUC = {auc_score:.3f})')
axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('A. ROC Curve - External Validation')
axes[0, 0].legend(loc="lower right")
axes[0, 0].grid(True, alpha=0.3)

# B. Radar Chart (Metrics)
metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC', 'F1']
tn, fp, fn, tp = confusion_matrix(y_ext, y_pred_ext).ravel()
sens = recall_score(y_ext, y_pred_ext)
spec = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = f1_score(y_ext, y_pred_ext)
values = [acc, sens, spec, auc_score, f1]

# Radar Plot Setup
ax_radar = plt.subplot(2, 2, 2, polar=True)
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
values += values[:1]
angles += angles[:1]
ax_radar.plot(angles, values, color='green', linewidth=2, linestyle='solid')
ax_radar.fill(angles, values, color='green', alpha=0.25)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(metrics)
ax_radar.set_title('B. Metrics Radar - External Validation', y=1.1)

# C. DCA
thresholds = np.linspace(0, 1, 101)
net_benefits = []
for thresh in thresholds:
    if thresh == 0:
        nb = np.mean(y_ext)
    elif thresh == 1:
        nb = 0
    else:
        y_pred_t = (y_prob_ext >= thresh).astype(int)
        tp_t = np.sum((y_pred_t == 1) & (y_ext == 1))
        fp_t = np.sum((y_pred_t == 1) & (y_ext == 0))
        n = len(y_ext)
        nb = (tp_t / n) - (fp_t / n) * (thresh / (1 - thresh))
    net_benefits.append(nb)

nb_all = [np.mean(y_ext) - (1 - np.mean(y_ext)) * (t / (1 - t)) for t in thresholds]
axes[1, 0].plot(thresholds, net_benefits, color='red', lw=2, label='Model')
axes[1, 0].plot(thresholds, nb_all, color='gray', linestyle='--', label='Treat All')
axes[1, 0].plot(thresholds, [0] * len(thresholds), color='black', linestyle='-', label='Treat None')
axes[1, 0].set_ylim([-0.05, 0.4])
axes[1, 0].set_xlabel('Threshold Probability')
axes[1, 0].set_ylabel('Net Benefit')
axes[1, 0].set_title('C. DCA - External Validation')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# D. CIC
num_high_risk = []
num_true_pos = []
pop_size = 1000
for thresh in thresholds:
    y_pred_t = (y_prob_ext >= thresh).astype(int)
    factor = pop_size / len(y_ext)
    num_high_risk.append(np.sum(y_pred_t == 1) * factor)
    num_true_pos.append(np.sum((y_pred_t == 1) & (y_ext == 1)) * factor)

axes[1, 1].plot(thresholds, num_high_risk, color='red', lw=2, label='Number High Risk')
axes[1, 1].plot(thresholds, num_true_pos, color='blue', linestyle='--', lw=2, label='Number True Positives')
axes[1, 1].set_xlabel('Threshold Probability')
axes[1, 1].set_ylabel('Number per 1000 Patients')
axes[1, 1].set_title('D. CIC - External Validation')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('External_Validation_Analysis.png', dpi=300)
print("绘图完成！图片已保存为: External_Validation_Analysis.png")
plt.show()
