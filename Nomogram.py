import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 基础配置 (顶刊风格设置)
# ==========================================
# 配色方案 (Dark Slate Blue / Professional Grey)
COLOR_AXIS = '#2C3E50'  # 主轴颜色 (深岩蓝)
COLOR_TICK = '#2C3E50'  # 刻度颜色
COLOR_TEXT_MAIN = '#000000'  # 主标题颜色 (纯黑)
COLOR_TEXT_SUB = '#34495E'  # 次级文字颜色 (深灰)

# 字体设置 (强制使用 Arial)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

train_file = "修正后的ppcs数据集.xlsx"

# 英文特征映射 (确保图表全英文)
col_map_en = {
    '年龄': 'Age', 'BMI': 'BMI',
    '麻醉时间': 'Anesthesia Duration', '手术时间': 'Surgery Duration',
    '术前白蛋白': 'Preop Albumin', '术前C反应蛋白': 'Preop CRP',
    '术前血红蛋白': 'Preop Hemoglobin', '术前白细胞计数': 'Preop WBC',
    '术前中性粒细胞百分比': 'Preop Neutrophil %', '术前淋巴细胞百分比': 'Preop Lymphocyte %',
    '性别': 'Gender', '吸烟': 'Smoking History',
    '手术类型': 'Surgery Type', '手术方式': 'Surgical Approach',
    '手术部位': 'Surgical Site', '术前肺部情况': 'Preop Lung Condition',
    '术前合并症除外肺部疾病': 'Comorbidities', '镇痛方式': 'Analgesia Method',
    'PPCS': 'PPCS'
}


# ==========================================
# 2. 数据读取与处理
# ==========================================
def load_and_prep(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None
    try:
        df = pd.read_excel(path, engine='openpyxl')
    except:
        return None

    # 清洗逻辑
    if 'PPCS' not in df.columns and len(df.columns) > 0: df.rename(columns={df.columns[0]: 'PPCS'}, inplace=True)
    df = df.drop(columns=['序号', 'cens', '术后24h因低氧未拔管'], errors='ignore')

    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    if df['PPCS'].dtype == 'object': df['PPCS'] = df['PPCS'].map({'PPCS异常组': 1, 'PPCS正常组': 0})
    if '性别' in df.columns and df['性别'].dtype == 'object': df['性别'] = df['性别'].map({'男性': 1, '女性': 0})

    return df.dropna().rename(columns=col_map_en)


df = load_and_prep(train_file)
if df is None: exit()

X = df.drop(columns=['PPCS'])
y = df['PPCS']

# ==========================================
# 3. 特征筛选 (Lasso) & 模型训练
# ==========================================
print("Selecting features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用 Lasso 筛选关键特征 (保持列线图简洁)
sel_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.05, random_state=42)
sel_model.fit(X_scaled, y)
selected_mask = sel_model.coef_[0] != 0

# 如果筛选太少，稍微放宽正则化
if np.sum(selected_mask) < 3:
    sel_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
    sel_model.fit(X_scaled, y)
    selected_mask = sel_model.coef_[0] != 0

X_selected = X.loc[:, selected_mask]
feature_names = X_selected.columns.tolist()
print(f"Selected Features: {feature_names}")

# 训练最终逻辑回归模型 (用于计算刻度)
lr = LogisticRegression(max_iter=5000)
lr.fit(X_selected, y)
coefs = lr.coef_[0]
intercept = lr.intercept_[0]

# ==========================================
# 4. 绘制高级列线图 (High-End Rendering)
# ==========================================
print("Rendering High-Quality Nomogram...")

# 计算 Point 转换比例
ranges = []
for i, col in enumerate(feature_names):
    v_min, v_max = X_selected[col].min(), X_selected[col].max()
    contrib_min = coefs[i] * v_min
    contrib_max = coefs[i] * v_max
    ranges.append(abs(contrib_max - contrib_min))
max_range = max(ranges)
points_per_unit = 100 / max_range

# 创建画布 (高 DPI 保证清晰度)
fig, ax = plt.subplots(figsize=(14, len(feature_names) + 6), dpi=600)

y_start = len(feature_names) + 3
ax.set_ylim(0, y_start + 1.5)
ax.set_xlim(-15, 115)
ax.axis('off')  # 去掉默认坐标轴

# 标题
ax.text(50, y_start + 1.2, 'Nomogram for PPCS Risk Prediction',
        ha='center', fontsize=20, fontweight='bold', color=COLOR_AXIS)

# --- A. 绘制 Points 标尺 ---
ax.plot([0, 100], [y_start, y_start], '-', lw=1.5, color=COLOR_AXIS)
ax.text(-5, y_start, 'Points', ha='right', va='center', fontsize=13, fontweight='bold', color=COLOR_TEXT_MAIN)
for i in range(0, 101, 10):
    ax.plot([i, i], [y_start, y_start + 0.15], '-', lw=1.2, color=COLOR_AXIS)
    ax.text(i, y_start + 0.35, str(i), ha='center', fontsize=11, color=COLOR_TEXT_SUB)

# --- B. 绘制各个特征行 ---
sum_min_contrib = 0
total_points_max = 0

for i, col in enumerate(feature_names):
    current_y = y_start - 1.2 - (i * 1.0)  # 增加行间距，显得不拥挤
    v_min, v_max = X_selected[col].min(), X_selected[col].max()

    # 计算贡献范围
    min_c = min(coefs[i] * v_min, coefs[i] * v_max)
    max_c = max(coefs[i] * v_min, coefs[i] * v_max)
    sum_min_contrib += min_c
    total_points_max += (max_c - min_c) * points_per_unit

    # 绘制横线 (使用浅灰色，突出刻度)
    ax.plot([0, 100], [current_y, current_y], '-', lw=1, color='#BDC3C7')
    ax.text(-5, current_y, col, ha='right', va='center', fontsize=13, fontweight='normal', color=COLOR_TEXT_MAIN)

    # 智能刻度生成
    unique_vals = np.sort(X_selected[col].unique())
    if len(unique_vals) <= 5:
        ticks = unique_vals
    else:
        ticks = np.linspace(v_min, v_max, 6)

    for val in ticks:
        contrib = coefs[i] * val
        px = (contrib - min_c) * points_per_unit

        # 刻度线
        tick_h = 0.12  # 刻度高度
        ax.plot([px, px], [current_y, current_y + tick_h], '-', lw=1.2, color=COLOR_TICK)

        # 标签
        if len(unique_vals) <= 2:
            label = str(int(val))
        else:
            label = f"{int(val)}" if abs(val) >= 10 else f"{val:.1f}"
        ax.text(px, current_y + 0.25, label, ha='center', fontsize=10, color=COLOR_TEXT_SUB)

# --- C. Total Points 轴 ---
tp_y = 1.8
ax.plot([0, 100], [tp_y, tp_y], '-', lw=1.5, color=COLOR_AXIS)
ax.text(-5, tp_y, 'Total Points', ha='right', va='center', fontsize=13, fontweight='bold', color=COLOR_TEXT_MAIN)

visual_scale = 100 / total_points_max
tp_ticks = np.linspace(0, total_points_max, 10)
for tp in tp_ticks:
    vx = tp * visual_scale
    ax.plot([vx, vx], [tp_y, tp_y + 0.15], '-', lw=1.2, color=COLOR_AXIS)
    ax.text(vx, tp_y + 0.35, f"{int(tp)}", ha='center', fontsize=11, color=COLOR_TEXT_SUB)

# --- D. 风险轴 (Risk Axis) - 带红绿渐变 ---
prob_y = 0.5
ax.text(-5, prob_y, 'Risk of PPCS', ha='right', va='center', fontsize=13, fontweight='bold', color=COLOR_TEXT_MAIN)

# 制作渐变色带 (Gradient Bar)
probs = np.linspace(0.01, 0.99, 500)
x_coords = []
for p in probs:
    logit = np.log(p / (1 - p))
    tp_val = (logit - intercept - sum_min_contrib) * points_per_unit
    vx = tp_val * visual_scale
    x_coords.append(vx)

# 过滤有效范围 0-100
valid_indices = [i for i, x in enumerate(x_coords) if 0 <= x <= 100]
if valid_indices:
    valid_x = [x_coords[i] for i in valid_indices]
    valid_p = [probs[i] for i in valid_indices]

    # 颜色映射: 绿色(低风险) -> 黄色 -> 红色(高风险)
    cmap = plt.get_cmap('RdYlGn_r')

    # 绘制密集的彩色线段，模拟渐变条
    for i in range(len(valid_x) - 1):
        x1, x2 = valid_x[i], valid_x[i + 1]
        p_val = valid_p[i]
        color = cmap(p_val)
        # lw=8 让线条变粗，看起来像一个 Bar
        ax.plot([x1, x2], [prob_y, prob_y], '-', lw=8, color=color, solid_capstyle='butt')

# 在渐变条上方添加刻度
major_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
for p in major_probs:
    logit = np.log(p / (1 - p))
    tp_val = (logit - intercept - sum_min_contrib) * points_per_unit
    vx = tp_val * visual_scale
    if 0 <= vx <= 100:
        ax.plot([vx, vx], [prob_y - 0.1, prob_y + 0.1], '-', lw=1.5, color='black')
        ax.text(vx, prob_y - 0.3, f"{p}", ha='center', fontsize=11, fontweight='bold', color='black')

plt.tight_layout()
output_file = 'Nomogram_HighQuality.png'
plt.savefig(output_file, dpi=600, bbox_inches='tight')
print(f"Success! High-quality Nomogram saved to: {output_file}")
plt.show()