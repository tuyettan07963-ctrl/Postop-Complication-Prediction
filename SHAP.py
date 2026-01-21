import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shap  # 必须安装: pip install shap
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import GradientBoostingClassifier

# ==========================================
# 1. 基础配置
# ==========================================
# 设置绘图风格，尽量贴近学术发表 (Arial字体)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
train_file = "修正后的ppcs数据集.xlsx"

# 特征名中英文对照表 (确保生成全英文图表)
col_map_en = {
    '年龄': 'Age',
    'BMI': 'BMI',
    '麻醉时间': 'Anesthesia Duration',
    '手术时间': 'Surgery Duration',
    '术前白蛋白': 'Preop Albumin',
    '术前C反应蛋白': 'Preop CRP',
    '术前血红蛋白': 'Preop Hemoglobin',
    '术前白细胞计数': 'Preop WBC',
    '术前中性粒细胞百分比': 'Preop Neutrophil %',
    '术前淋巴细胞百分比': 'Preop Lymphocyte %',
    '性别': 'Gender',
    '吸烟': 'Smoking History',
    '手术类型': 'Surgery Type',
    '手术方式': 'Surgical Approach',
    '手术部位': 'Surgical Site',
    '术前肺部情况': 'Preop Lung Condition',
    '术前合并症除外肺部疾病': 'Comorbidities',
    '镇痛方式': 'Analgesia Method',
    'PPCS': 'PPCS'  # 目标变量
}


# ==========================================
# 2. 数据读取与处理
# ==========================================
def load_and_process(file_path):
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return None

    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except:
        return None

    # 1. 修复列名
    if 'PPCS' not in df.columns:
        if len(df.columns) > 0: df.rename(columns={df.columns[0]: 'PPCS'}, inplace=True)

    # 2. 剔除无关列
    df = df.drop(columns=['序号', 'cens', '术后24h因低氧未拔管'], errors='ignore')

    # 3. 清理字符串空格
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    # 4. 映射数值
    if df['PPCS'].dtype == 'object':
        df['PPCS'] = df['PPCS'].map({'PPCS异常组': 1, 'PPCS正常组': 0})
    if '性别' in df.columns and df['性别'].dtype == 'object':
        df['性别'] = df['性别'].map({'男性': 1, '女性': 0})

    # 5. 【关键】剔除空值 (SHAP 对 NaN 非常敏感)
    df = df.dropna()

    # 6. 【关键】重命名为英文特征 (这样画出来的图就是英文的)
    df = df.rename(columns=col_map_en)

    return df


print("Loading data...")
df = load_and_process(train_file)
if df is None: exit()

# ==========================================
# 3. 模型训练
# ==========================================
# 准备数据
target_col = 'PPCS'
X = df.drop(columns=[target_col])
y = df[target_col]

# 划分 (为了计算 SHAP，我们需要一个测试集或全集，这里用测试集演示泛化性)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 数据平衡 (只对训练集做)
train_data = pd.concat([X_train, y_train], axis=1)
maj = train_data[train_data[target_col] == 0]
min_cls = train_data[train_data[target_col] == 1]
min_upsampled = resample(min_cls, replace=True, n_samples=len(maj), random_state=42)
train_bal = pd.concat([maj, min_upsampled])
X_train_bal = train_bal.drop(target_col, axis=1)
y_train_bal = train_bal[target_col]

# 训练模型 (Gradient Boosting 是最适合 SHAP 的树模型之一)
# 注意：这里我们使用原始数据训练，不进行 StandardScaler 标准化
# 这样 SHAP 图的横坐标（特征值）才是真实的临床数值（如年龄=70岁），而不是归一化后的数字
print("Training Gradient Boosting Model...")
model = GradientBoostingClassifier(random_state=42, max_depth=3, learning_rate=0.05, n_estimators=100)
model.fit(X_train_bal, y_train_bal)

# ==========================================
# 4. SHAP 分析与绘图
# ==========================================
print("Calculating SHAP values (this might take a moment)...")

# 创建解释器
explainer = shap.TreeExplainer(model)

# 计算 SHAP 值 (使用测试集 X_test)
shap_values = explainer.shap_values(X_test)

# 兼容性处理：如果是二分类，shap_values 可能是 list，取第二个（正类）
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

# --- 图 1: SHAP Beeswarm Plot (参考您上传的风格) ---
print("Generating Beeswarm Plot...")
plt.figure(figsize=(10, 8), dpi=300)

# plot_type="dot" 就是蜂群图风格
# show=False 允许我们后续保存图片
shap.summary_plot(shap_vals, X_test, plot_type="dot", show=False)

# 调整字体和标题
plt.title("SHAP Summary Plot", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("SHAP value (impact on model output)", fontsize=12)
# 强制调整布局防止文字被切掉
plt.tight_layout()
# 保存
plt.savefig('SHAP_Beeswarm_English.png', dpi=300, bbox_inches='tight')
print("-> Saved: SHAP_Beeswarm_English.png")
plt.close()

# --- 图 2: SHAP Bar Plot (柱状图) ---
print("Generating Bar Plot...")
plt.figure(figsize=(10, 8), dpi=300)

shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False)

plt.title("Feature Importance (SHAP)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("mean(|SHAP value|) (average impact on model output density)", fontsize=12)
plt.tight_layout()
plt.savefig('SHAP_BarPlot_English.png', dpi=300, bbox_inches='tight')
print("-> Saved: SHAP_BarPlot_English.png")
plt.close()

print("\nDone! High-resolution English SHAP plots have been generated.")