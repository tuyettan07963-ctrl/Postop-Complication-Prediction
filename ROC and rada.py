import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import pi
import os

# Sklearn Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# ==========================================
# 1. Setup: Fonts and File Path
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# 文件路径：请确保文件名与您的数据集一致
file_path = "修正后的ppcs数据集.xlsx"

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
if not os.path.exists(file_path):
    print(f"[Error] File not found: {file_path}")
    # 模拟环境或手动检查时可跳过
else:
    df = pd.read_excel(file_path, engine='openpyxl')
    cols_to_drop = ['序号', 'cens', '术后24h因低氧未拔管']
    df_corrected = df.drop(columns=cols_to_drop, errors='ignore')

    # 分离特征与目标变量
    X = df_corrected.drop(columns=['PPCS'], errors='ignore')
    # 假设目标变量映射：PPCS异常组为1，PPCS正常组为0
    y = df_corrected['PPCS'].map({'PPCS异常组': 1, 'PPCS正常组': 0})

    # ==========================================
    # 3. Data Splitting & Balancing
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 手动过采样 (针对需要平衡数据的模型)
    train_data = pd.concat([X_train, y_train], axis=1)
    majority = train_data[train_data['PPCS'] == 0]
    minority = train_data[train_data['PPCS'] == 1]
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    train_balanced = pd.concat([majority, minority_upsampled])

    X_train_bal = train_balanced.drop('PPCS', axis=1)
    y_train_bal = train_balanced['PPCS']

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_bal_scaled = scaler.transform(X_train_bal)

    # ==========================================
    # 4. Define 9 Models (1 Regression + 8 ML)
    # ==========================================
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=0.01, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, max_depth=3, class_weight='balanced'),
        'Extra Trees': ExtraTreesClassifier(random_state=42, max_depth=3, class_weight='balanced'),
        'SVM': SVC(probability=True, random_state=42, C=0.5, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=3, class_weight='balanced'),
        'KNN': KNeighborsClassifier(n_neighbors=15),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, max_depth=2, n_estimators=100),
        'AdaBoost': AdaBoostClassifier(random_state=42, n_estimators=30)
    }

    # ==========================================
    # 5. Training, Evaluation & Plotting
    # ==========================================
    metrics_cols = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC', 'F1']


    def get_metrics(y_true, y_pred, y_prob):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        auc_val = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred)
        return [acc, sens, spec, auc_val, f1]


    fig = plt.figure(figsize=(20, 18), dpi=300)
    ax_roc_train = fig.add_subplot(2, 2, 1)
    ax_roc_test = fig.add_subplot(2, 2, 2)
    ax_radar_train = fig.add_subplot(2, 2, 3, polar=True)
    ax_radar_test = fig.add_subplot(2, 2, 4, polar=True)

    colors = matplotlib.colormaps['tab20']  # 9个模型建议使用tab20以获得更好的区分度

    for i, (name, model) in enumerate(models.items()):
        color = colors(i)

        # 训练策略选择
        if name in ['Logistic Regression', 'Random Forest', 'SVM', 'Decision Tree', 'Extra Trees']:
            model.fit(X_train_scaled, y_train)
            curr_X_tr, curr_y_tr = X_train_scaled, y_train
        else:
            model.fit(X_train_bal_scaled, y_train_bal)
            curr_X_tr, curr_y_tr = X_train_bal_scaled, y_train_bal

        # 预测
        y_tr_prob = model.predict_proba(curr_X_tr)[:, 1]
        y_te_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_tr_pred = model.predict(curr_X_tr)
        y_te_pred = model.predict(X_test_scaled)

        # ROC 绘制
        fpr_tr, tpr_tr, _ = roc_curve(curr_y_tr, y_tr_prob)
        ax_roc_train.plot(fpr_tr, tpr_tr, color=color, lw=2, label=f'{name} (AUC={auc(fpr_tr, tpr_tr):.3f})')

        fpr_te, tpr_te, _ = roc_curve(y_test, y_te_prob)
        ax_roc_test.plot(fpr_te, tpr_te, color=color, lw=2, label=f'{name} (AUC={auc(fpr_te, tpr_te):.3f})')

        # 指标计算
        m_train = get_metrics(curr_y_tr, y_tr_pred, y_tr_prob)
        m_test = get_metrics(y_test, y_te_pred, y_te_prob)

        # 雷达图绘制
        N = len(metrics_cols)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        ax_radar_train.plot(angles, m_train + m_train[:1], color=color, linewidth=2, label=name)
        ax_radar_train.fill(angles, m_train + m_train[:1], color=color, alpha=0.05)

        ax_radar_test.plot(angles, m_test + m_test[:1], color=color, linewidth=2, label=name)
        ax_radar_test.fill(angles, m_test + m_test[:1], color=color, alpha=0.05)

    # 样式美化
    for ax, title in zip([ax_roc_train, ax_roc_test], ['A. Training Set ROC', 'B. Validation Set ROC']):
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.legend(loc="lower right", fontsize='small', ncol=2)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')

    for ax, title in zip([ax_radar_train, ax_radar_test], ['C. Training Metrics Radar', 'D. Validation Metrics Radar']):
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_cols, fontsize=12)
        ax.set_title(title, y=1.1, fontsize=15, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize='small')

    plt.tight_layout()
    plt.savefig('Model_Comparison_ROC_Radar.png', dpi=300, bbox_inches='tight')
    print("Success: Final_English_Model_Evaluation.png has been saved.")