# -*- coding: utf-8 -*-
"""
mfap_risk_model.py
- 非整倍体风险预测（ques4 的重构）
- 训练集上做 SMOTE（仅训练集），训练决策树并用 PSO 搜索超参数（最大化 F1）
- 在测试集上给出最终评估（AUC/F1/Confusion Matrix）并保存风险评分
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    classification_report, f1_score
)

from preprocessing import load_train, load_test

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "mfap_risk")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 字体
CHINESE_FONT_NAME = 'Microsoft YaHei'
try:
    chinese_font = FontProperties(family=CHINESE_FONT_NAME)
except Exception:
    chinese_font = FontProperties()

plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def prepare_features(df):
    """
    返回 X (DataFrame) 和 y (Series) 供建模
    - one-hot IVF/Number of Pregnancies
    - 如果包含 'Gestational Week at Detection' -> 保底转换（若 preprocessing 已做则可忽略）
    """
    dfc = df.copy()
    # 处理孕周列名
    if "Gestational Week at Detection" in dfc.columns and "检测孕周_days" not in dfc.columns:
        def gest_to_days(s):
            if pd.isna(s):
                return np.nan
            s = str(s).strip().lower()
            if "w+" in s:
                try:
                    w,d = s.split("w+")
                    return int(w)*7 + int(d)
                except:
                    return np.nan
            elif "w" in s:
                try:
                    w = s.replace("w", "")
                    return int(w)*7
                except:
                    return np.nan
            else:
                return np.nan
        dfc["检测孕周_days"] = dfc["Gestational Week at Detection"].apply(gest_to_days)
    # 特征列：选取大部分数值列（与 ques4 保持一致）
    feature_cols = [
        "Age","Height","Weight","Number of Detection Blood Draws","检测孕周_days",
        "孕妇BMI","Number of Raw Reads","Proportion Aligned to Reference Genome",
        "Proportion of Duplicate Reads","Number of Uniquely Aligned Reads","GC Content",
        "Z-Score of Chromosome 13","Z-Score of Chromosome 18","Z-Score of Chromosome 21",
        "Z-Score of Chromosome X","Z-Score of Chromosome Y","Concentration of Chromosome X",
        "GC Content of Chromosome 13","GC Content of Chromosome 18","GC Content of Chromosome 21",
        "Proportion of Filtered Reads","Number of Deliveries"
    ]
    feature_cols = [c for c in feature_cols if c in dfc.columns]
    X = dfc[feature_cols].copy()
    
    # 添加分类变量到特征中（如果存在但不在feature_cols中）
    additional_cat_cols = []
    if "IVF Pregnancy (IVF: In Vitro Fertilization)" in dfc.columns and "IVF Pregnancy (IVF: In Vitro Fertilization)" not in feature_cols:
        additional_cat_cols.append("IVF Pregnancy (IVF: In Vitro Fertilization)")
    if "Number of Pregnancies" in dfc.columns and "Number of Pregnancies" not in feature_cols:
        additional_cat_cols.append("Number of Pregnancies")
    
    # 如果有额外的分类列，需要添加到X中
    if additional_cat_cols:
        X = pd.concat([X, dfc[additional_cat_cols]], axis=1)
    
    # one-hot
    cat_cols = [c for c in ["IVF Pregnancy (IVF: In Vitro Fertilization)", "Number of Pregnancies"] if c in X.columns]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        # 确保特征列名一致性
        X.columns = X.columns.str.replace("IVF Pregnancy \(IVF: In Vitro Fertilization\)_IVF\(In Vitro Fertilization\)", 
                                          "IVF Pregnancy (IVF: In Vitro Fertilization)_IVF Pregnancy (IVF: In Vitro Fertilization)", 
                                          regex=True)
    # y：非整倍体标签（No -> 1）
    if "Fetal Health Status (Yes/No)" in dfc.columns:
        y = dfc["Fetal Health Status (Yes/No)"].map({"Yes": 0, "No": 1})
    elif "达标" in dfc.columns:
        # 若达标存在：达标==1 -> 正常 -> 非整倍体 = 1 - 达标
        y = 1 - dfc["达标"]
    else:
        raise KeyError("缺失标签列: Fetal Health Status (Yes/No)")
    return X, y

# PSO 参数搜索（离散）
def pso_optimize(X_train, y_train, X_val, y_val, n_particles=12, n_iters=25, bounds=None):
    """
    搜索 DecisionTree 的 max_depth, min_samples_split, min_samples_leaf（整数）
    bounds: np.array([[low, high], ...]) shape (3,2)
    返回最佳参数 (max_depth, min_samples_split, min_samples_leaf)
    """
    if bounds is None:
        bounds = np.array([[2, 20],[2, 20],[1, 10]])
    dim = bounds.shape[0]
    # 初始化
    pos = np.random.uniform(bounds[:,0], bounds[:,1], (n_particles, dim))
    vel = np.zeros_like(pos)
    pbest_pos = pos.copy()
    pbest_val = np.zeros(n_particles)
    # 评估初值
    for i in range(n_particles):
        md, mss, msl = int(pos[i,0]), int(pos[i,1]), int(pos[i,2])
        clf = DecisionTreeClassifier(max_depth=md, min_samples_split=mss, min_samples_leaf=msl, random_state=42)
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            pbest_val[i] = f1_score(y_val, y_pred)
        except:
            pbest_val[i] = 0.0
    gbest_idx = int(np.argmax(pbest_val))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]
    w, c1, c2 = 0.7, 1.5, 1.5
    for it in range(n_iters):
        for i in range(n_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            vel[i] = w * vel[i] + c1 * r1 * (pbest_pos[i] - pos[i]) + c2 * r2 * (gbest_pos - pos[i])
            pos[i] = pos[i] + vel[i]
            pos[i] = np.clip(pos[i], bounds[:,0], bounds[:,1])
            md, mss, msl = int(pos[i,0]), int(pos[i,1]), int(pos[i,2])
            clf = DecisionTreeClassifier(max_depth=md, min_samples_split=mss, min_samples_leaf=msl, random_state=42)
            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)
                score = f1_score(y_val, y_pred)
            except:
                score = 0.0
            if score > pbest_val[i]:
                pbest_val[i] = score
                pbest_pos[i] = pos[i].copy()
            if score > gbest_val:
                gbest_val = score
                gbest_pos = pos[i].copy()
        print(f"[PSO] iter {it+1}/{n_iters}, best F1 = {gbest_val:.4f}")
    return int(round(gbest_pos[0])), int(round(gbest_pos[1])), int(round(gbest_pos[2])), gbest_val

# 主流程
def run_mfap_risk_model():
    train = load_train()
    test = load_test()
    
    # 预处理：修复IVF列值不一致问题
    ivf_col_name = "IVF Pregnancy (IVF: In Vitro Fertilization)"
    ivf_value = "IVF(In Vitro Fertilization)"
    if ivf_col_name in train.columns:
        train[ivf_col_name] = train[ivf_col_name].replace(ivf_value, ivf_col_name)
    if ivf_col_name in test.columns:
        test[ivf_col_name] = test[ivf_col_name].replace(ivf_value, ivf_col_name)

    X_train_raw, y_train_raw = prepare_features(train)
    X_test_raw, y_test_raw = prepare_features(test)
    
    # 确保训练集和测试集的特征列一致
    # 获取训练集的特征列名
    train_columns = X_train_raw.columns.tolist()
    
    # 对测试集进行相同的列处理
    for col in train_columns:
        if col not in X_test_raw.columns:
            X_test_raw[col] = 0  # 添加缺失的列并填充默认值
    
    # 确保列顺序一致
    X_test_raw = X_test_raw[train_columns]

    # 丢弃含 NaN 的行
    train_mask = X_train_raw.notna().all(axis=1) & (~y_train_raw.isna())
    X_train_raw = X_train_raw.loc[train_mask]
    y_train_raw = y_train_raw.loc[train_mask]

    test_mask = X_test_raw.notna().all(axis=1) & (~y_test_raw.isna())
    X_test_raw = X_test_raw.loc[test_mask]
    y_test_raw = y_test_raw.loc[test_mask]

    print("训练集样本数（原始）:", len(X_train_raw))
    print("测试集样本数（原始）:", len(X_test_raw))

    # SMOTE（只用于训练）
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_raw, y_train_raw)
    print("过采样后类别分布：")
    print(pd.Series(y_res).value_counts())

    # 将过采样后的训练集再划分为 train/val 供 PSO 优化
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

    # PSO 优化决策树超参
    md, mss, msl, best_f1 = pso_optimize(X_tr, y_tr, X_val, y_val, n_particles=12, n_iters=25)

    print("PSO 最优参数：", md, mss, msl, "best_f1:", best_f1)

    # 使用最优参在过采样完整训练集上训练最终模型
    final_clf = DecisionTreeClassifier(max_depth=md, min_samples_split=mss, min_samples_leaf=msl, random_state=42)
    final_clf.fit(X_res, y_res)

    # 在测试集上评估
    y_prob_test = final_clf.predict_proba(X_test_raw)[:, 1]
    y_pred_test = final_clf.predict(X_test_raw)
    try:
        auc = roc_auc_score(y_test_raw, y_prob_test)
    except:
        auc = np.nan
    f1 = f1_score(y_test_raw, y_pred_test)
    cm = confusion_matrix(y_test_raw, y_pred_test)
    print("测试集 AUC:", auc, "F1:", f1)
    print("混淆矩阵:\n", cm)
    # 保存 ROC
    fpr, tpr, _ = roc_curve(y_test_raw, y_prob_test)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate", fontproperties=chinese_font)
    plt.ylabel("True Positive Rate", fontproperties=chinese_font)
    plt.title("ROC Curve - 决策树（测试集）", fontproperties=chinese_font)
    plt.legend(loc="lower right", prop=chinese_font)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_test.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 风险评分（在原始未过采样的 X_test_raw 上）
    risk_score_full = final_clf.predict_proba(X_test_raw)[:, 1]
    risk_level_full = pd.cut(risk_score_full, bins=[-np.inf, 0.3, 0.7, np.inf], labels=["低风险","中风险","高风险"])
    results_df = pd.DataFrame({
        "y_true": y_test_raw.values,
        "y_pred": y_pred_test,
        "risk_score": risk_score_full,
        "risk_level": risk_level_full
    })
    results_df.to_csv(os.path.join(RESULTS_DIR, "risk_assessment_results_test.csv"), index=False, encoding="utf-8-sig")
    print(f"✅ 已保存风险评估结果至 {os.path.join(RESULTS_DIR, 'risk_assessment_results_test.csv')}")
    return {
        "final_clf": final_clf,
        "auc_test": auc,
        "f1_test": f1,
        "confusion_matrix": cm,
        "risk_df": results_df
    }

if __name__ == "__main__":
    run_mfap_risk_model()
