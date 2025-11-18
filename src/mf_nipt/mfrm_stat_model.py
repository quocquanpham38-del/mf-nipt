# -*- coding: utf-8 -*-
"""
mfrm_stat_model.py
- 对全量数据做统计分析（来自 ques1.py）
- OLS 回归、VIF、多重共线性、Spearman 相关热图、clustermap
- 不参与 train/test 划分（使用 full 数据）
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.font_manager import FontProperties

from preprocessing import load_full

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "mfrm_stat")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 字体设置（避免中文乱码）
CHINESE_FONT_NAME = 'Microsoft YaHei'
try:
    chinese_font = FontProperties(family=CHINESE_FONT_NAME)
except Exception:
    chinese_font = FontProperties()

plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 特征列（按照 ques1.py）
FEATURE_COLS = [
    "Age",
    "Height",
    "Weight",
    "Number of Detection Blood Draws",
    "Gestational Week at Detection",
    "Pregnant Woman's BMI (BMI: Body Mass Index)",
    "Number of Raw Reads",
    "Proportion Aligned to Reference Genome",
    "Proportion of Duplicate Reads",
    "Number of Uniquely Aligned Reads",
    "GC Content",
    "Z-Score of Chromosome 13",
    "Z-Score of Chromosome 18",
    "Z-Score of Chromosome 21",
    "Z-Score of Chromosome X",
    "Z-Score of Chromosome Y",
    "Concentration of Chromosome X",
    "GC Content of Chromosome 13",
    "GC Content of Chromosome 18",
    "GC Content of Chromosome 21",
    "Proportion of Filtered Reads",
    "Number of Deliveries",
    "IVF Pregnancy (IVF: In Vitro Fertilization)",
    "Number of Pregnancies"
]

def run_stat_analysis():
    df = load_full()
    
    # 预处理：修复IVF列值不一致问题
    ivf_col_name = "IVF Pregnancy (IVF: In Vitro Fertilization)"
    if ivf_col_name in df.columns:
        # 将数据中的"IVF(In Vitro Fertilization)"替换为"IVF Pregnancy (IVF: In Vitro Fertilization)"
        df[ivf_col_name] = df[ivf_col_name].replace("IVF(In Vitro Fertilization)", ivf_col_name)

    # 目标变量 y
    if "Fetal Health Status (Yes/No)" not in df.columns:
        raise KeyError("缺失列: Fetal Health Status (Yes/No)")
    y = df["Fetal Health Status (Yes/No)"].map({"Yes": 1, "No": 0})

    # 生成 X（严格选取 FEATURE_COLS 中存在的列）
    avail_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[avail_cols].copy()

    # 孕周转换（若存在）
    if "Gestational Week at Detection" in X.columns:
        # 转换为天数（如果已经在 preprocessing 做过，这里再保底处理）
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
        X["Gestational_Week_days"] = X["Gestational Week at Detection"].apply(gest_to_days)
        X = X.drop(columns=["Gestational Week at Detection"])

    # 简单 one-hot（IVF, Number of Pregnancies）
    cat_cols = [c for c in ["IVF Pregnancy (IVF: In Vitro Fertilization)", "Number of Pregnancies"] if c in X.columns]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # 丢弃样本缺失的行
    valid_idx = y.notna() & X.notna().all(axis=1)
    y_clean = y[valid_idx]
    X_clean = X[valid_idx]

    X_clean = X_clean.astype(float)
    y_clean = y_clean.astype(float)

    # 常数项
    Xc = sm.add_constant(X_clean)
    model = sm.OLS(y_clean, Xc).fit()
    # 保存回归摘要
    summary_txt = model.summary().as_text()
    with open(os.path.join(RESULTS_DIR, "ols_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_txt)

    # VIF
    vif_df = pd.DataFrame({
        "feature": Xc.columns,
        "VIF": [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]
    }).sort_values("VIF", ascending=False)
    vif_df.to_csv(os.path.join(RESULTS_DIR, "vif.csv"), index=False, encoding="utf-8-sig")

    print("✅ OLS summary and VIF saved to results/mfrm_stat/")

    # 显著系数条形图
    summary_df = pd.DataFrame({'coef': model.params, 'pval': model.pvalues})
    significant = summary_df[(summary_df['pval'] < 0.05) & (summary_df.index != 'const')]
    if not significant.empty:
        coef_abs = significant['coef'].abs().sort_values(ascending=True)
        plt.figure(figsize=(8, max(4, len(coef_abs) * 0.4)))
        coef_abs.plot(kind='barh', color='steelblue')
        plt.title('显著变量对胎儿健康状态的影响（|回归系数|）', fontproperties=chinese_font, fontsize=14)
        plt.xlabel('绝对回归系数', fontproperties=chinese_font, fontsize=12)
        plt.xticks(fontproperties=chinese_font)
        plt.yticks(fontproperties=chinese_font)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "significant_coef.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Spearman 相关性热力图（X + y）
    corr_data = X_clean.copy()
    corr_data['Fetal_Health_Status'] = y_clean
    corr_matrix = corr_data.corr(method='spearman')
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, cbar=True)
    plt.title('斯皮尔曼相关性热力图（自变量 + 胎儿健康状态）', fontproperties=chinese_font, fontsize=16)
    plt.xticks(rotation=45, ha='right', fontproperties=chinese_font)
    plt.yticks(fontproperties=chinese_font)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "spearman_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Clustermap
    corr_clean = corr_matrix.replace([np.inf, -np.inf], np.nan).dropna(how='all').dropna(axis=1, how='all')
    if corr_clean.shape[0] > 1 and corr_clean.shape[1] > 1:
        g = sns.clustermap(corr_clean, cmap='coolwarm', center=0, linewidths=0.5, figsize=(16, 14))
        g.fig.suptitle('斯皮尔曼层次聚类热力图（自变量 + 胎儿健康状态）', fontproperties=chinese_font, fontsize=16, y=1.02)
        plt.tight_layout()
        g.savefig(os.path.join(RESULTS_DIR, "clustermap.png"), dpi=300, bbox_inches='tight')
        plt.close()

    print("✅ 统计分析图像保存在 results/mfrm_stat/")
    return {
        "ols_model": model,
        "vif": vif_df,
        "corr": corr_matrix
    }

if __name__ == "__main__":
    run_stat_analysis()
