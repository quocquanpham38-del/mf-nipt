# -*- coding: utf-8 -*-
"""
mfrm_timing_multi.py
- 多变量建模 + 基于多变量特征的贪婪分组（来自 ques3.py）
- 训练/测试划分已由 preprocessing.py 处理
- 多因素模型训练（GBDT），在 test 上评估 AUC/Brier
- 基于训练集的 df_gstar（按孕妇聚合）沿 BMI 排序做贪婪分组（以保持临床可解释性）
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, confusion_matrix

from preprocessing import load_train, load_test, compute_g_star

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "mfrm_timing_multi")
os.makedirs(RESULTS_DIR, exist_ok=True)

possible_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
chinese_font = None
for font_name in possible_fonts:
    try:
        if font_name in [f.name for f in fm.fontManager.ttflist]:
            chinese_font = fm.FontProperties(family=font_name)
            break
    except:
        continue
if chinese_font is None:
    chinese_font = fm.FontProperties()

# 特征列（与 ques3.py 对齐，但仅保留在训练数据中存在的列）
BASE_FEATURES = [
    "Age",
    "Number of Detection Blood Draws",
    "检测孕周_days",
    "孕妇BMI",
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
    "Number of Deliveries"
]

def weeks_str(days):
    if pd.isna(days):
        return ""
    days = int(round(days))
    w, d = divmod(days, 7)
    return f"{w}w+{d}"

# 贪婪分组函数与 single 中的相似（抽取出来复用）
def group_loss_at_t(g_star_array, t, gamma=3.0, delta=0.5):
    if len(g_star_array) == 0:
        return 0.0
    gstar = np.asarray(g_star_array, dtype=float)
    late = np.maximum(t - gstar, 0.0)
    early = np.maximum(gstar - t, 0.0)
    # trimester weight
    if t <= 84:
        w = 1.0
    elif 85 <= t <= 189:
        w = 2.0
    else:
        w = 4.0
    loss = gamma * w * late + delta * early
    return float(np.nanmean(loss))

T_candidates = np.arange(70, 196, 1, dtype=int)

def best_t_for_group(g_star_array, gamma=3.0, delta=0.5):
    best_t, best_L = None, np.inf
    for t in T_candidates:
        L = group_loss_at_t(g_star_array, t, gamma=gamma, delta=delta)
        if L < best_L:
            best_t, best_L = t, L
    return best_t, best_L

def greedy_partition(df_gstar, K=5, min_group_size=20, min_gain=1e-4):
    df_part = df_gstar.sort_values("孕妇BMI").reset_index(drop=True)
    n = len(df_part)
    if n == 0:
        return [], {}, [], []
    intervals = [(0, n)]
    info = {}
    t0, L0 = best_t_for_group(df_part.loc[0:n-1, "g_star"].values)
    info[(0, n)] = {"t": t0, "loss": L0}
    total_losses = [L0]
    steps = [1]
    while len(intervals) < K:
        best_gain, best_move = 0.0, None
        for (l, r) in intervals:
            if r - l < 2 * min_group_size:
                continue
            for m in range(l + min_group_size, r - min_group_size + 1):
                base_L = info[(l, r)]["loss"]
                tL, LL = best_t_for_group(df_part.loc[l:m-1, "g_star"].values)
                tR, LR = best_t_for_group(df_part.loc[m:r-1, "g_star"].values)
                new_avg = (LL * (m - l) + LR * (r - m)) / (r - l)
                gain = base_L - new_avg
                if gain > best_gain:
                    best_gain = gain
                    best_move = (l, m, r, tL, LL, tR, LR)
        if best_move is None or best_gain < min_gain:
            break
        l, m, r, tL, LL, tR, LR = best_move
        intervals.remove((l, r))
        intervals.extend([(l, m), (m, r)])
        intervals.sort(key=lambda x: x[0])
        info.pop((l, r), None)
        info[(l, m)] = {"t": tL, "loss": LL}
        info[(m, r)] = {"t": tR, "loss": LR}
        tot_loss = sum(info[(a, b)]["loss"] * (b - a) for (a, b) in intervals)
        total_n = sum(b - a for (a, b) in intervals)
        total_losses.append(tot_loss / total_n if total_n > 0 else np.nan)
        steps.append(len(intervals))
    return intervals, info, steps, total_losses, df_part

# 主流程
def run_mfrm_timing_multi():
    train = load_train()
    test = load_test()

    # 预处理：修复IVF列值不一致问题
    ivf_col_name = "IVF Pregnancy (IVF: In Vitro Fertilization)"
    if ivf_col_name in train.columns:
        # 将数据中的"IVF(In Vitro Fertilization)"替换为"IVF Pregnancy (IVF: In Vitro Fertilization)"
        train[ivf_col_name] = train[ivf_col_name].replace("IVF(In Vitro Fertilization)", ivf_col_name)
    if ivf_col_name in test.columns:
        test[ivf_col_name] = test[ivf_col_name].replace("IVF(In Vitro Fertilization)", ivf_col_name)

    # 1) 确定实际可用特征
    avail = [c for c in BASE_FEATURES if c in train.columns]
    # 准备训练矩阵
    X_train = train[avail].copy()
    X_test = test[avail].copy()
    
    # 支撑性处理：one-hot IVF & Number of Pregnancies 若存在
    # 先检查这些列是否在avail中，如果不在需要添加进去
    additional_cat_cols = []
    if "IVF Pregnancy (IVF: In Vitro Fertilization)" in train.columns and "IVF Pregnancy (IVF: In Vitro Fertilization)" not in avail:
        additional_cat_cols.append("IVF Pregnancy (IVF: In Vitro Fertilization)")
    if "Number of Pregnancies" in train.columns and "Number of Pregnancies" not in avail:
        additional_cat_cols.append("Number of Pregnancies")
    
    # 如果有额外的分类列，需要添加到X_train和X_test中
    if additional_cat_cols:
        X_train = pd.concat([X_train, train[additional_cat_cols]], axis=1)
        X_test = pd.concat([X_test, test[additional_cat_cols]], axis=1)
    
    # 确定需要one-hot编码的分类列
    cat_cols = [c for c in ["IVF Pregnancy (IVF: In Vitro Fertilization)", "Number of Pregnancies"] if c in X_train.columns]
    if cat_cols:
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
        # 对齐列
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # 目标 y
    y_train = train["达标"].astype(int).values
    y_test = test["达标"].astype(int).values

    # 丢弃含 NaN 的样本（简单方案）
    mask_train = X_train.notna().all(axis=1) & (~pd.isna(y_train))
    X_train = X_train.loc[mask_train]
    y_train = pd.Series(y_train).loc[mask_train].values

    mask_test = X_test.notna().all(axis=1) & (~pd.isna(y_test))
    X_test = X_test.loc[mask_test]
    y_test = pd.Series(y_test).loc[mask_test].values

    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_prob_test = clf.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, y_prob_test)
    except:
        auc = np.nan
    brier = brier_score_loss(y_test, y_prob_test)
    print(f"[multi] Test AUC: {auc:.3f}, Brier: {brier:.4f}")

    # 保存 ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], '--', color='gray', lw=1)
    plt.xlabel('假正率', fontproperties=chinese_font)
    plt.ylabel('真正率', fontproperties=chinese_font)
    plt.title('ROC 曲线（多变量模型）', fontproperties=chinese_font, fontsize=14)
    plt.legend(prop=chinese_font)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve_multivar_test.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2) 准备 df_gstar（训练集按孕妇聚合）
    df_gstar_train = compute_g_star(train)
    df_gstar_train = df_gstar_train[
        (df_gstar_train["孕妇BMI"] > 10) &
        (df_gstar_train["孕妇BMI"] < 60) &
        (df_gstar_train["g_star"] >= 56) &
        (df_gstar_train["g_star"] <= 224)
    ].reset_index(drop=True)
    print(f"[multi] 用于分组优化的训练孕妇数: {len(df_gstar_train)}")
    df_gstar_train.to_csv(os.path.join(RESULTS_DIR, "df_gstar_train_multivar.csv"), index=False, encoding="utf-8-sig")

    # 3) 贪婪分组（沿 BMI 排序）
    intervals, info, steps, total_losses, df_part = greedy_partition(df_gstar_train, K=5, min_group_size=20)
    # 绘制收敛
    if len(steps) > 0:
        plt.figure(figsize=(7, 5))
        plt.plot(steps, total_losses, marker='o', linewidth=2, markersize=6)
        plt.xlabel("分组数量", fontproperties=chinese_font)
        plt.ylabel("目标函数值（加权平均损失）", fontproperties=chinese_font)
        plt.title("目标规划：贪婪分组收敛曲线（多变量模型）", fontproperties=chinese_font, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "greedy_convergence_multivar.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 4) 输出组表并在 test 上验证覆盖率
    df_gstar_test = compute_g_star(test)
    rows = []
    for (l, r) in intervals:
        seg = df_part.loc[l:r-1]
        if len(seg) == 0:
            continue
        t = info[(l, r)]["t"]
        L = info[(l, r)]["loss"]
        cover_train = float(np.mean(t >= seg["g_star"].values))
        mask = (df_gstar_test["孕妇BMI"] >= float(seg["孕妇BMI"].min())) & (df_gstar_test["孕妇BMI"] <= float(seg["孕妇BMI"].max()))
        seg_test = df_gstar_test[mask]
        cover_test = float(np.mean(t >= seg_test["g_star"].values)) if len(seg_test) > 0 else np.nan
        rows.append({
            "BMI_L": float(seg["孕妇BMI"].min()),
            "BMI_U": float(seg["孕妇BMI"].max()),
            "组内样本数": len(seg),
            "推荐NIPT时点_天": int(round(t)),
            "推荐NIPT时点_周表达": weeks_str(t),
            "组内目标函数值": round(L, 3),
            "训练集覆盖率": round(cover_train, 3),
            "测试集覆盖率": round(cover_test, 3),
            "g*_中位数(天)": float(np.nanmedian(seg["g_star"].values)),
        })
    res_table = pd.DataFrame(rows).sort_values("BMI_L").reset_index(drop=True)
    res_table.to_csv(os.path.join(RESULTS_DIR, "bmi_groups_detail_multivar.csv"), index=False, encoding="utf-8-sig")
    print(f"✅ 已保存分组详情至 {os.path.join(RESULTS_DIR, 'bmi_groups_detail_multivar.csv')}")
    return {
        "clf": clf,
        "roc_auc_test": auc,
        "brier_test": brier,
        "groups": res_table
    }

if __name__ == "__main__":
    run_mfrm_timing_multi()
