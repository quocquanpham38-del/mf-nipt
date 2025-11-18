# -*- coding: utf-8 -*-
"""
mfrm_timing_single.py
- 单因子（BMI）建模 + 贪婪分组目标规划（来自 ques2.py）
- 使用 train.csv 进行训练（GBDT），并在 test.csv 上评估
- 基于训练集的 g* 进行贪婪分组（并在 test 上验证覆盖率）
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, brier_score_loss

from preprocessing import load_train, load_test, compute_g_star

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "mfrm_timing_single")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 字体
possible_fonts = ['SimHei', 'Microsoft YaHei', 'STSong', 'Arial Unicode MS']
chinese_font = None
for font_name in possible_fonts:
    try:
        if font_name in [f.name for f in fm.fontManager.ttflist]:
            chinese_font = fm.FontProperties(fname=fm.findfont(fm.FontProperties(family=font_name)))
            break
    except:
        continue
if chinese_font is None:
    chinese_font = fm.FontProperties()

# 工具函数
def weeks_str(days):
    if pd.isna(days):
        return ""
    days = int(round(days))
    w, d = divmod(days, 7)
    return f"{w}w+{d}"

def trimester_weight(days):
    if pd.isna(days):
        return 1.0
    if days <= 84:
        return 1.0
    elif 85 <= days <= 189:
        return 2.0
    elif days >= 196:
        return 4.0
    else:
        return 2.0

# 贪婪分组实现（基于训练集 g*）
def group_loss_at_t(g_star_array, t, gamma=3.0, delta=0.5):
    if len(g_star_array) == 0:
        return 0.0
    gstar = np.asarray(g_star_array, dtype=float)
    late = np.maximum(t - gstar, 0.0)
    early = np.maximum(gstar - t, 0.0)
    w = trimester_weight(t)
    loss = gamma * w * late + delta * early
    return float(np.nanmean(loss))

T_candidates = np.arange(70, 196, 1, dtype=int)

def best_t_for_indices(gstar_array, gamma=3.0, delta=0.5):
    best_t, best_L = None, np.inf
    for t in T_candidates:
        L = group_loss_at_t(gstar_array, t, gamma=gamma, delta=delta)
        if L < best_L:
            best_t, best_L = t, L
    return best_t, best_L

def greedy_partition_on_df(df_gstar, K=5, min_group_size=20, gamma=3.0, delta=0.5, min_gain=1e-4):
    """
    df_gstar expected columns: ['孕妇BMI', 'g_star']
    返回 intervals (list of (l,r)), info dict, steps, total_losses
    """
    df_part = df_gstar.sort_values("孕妇BMI").reset_index(drop=True)
    n = len(df_part)
    if n == 0:
        return [], {}, [], []
    intervals = [(0, n)]
    interval_info = {}
    t0, L0 = best_t_for_indices(df_part.loc[0:n-1, "g_star"].values, gamma=gamma, delta=delta)
    interval_info[(0, n)] = {"t": t0, "loss": L0}
    total_losses = [L0]
    steps = [1]

    while len(intervals) < K:
        best_gain = 0.0
        best_move = None
        for (l, r) in intervals:
            size = r - l
            if size < 2 * min_group_size:
                continue
            for m in range(l + min_group_size, r - min_group_size + 1):
                base_L = interval_info[(l, r)]["loss"]
                tL, LL = best_t_for_indices(df_part.loc[l:m-1, "g_star"].values, gamma=gamma, delta=delta)
                tR, LR = best_t_for_indices(df_part.loc[m:r-1, "g_star"].values, gamma=gamma, delta=delta)
                new_avg = (LL * (m - l) + LR * (r - m)) / (r - l)
                gain = base_L - new_avg
                if gain > best_gain:
                    best_gain = gain
                    best_move = (l, m, r, tL, LL, tR, LR)
        if (best_move is None) or (best_gain < min_gain):
            break
        l, m, r, tL, LL, tR, LR = best_move
        intervals.remove((l, r))
        intervals.extend([(l, m), (m, r)])
        intervals.sort(key=lambda x: x[0])
        interval_info.pop((l, r), None)
        interval_info[(l, m)] = {"t": tL, "loss": LL}
        interval_info[(m, r)] = {"t": tR, "loss": LR}
        tot_loss = 0.0
        total_n = 0
        for (a, b) in intervals:
            L_val = interval_info[(a, b)]["loss"]
            cnt = b - a
            tot_loss += L_val * cnt
            total_n += cnt
        avg_loss = tot_loss / total_n if total_n > 0 else np.nan
        total_losses.append(avg_loss)
        steps.append(len(intervals))
    # 返回基于排序后的 df_part，注意 intervals 是基于 df_part 索引
    return intervals, interval_info, steps, total_losses, df_part

# 主流程
def run_mfrm_timing_single(save_models=False):
    train = load_train()
    test = load_test()

    # (1) 分类模型（仅 BMI）
    df_clf_train = train[["孕妇BMI", "达标"]].dropna()
    df_clf_train = df_clf_train[(df_clf_train["孕妇BMI"] > 10) & (df_clf_train["孕妇BMI"] < 60)].reset_index(drop=True)

    df_clf_test = test[["孕妇BMI", "达标"]].dropna()
    df_clf_test = df_clf_test[(df_clf_test["孕妇BMI"] > 10) & (df_clf_test["孕妇BMI"] < 60)].reset_index(drop=True)

    X_train = df_clf_train[["孕妇BMI"]].values
    y_train = df_clf_train["达标"].values
    X_test = df_clf_test[["孕妇BMI"]].values
    y_test = df_clf_test["达标"].values

    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_prob_train = clf.predict_proba(X_train)[:, 1]
    y_prob_test = clf.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= 0.5).astype(int)

    # 评估在 test 集上
    try:
        auc = roc_auc_score(y_test, y_prob_test)
    except:
        auc = np.nan
    brier = brier_score_loss(y_test, y_prob_test)
    print(f"[single] Test AUC: {auc:.3f}, Brier: {brier:.4f}")

    # 保存 ROC 图片（test）
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], '--', color='gray', lw=1)
    plt.xlabel('假正率', fontproperties=chinese_font)
    plt.ylabel('真正率', fontproperties=chinese_font)
    plt.title('ROC 曲线（仅 BMI）', fontproperties=chinese_font, fontsize=14)
    plt.legend(prop=chinese_font)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve_test.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # (2) 构建 g* 数据（基于 train）
    df_train_full = train.copy()
    df_gstar_train = compute_g_star(df_train_full)
    # 过滤合理范围
    df_gstar_train = df_gstar_train[
        (df_gstar_train["孕妇BMI"] > 10) &
        (df_gstar_train["孕妇BMI"] < 60) &
        (df_gstar_train["g_star"] >= 56) &
        (df_gstar_train["g_star"] <= 224)
    ].reset_index(drop=True)
    print(f"[single] 用于分组优化的训练孕妇数: {len(df_gstar_train)}")
    df_gstar_train.to_csv(os.path.join(RESULTS_DIR, "df_gstar_train.csv"), index=False, encoding="utf-8-sig")

    # (3) 贪婪分组（基于 train 的 df_gstar_train）
    intervals, interval_info, steps, total_losses, df_part = greedy_partition_on_df(df_gstar_train, K=5, min_group_size=20)
    # 保存收敛图
    if len(steps) > 0:
        plt.figure(figsize=(7, 5))
        plt.plot(steps, total_losses, marker='o', linewidth=2, markersize=6)
        plt.xlabel("分组数量", fontproperties=chinese_font)
        plt.ylabel("目标函数值（加权平均损失）", fontproperties=chinese_font)
        plt.title("目标规划：贪婪分组的迭代收敛图（仅 BMI）", fontproperties=chinese_font, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "greedy_convergence.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # (4) 输出分组详情（并用 test 验证覆盖率）
    rows = []
    for (l, r) in intervals:
        seg = df_part.loc[l:r-1]
        if len(seg) == 0:
            continue
        t = interval_info[(l, r)]["t"]
        L = interval_info[(l, r)]["loss"]
        cover_train = float(np.mean(t >= seg["g_star"].values))
        # 用 test 上的孕妇按照 BMI 属于该区间计算覆盖率（使用 test 的 g*）
        df_gstar_test = compute_g_star(test)
        df_gstar_test = df_gstar_test.dropna(subset=["孕妇BMI", "g_star"])
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
    res_table.to_csv(os.path.join(RESULTS_DIR, "bmi_groups_detail.csv"), index=False, encoding="utf-8-sig")
    print(f"✅ 已保存分组详情至 {os.path.join(RESULTS_DIR, 'bmi_groups_detail.csv')}")
    return {
        "clf": clf,
        "roc_auc_test": auc,
        "brier_test": brier,
        "groups": res_table
    }

if __name__ == "__main__":
    run_mfrm_timing_single()
