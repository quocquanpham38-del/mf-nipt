# -*- coding: utf-8 -*-
"""
main.py
- 统一调度整个 MF-NIPT pipeline
- 1) preprocessing.ensure_data_split()
- 2) run statistical analysis (mfrm_stat_model)
- 3) run timing single / multi
- 4) run risk model
"""
import os
from preprocessing import ensure_data_split
import mfrm_stat_model
import mfrm_timing_single
import mfrm_timing_multi
import mfap_risk_model

# 全局参数（可在未来提取到 config.py）
TRAIN_FRAC = 0.8
RANDOM_STATE = 42

def main():
    print(">>> MF-NIPT pipeline start")
    # 1) 数据划分（若已存在将跳过）
    ensure_data_split(train_frac=TRAIN_FRAC, random_state=RANDOM_STATE)
    # 2) 统计分析（全量数据）
    print(">>> Running statistical analysis (MFRM-1)")
    stat_res = mfrm_stat_model.run_stat_analysis()
    # 3) 单因子时机模型（BMI）
    print(">>> Running single-factor timing model (MFRM-2)")
    single_res = mfrm_timing_single.run_mfrm_timing_single()
    # 4) 多因素时机模型（MFRM-3）
    print(">>> Running multi-factor timing model (MFRM-3)")
    multi_res = mfrm_timing_multi.run_mfrm_timing_multi()
    # 5) 风险模型（MFAP）
    print(">>> Running anomaly risk model (MFAP)")
    risk_res = mfap_risk_model.run_mfap_risk_model()
    print(">>> MF-NIPT pipeline finished")
    # 可在此增加结果汇总、生成 README 报告等
    return {
        "stat": stat_res,
        "single": single_res,
        "multi": multi_res,
        "risk": risk_res
    }

if __name__ == "__main__":
    main()
