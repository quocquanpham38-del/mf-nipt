# -*- coding: utf-8 -*-
"""
main.py — MF-NIPT Pipeline Controller

Pipeline Steps:
1. ensure_data_split()
2. Statistical Analysis (MFRM-1)
3. BMI Single-Factor Timing Model (MFRM-2)
4. Multi-Factor Timing Model (MFRM-3)
5. Abnormality Risk Prediction Model (MFAP)
"""

import os
import traceback

from preprocessing import ensure_data_split
import mfrm_stat_model
import mfrm_timing_single
import mfrm_timing_multi
import mfap_risk_model


# 全局参数
TRAIN_FRAC = 0.8
RANDOM_STATE = 42


def run_step(step_name, func):
    """统一的模块运行封装"""
    print(f"\n>>> [START] {step_name}")

    try:
        result = func()
        print(f">>> [DONE]  {step_name}")
        return result

    except Exception as e:
        print(f"\n!!! [ERROR] {step_name} 运行失败")
        traceback.print_exc()
        return None


def main():
    print("\n===============================")
    print("     MF-NIPT Pipeline Start     ")
    print("===============================")

    # -----------------------------------------------
    # 1. 数据准备（train/test 自动划分）
    # -----------------------------------------------
    print("\n>>> Checking dataset status...")
    ensure_data_split(train_frac=TRAIN_FRAC, random_state=RANDOM_STATE)
    print(">>> Dataset ready: train.csv & test.csv are prepared.")

    # -----------------------------------------------
    # 2. 统计建模（MFRM-1）
    # -----------------------------------------------
    stat_res = run_step(
        "MFRM-1 Statistical Relationship Modeling",
        mfrm_stat_model.run_stat_analysis
    )

    # -----------------------------------------------
    # 3. 单因子（BMI）时机模型（MFRM-2）
    # -----------------------------------------------
    single_res = run_step(
        "MFRM-2 BMI Single-Factor Timing Model",
        mfrm_timing_single.run_mfrm_timing_single
    )

    # -----------------------------------------------
    # 4. 多因子时机模型（MFRM-3）
    # -----------------------------------------------
    multi_res = run_step(
        "MFRM-3 Multi-Factor Timing Model",
        mfrm_timing_multi.run_mfrm_timing_multi
    )

    # -----------------------------------------------
    # 5. 异常风险预测模型（MFAP）
    # -----------------------------------------------
    risk_res = run_step(
        "MFAP Abnormality Risk Prediction Model",
        mfap_risk_model.run_mfap_risk_model
    )

    print("\n===============================")
    print("     MF-NIPT Pipeline Done     ")
    print("===============================")

    return {
        "statistical_analysis": stat_res,
        "timing_single_factor": single_res,
        "timing_multi_factor": multi_res,
        "risk_prediction": risk_res
    }


if __name__ == "__main__":
    main()
