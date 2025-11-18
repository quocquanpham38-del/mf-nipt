# -*- coding: utf-8 -*-
"""
preprocessing.py
- 统一从 Hugging Face 加载原始数据（或从本地 raw.csv）
- 做清洗（列名修整、孕周解析、BMI 计算、one-hot 等）
- 生成 g*（每个孕妇最早异常孕周）
- 划分 train/test 并保存到 data/
- 暴露加载接口：load_raw(), load_train(), load_test(), ensure_data_split()
"""
import os
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

HF_REPO = "cumcm-dataset/CUMCM-2025c-dataset"

# ----------------------------
# 工具：孕周字符串转天数
# ----------------------------
def convert_gestational_age(ga_str):
    if pd.isna(ga_str):
        return np.nan
    s = str(ga_str).strip().lower().replace(" ", "")
    try:
        if "w+" in s:
            w, d = s.split("w+")
            return int(w) * 7 + int(d)
        elif "w" in s:
            w = s.replace("w", "")
            return int(w) * 7
        else:
            if s.isdigit():
                return int(s)
    except Exception:
        pass
    return np.nan

# ----------------------------
# 加载原始数据（优先本地 raw.csv）
# ----------------------------
def load_raw(local_path=None, hf_repo=HF_REPO, split="train"):
    """
    返回原始 DataFrame（列名已 strip）
    优先读取 local_path（raw.csv），若不存在则从 Hugging Face 下载
    """
    if local_path is None:
        local_path = os.path.join(DATA_DIR, "raw.csv")
    if os.path.exists(local_path):
        df = pd.read_csv(local_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        return df
    # 否则尝试从 Hugging Face 加载
    print("✅ 正在从 Hugging Face 加载数据...")
    ds = load_dataset(hf_repo, split=split)
    df = ds.to_pandas()
    df.columns = df.columns.str.strip()
    # 保存一份 raw.csv 方便复现
    df.to_csv(local_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存 raw data -> {local_path}")
    return df

# ----------------------------
# 主要预处理函数（统一）
# ----------------------------
def preprocess(df):
    df = df.copy()
    # 标准列名 strip（已在 load_raw 中做一次）
    df.columns = df.columns.str.strip()

    # 1) 孕周（天）
    if "Gestational Week at Detection" in df.columns:
        df["检测孕周_days"] = df["Gestational Week at Detection"].apply(convert_gestational_age)
    else:
        df["检测孕周_days"] = np.nan

    # 2) BMI（若未提供或 NaN 则按 Weight/Height 计算）
    if "Pregnant Woman's BMI (BMI: Body Mass Index)" in df.columns:
        df["孕妇BMI"] = df["Pregnant Woman's BMI (BMI: Body Mass Index)"]
    else:
        # 需要 Height & Weight 列
        if "Height" in df.columns and "Weight" in df.columns:
            df["孕妇BMI"] = df["Weight"] / ((df["Height"] / 100.0) ** 2)
        else:
            df["孕妇BMI"] = np.nan

    # 3) 目标标签：达标 / 非达标（原始 Yes/No -> 1/0）
    if "Fetal Health Status (Yes/No)" in df.columns:
        df["达标"] = df["Fetal Health Status (Yes/No)"].map({"Yes": 1, "No": 0})
    else:
        df["达标"] = np.nan

    # 4) Pregnant Woman Code 名称归一
    if "Pregnant Woman Code" not in df.columns and "Pregnant Woman Code " in df.columns:
        df.rename(columns={"Pregnant Woman Code ": "Pregnant Woman Code"}, inplace=True)

    # 5) one-hot 处理占位（实际建模时会在各模块按需处理）
    # 这里只保留原始列，不强行 one-hot

    # 6) 清理列空白
    df = df.replace({np.inf: np.nan, -np.inf: np.nan})

    return df

# ----------------------------
# 计算 g*（每个孕妇最早达标=1 的检测孕周）
# ----------------------------
def compute_g_star(df, id_col="Pregnant Woman Code"):
    """
    基于检测记录（多次检测），找到每位孕妇达标（达标==1）的最早检测天数 g*
    返回 DataFrame: ['Pregnant Woman Code', 'g_star', '孕妇BMI'（中位数）]
    """
    df_valid = df.dropna(subset=[id_col, "检测孕周_days", "达标"])
    # 只保留达标==1 的记录并取最早检测天
    try:
        gstar_per_id = (
            df_valid[df_valid["达标"] == 1]
            .sort_values([id_col, "检测孕周_days"])
            .groupby(id_col)["检测孕周_days"]
            .first()
        )
    except Exception:
        # 若没有达标列或其他问题，返回空
        return pd.DataFrame(columns=[id_col, "g_star", "孕妇BMI"])
    bmi_per_id = df_valid.groupby(id_col)["孕妇BMI"].median()
    df_gstar = pd.DataFrame({"孕妇BMI": bmi_per_id, "g_star": gstar_per_id}).reset_index()
    df_gstar = df_gstar.dropna(subset=["孕妇BMI", "g_star"])
    return df_gstar

# ----------------------------
# 划分 train/test 并保存
# ----------------------------
def ensure_data_split(df=None, train_frac=0.8, random_state=42, force=False):
    """
    如果 data/train.csv 与 data/test.csv 不存在，或 force=True，则进行一次划分并保存
    返回 train_df, test_df
    """
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    if not force and os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path, encoding="utf-8-sig")
        test_df = pd.read_csv(test_path, encoding="utf-8-sig")
        return train_df, test_df

    if df is None:
        df = load_raw()
    df = preprocess(df)

    # 做一次随机划分：注意 stratify 可选（如果标签分布不极端，直接随机）
    # 如果存在达标列，尝试按达标分层
    stratify_col = None
    if "达标" in df.columns and df["达标"].notna().any():
        stratify_col = df["达标"]
    try:
        train_df, test_df = train_test_split(df, test_size=1 - train_frac, random_state=random_state, stratify=stratify_col)
    except Exception:
        train_df, test_df = train_test_split(df, test_size=1 - train_frac, random_state=random_state)

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已生成并保存 train/test: {train_path}, {test_path}")
    return train_df, test_df

# ----------------------------
# 加载接口
# ----------------------------
def load_train():
    p = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(p):
        ensure_data_split()
    return pd.read_csv(p, encoding="utf-8-sig")

def load_test():
    p = os.path.join(DATA_DIR, "test.csv")
    if not os.path.exists(p):
        ensure_data_split()
    return pd.read_csv(p, encoding="utf-8-sig")

def load_full():
    raw_p = os.path.join(DATA_DIR, "raw.csv")
    if os.path.exists(raw_p):
        df = pd.read_csv(raw_p, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        return preprocess(df)
    else:
        df = load_raw()
        return preprocess(df)

# ----------------------------
# 如果直接运行此文件，会执行一次 split 并保存
# ----------------------------
if __name__ == "__main__":
    print(">>> preprocessing.py run: ensure data split")
    ensure_data_split()
