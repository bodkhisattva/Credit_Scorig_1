from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Check


@dataclass(frozen=True)
class DataPrepConfig:
    raw_path: str
    target_col: str = "default.payment.next.month"
    # Возрастные бины
    age_bins: tuple[int, ...] = (18, 25, 35, 45, 55, 65, 120)


def load_raw_uci_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def primary_cleaning(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()

    # Нормализуем имена колонок
    df.columns = [c.strip() for c in df.columns]

    # Удалим дубликаты по ID
    if "ID" in df.columns:
        df = df.drop_duplicates(subset=["ID"])

    # Приводим типы 
    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce").astype("Int64")

    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])

    return df


def feature_engineering(df: pd.DataFrame, config: DataPrepConfig) -> pd.DataFrame:
    df = df.copy()

    # Агрегаты по истории платежей
    pay_amt_cols = [c for c in df.columns if c.startswith("PAY_AMT")]
    if pay_amt_cols:
        df["pay_amt_sum_6m"] = df[pay_amt_cols].sum(axis=1)
        df["pay_amt_mean_6m"] = df[pay_amt_cols].mean(axis=1)
        df["pay_amt_std_6m"] = df[pay_amt_cols].std(axis=1).fillna(0.0)

    # Агрегаты по биллам 
    bill_amt_cols = [c for c in df.columns if c.startswith("BILL_AMT")]
    if bill_amt_cols:
        df["bill_amt_sum_6m"] = df[bill_amt_cols].sum(axis=1)
        df["bill_amt_mean_6m"] = df[bill_amt_cols].mean(axis=1)
        df["bill_amt_std_6m"] = df[bill_amt_cols].std(axis=1).fillna(0.0)

    # Ratio: сколько платит относительно билла 
    if pay_amt_cols and bill_amt_cols:
        df["pay_to_bill_ratio_6m"] = (
            df["pay_amt_sum_6m"] / (df["bill_amt_sum_6m"].abs() + 1.0)
        )

    # Биннинг возраста
    if "AGE" in df.columns:
        bins = list(config.age_bins)
        labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]
        df["age_bin"] = pd.cut(df["AGE"], bins=bins, labels=labels, right=False)
        df["age_bin_ord"] = df["age_bin"].cat.codes.astype("int64")

    # Утилизация лимита по последнему месяцу
    if "LIMIT_BAL" in df.columns and "BILL_AMT1" in df.columns:
        df["utilization_m1"] = df["BILL_AMT1"].abs() / (df["LIMIT_BAL"].abs() + 1.0)

    return df


def build_input_schema(target_col: str) -> pa.DataFrameSchema:
    pay_status_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    columns: dict[str, pa.Column] = {
        "ID": pa.Column(int, Check.ge(1), nullable=False, coerce=True),
        "LIMIT_BAL": pa.Column(float, Check.ge(1), Check.le(2_000_000), nullable=False, coerce=True),
        "AGE": pa.Column(int, Check.ge(18), Check.le(120), nullable=False, coerce=True),
        target_col: pa.Column(int, Check.isin([0, 1]), nullable=False, coerce=True),
    }

    # Категориальные поля 
    for col, allowed in [
        ("SEX", [1, 2]),
        ("EDUCATION", [0, 1, 2, 3, 4, 5, 6]),
        ("MARRIAGE", [0, 1, 2, 3]),
    ]:
        if col in ("SEX", "EDUCATION", "MARRIAGE"):
            columns[col] = pa.Column(int, Check.isin(allowed), nullable=False, coerce=True)

    # У статуса оплаты более широкий диапазон значений
    for c in pay_status_cols:
        columns[c] = pa.Column(int, Check.ge(-2), Check.le(10), nullable=False, coerce=True)

    numeric_soft = {
        # BILL
        "BILL_AMT1": (-1_000_000, 10_000_000),
        "BILL_AMT2": (-1_000_000, 10_000_000),
        "BILL_AMT3": (-1_000_000, 10_000_000),
        "BILL_AMT4": (-1_000_000, 10_000_000),
        "BILL_AMT5": (-1_000_000, 10_000_000),
        "BILL_AMT6": (-1_000_000, 10_000_000),
        # PAY
        "PAY_AMT1": (0, 10_000_000),
        "PAY_AMT2": (0, 10_000_000),
        "PAY_AMT3": (0, 10_000_000),
        "PAY_AMT4": (0, 10_000_000),
        "PAY_AMT5": (0, 10_000_000),
        "PAY_AMT6": (0, 10_000_000),
    }
    for col, (lo, hi) in numeric_soft.items():
        columns[col] = pa.Column(float, Check.ge(lo), Check.le(hi), nullable=False, coerce=True)

    schema = pa.DataFrameSchema(
        columns=columns,
        strict=False,  
        checks=[
            # Проверка уникальности ID
            Check(lambda s: s["ID"].is_unique, error="ID должен быть уникальным"),
        ],
    )
    return schema

def validate(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    return schema.validate(df, lazy=False)

def run_stage_1(config: DataPrepConfig) -> pd.DataFrame:
    raw = load_raw_uci_csv(config.raw_path)
    cleaned = primary_cleaning(raw, target_col=config.target_col)
    fe = feature_engineering(cleaned, config=config)

    schema = build_input_schema(target_col=config.target_col)
    validated = validate(fe, schema=schema)

    return validated


if __name__ == "__main__":
    cfg = DataPrepConfig(raw_path="data/raw/ ")
    df_out = run_stage_1(cfg)
    print(df_out.shape)
    print(df_out.head(3))