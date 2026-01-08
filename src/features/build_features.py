import numpy as np
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    pay_amt_cols = [c for c in df.columns if c.startswith("PAY_AMT")]
    if pay_amt_cols:
        df["pay_amt_sum_6m"] = df[pay_amt_cols].sum(axis=1)
        df["pay_amt_mean_6m"] = df[pay_amt_cols].mean(axis=1)
        df["pay_amt_std_6m"] = df[pay_amt_cols].std(axis=1).fillna(0.0)

    bill_amt_cols = [c for c in df.columns if c.startswith("BILL_AMT")]
    if bill_amt_cols:
        df["bill_amt_sum_6m"] = df[bill_amt_cols].sum(axis=1)
        df["bill_amt_mean_6m"] = df[bill_amt_cols].mean(axis=1)
        df["bill_amt_std_6m"] = df[bill_amt_cols].std(axis=1).fillna(0.0)

    if pay_amt_cols and bill_amt_cols:
        df["pay_to_bill_ratio_6m"] = df["pay_amt_sum_6m"] / (df["bill_amt_sum_6m"].abs() + 1.0)

    if "AGE" in df.columns:
        bins = [18, 25, 35, 45, 55, 65, 120]
        labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]
        df["age_bin"] = pd.cut(df["AGE"], bins=bins, labels=labels, right=False)

    if "LIMIT_BAL" in df.columns and "BILL_AMT1" in df.columns:
        df["utilization_m1"] = df["BILL_AMT1"].abs() / (df["LIMIT_BAL"].abs() + 1.0)

    return df
