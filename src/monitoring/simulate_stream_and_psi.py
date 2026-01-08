from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests


TARGET_COL_DEFAULT = "default.payment.next.month"


def psi(
    baseline: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    strategy: str = "quantile",
    eps: float = 1e-6,
) -> Tuple[float, np.ndarray]:
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]

    if baseline.size == 0 or current.size == 0:
        return float("nan"), np.array([])

    if strategy == "quantile":
        qs = np.linspace(0, 1, bins + 1)
        edges = np.quantile(baseline, qs)
    elif strategy == "uniform":
        lo, hi = float(np.min(baseline)), float(np.max(baseline))
        edges = np.linspace(lo, hi, bins + 1) if not math.isclose(lo, hi) else np.array([lo, hi])
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    edges = np.unique(edges)
    if edges.size < 3:
        v = float(np.median(baseline))
        edges = np.array([v - 1e-9, v, v + 1e-9])

    b_counts, _ = np.histogram(baseline, bins=edges)
    c_counts, _ = np.histogram(current, bins=edges)

    b_perc = b_counts / max(baseline.size, 1)
    c_perc = c_counts / max(current.size, 1)

    b_perc = np.clip(b_perc, eps, None)
    c_perc = np.clip(c_perc, eps, None)

    psi_val = float(np.sum((c_perc - b_perc) * np.log(c_perc / b_perc)))
    return psi_val, edges


def call_api_predict_one(api_url: str, row: Dict) -> float:
    url = api_url.rstrip("/") + "/predict"
    r = requests.post(url, json=row, timeout=30)
    r.raise_for_status()
    data = r.json()
    return float(data["default_probability"])


def call_api_predict_batch(api_url: str, rows: List[Dict]) -> np.ndarray:
    probs = [call_api_predict_one(api_url, row) for row in rows]
    return np.asarray(probs, dtype=float)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=str, default="data/processed/train.csv")
    parser.add_argument("--test", type=str, default="data/processed/test.csv")

    parser.add_argument("--model", type=str, default="models/credit_default_model.joblib")
    parser.add_argument("--target", type=str, default=TARGET_COL_DEFAULT)

    parser.add_argument("--api", type=str, default="http://localhost:8000")

    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--batches", type=int, default=10)

    # PSI настройки
    parser.add_argument("--psi-bins", type=int, default=10)
    parser.add_argument("--psi-strategy", type=str, default="quantile", choices=["quantile", "uniform"])
    parser.add_argument(
        "--psi-features",
        type=str,
        default="LIMIT_BAL,AGE,pay_to_bill_ratio_6m,utilization_m1",
        help="Comma-separated features to compute PSI for (must exist in processed train/test).",
    )
    args = parser.parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    model_path = Path(args.model)

    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if args.target not in train_df.columns or args.target not in test_df.columns:
        raise ValueError(f"Target col '{args.target}' not found in processed train/test")

    # Разделяем X/y
    y_train = train_df[args.target].astype(int)
    X_train = train_df.drop(columns=[args.target])

    y_test = test_df[args.target].astype(int)
    X_test = test_df.drop(columns=[args.target])

    if "ID" in X_train.columns:
        X_train = X_train.drop(columns=["ID"])
    if "ID" in X_test.columns:
        X_test = X_test.drop(columns=["ID"])

    model = joblib.load(model_path)
    train_proba = model.predict_proba(X_train)[:, 1]

    feature_list = [f.strip() for f in args.psi_features.split(",") if f.strip()]
    feature_list = [f for f in feature_list if f in X_train.columns]

    print(f"API: {args.api}")
    print(f"Train: {train_path} | Test: {test_path} | Model: {model_path}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"PSI bins={args.psi_bins} strategy={args.psi_strategy}")
    print(f"PSI features: {feature_list if feature_list else '(none found)'}")
    print("-" * 80)

    rng = np.random.default_rng(42)
    test_idx = np.array(X_test.index)
    rng.shuffle(test_idx)

    n_batches = min(args.batches, int(math.ceil(len(test_idx) / args.batch_size)))

    def label(v: float) -> str:
        if np.isnan(v):
            return "NA"
        if v < 0.1:
            return "OK"
        if v < 0.25:
            return "WARN"
        return "ALERT"

    for i in range(n_batches):
        batch_idx = test_idx[i * args.batch_size : (i + 1) * args.batch_size]
        X_batch = X_test.loc[batch_idx]

        rows = X_batch.to_dict(orient="records")

        # Predict через API 
        proba_stream = call_api_predict_batch(args.api, rows)

        # PSI по вероятностям
        psi_proba, _ = psi(
            baseline=train_proba,
            current=proba_stream,
            bins=args.psi_bins,
            strategy=args.psi_strategy,
        )

        # PSI по фичам
        feat_psi: Dict[str, float] = {}
        for f in feature_list:
            psi_f, _ = psi(
                baseline=X_train[f].to_numpy(dtype=float),
                current=X_batch[f].to_numpy(dtype=float),
                bins=args.psi_bins,
                strategy=args.psi_strategy,
            )
            feat_psi[f] = psi_f

        print(f"Batch {i+1}/{n_batches} | n={len(X_batch)}")
        print(f"  PSI(proba) = {psi_proba:.4f} [{label(psi_proba)}]")
        for f, v in feat_psi.items():
            print(f"  PSI({f}) = {v:.4f} [{label(v)}]")
        print("-" * 80)


if __name__ == "__main__":
    main()