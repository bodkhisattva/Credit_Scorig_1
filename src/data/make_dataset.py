from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.build_features import build_features

RAW_PATH = "data/raw/UCI_Credit_Card.csv"
TARGET_COL = "default.payment.next.month"


def load_raw_uci_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "ID" in df.columns:
        df = df.drop_duplicates(subset=["ID"])

    # Приведение таргета
    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype("Int64")

    for col in df.columns:
        if col == TARGET_COL:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])

    return df


def main() -> None:
    raw = load_raw_uci_csv(RAW_PATH)
    cleaned = basic_clean(raw)
    featured = build_features(cleaned)

    if TARGET_COL not in featured.columns:
        raise ValueError(f"Target '{TARGET_COL}' not found.")

    train_df, test_df = train_test_split(
        featured,
        test_size=0.2,
        random_state=42,
        stratify=featured[TARGET_COL].astype(int),
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print(f"Saved: {out_dir/'train.csv'} ({len(train_df)})")
    print(f"Saved: {out_dir/'test.csv'} ({len(test_df)})")


if __name__ == "__main__":
    main()