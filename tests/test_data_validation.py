import pandas as pd
import pytest

from src.data_preparation import DataPrepConfig, load_raw_uci_csv, primary_cleaning, feature_engineering, build_input_schema, validate


def test_validation_fails_on_anomalies():
    cfg = DataPrepConfig(raw_path="data/raw/UCI_Credit_Card.csv")
    raw = load_raw_uci_csv(cfg.raw_path)
    df = feature_engineering(primary_cleaning(raw, cfg.target_col), cfg)

    schema = build_input_schema(target_col=cfg.target_col)

    # Проверка датафрейма
    _ = validate(df, schema)

    # 1) Аномалия: возраст вне диапазона
    bad = df.copy()
    bad.loc[bad.index[0], "AGE"] = 5

    with pytest.raises(Exception):
        _ = validate(bad, schema)

    # 2) Аномалия: неожиданный класс в таргете
    bad2 = df.copy()
    bad2.loc[bad2.index[0], cfg.target_col] = 2

    with pytest.raises(Exception):
        _ = validate(bad2, schema)

    # 3) Аномалия: null в обязательном поле
    bad3 = df.copy()
    bad3.loc[bad3.index[0], "LIMIT_BAL"] = None

    with pytest.raises(Exception):
        _ = validate(bad3, schema)