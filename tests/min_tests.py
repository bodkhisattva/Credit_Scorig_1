from pathlib import Path
import pandas as pd
from src.prepare import main as prepare_main

def test_prepare_creates_parquet(tmp_path, monkeypatch):
    prepare_main()
    out_path = Path("data/prepared/prepared.parquet")
    assert out_path.exists()

    df = pd.read_parquet(out_path)
    assert len(df) > 0
