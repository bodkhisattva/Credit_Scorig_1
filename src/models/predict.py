import joblib
import pandas as pd

MODEL_PATH = "models/credit_default_model.joblib"


def predict_one(features: dict) -> dict:
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([features])
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= 0.5)
    return {"class": pred, "probability": proba}