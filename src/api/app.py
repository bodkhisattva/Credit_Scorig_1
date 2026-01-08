from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path("models/credit_default_model.joblib")

app = FastAPI(title="Credit Default Prediction API")

model = None

class ClientData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int

    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Credit Default Prediction API is working!"}

@app.post("/predict")
def predict(data: ClientData):
    input_data = pd.DataFrame([data.model_dump()])
    proba = float(model.predict_proba(input_data)[0][1])
    pred = int(proba >= 0.5)
    return {"default_prediction": pred, "default_probability": proba}