from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from pathlib import Path

app = FastAPI(title="Fraud Detection API")

MODEL_PATH = Path("models/fraud_model.pkl")


class Transaction(BaseModel):
    features: list[float]


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run training first:\n"
            "python -m src.train --data-path data/creditcard.csv"
        )
    model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array(transaction.features).reshape(1, -1)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": float(probability)
    }
