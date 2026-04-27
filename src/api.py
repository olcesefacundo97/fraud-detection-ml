from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

MODEL_PATH = Path("models/fraud_pipeline.pkl")
EXPECTED_FEATURES = 30

app = FastAPI(
    title="Fraud Detection API",
    description="Production-oriented API for fraud risk scoring using a trained ML pipeline.",
    version="1.0.0",
)

model_artifact = None


class Transaction(BaseModel):
    features: List[float] = Field(
        ...,
        description="Transaction features in the same order used during training. Expected: 30 features.",
        min_length=EXPECTED_FEATURES,
        max_length=EXPECTED_FEATURES,
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, features):
        if len(features) != EXPECTED_FEATURES:
            raise ValueError(f"Expected {EXPECTED_FEATURES} features, received {len(features)}.")
        return features


class BatchTransactions(BaseModel):
    transactions: List[Transaction] = Field(..., min_length=1)


@app.on_event("startup")
def load_model():
    global model_artifact

    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model artifact not found. Train the model first with: "
            "python -m src.train --data-path data/creditcard.csv"
        )

    model_artifact = joblib.load(MODEL_PATH)

    if "pipeline" not in model_artifact or "threshold" not in model_artifact:
        raise RuntimeError("Invalid model artifact. Expected keys: 'pipeline' and 'threshold'.")


@app.get("/")
def root():
    return {
        "service": "Fraud Detection API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model_artifact is not None else "unavailable",
        "model_loaded": model_artifact is not None,
        "model_path": str(MODEL_PATH),
    }


def score_features(features: List[float]):
    if model_artifact is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    pipeline = model_artifact["pipeline"]
    threshold = float(model_artifact["threshold"])

    data = np.array(features).reshape(1, -1)
    fraud_probability = float(pipeline.predict_proba(data)[0][1])
    fraud_prediction = int(fraud_probability >= threshold)

    return {
        "fraud_prediction": fraud_prediction,
        "fraud_probability": fraud_probability,
        "threshold": threshold,
        "risk_level": "high" if fraud_prediction == 1 else "low",
    }


@app.post("/predict")
def predict(transaction: Transaction):
    return score_features(transaction.features)


@app.post("/predict-batch")
def predict_batch(batch: BatchTransactions):
    predictions = [score_features(transaction.features) for transaction in batch.transactions]
    return {
        "count": len(predictions),
        "predictions": predictions,
    }
