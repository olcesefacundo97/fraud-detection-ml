import argparse
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.evaluation import evaluate_models
from src.models import train_models


TARGET_COLUMN = "Class"
MODEL_OUTPUT_PATH = Path("models/fraud_model.pkl")


def load_dataset(data_path: str) -> pd.DataFrame:
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'.\n\n"
            "Download the Kaggle Credit Card Fraud Detection dataset and place it at:\n"
            "  data/creditcard.csv\n\n"
            "Then run:\n"
            "  python -m src.train --data-path data/creditcard.csv"
        )

    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found in the dataset. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def main(data_path: str) -> None:
    df = load_dataset(data_path)

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    models = train_models(X_train_res, y_train_res)
    results = evaluate_models(models, X_test, y_test)

    print("\nModel comparison:\n")
    print(results.sort_values(by="pr_auc", ascending=False).to_string(index=False))

    best_model_name = results.sort_values(by="pr_auc", ascending=False).iloc[0]["model"]
    best_model = models[best_model_name]

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_OUTPUT_PATH)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument("--data-path", type=str, required=True)
    args = parser.parse_args()

    main(args.data_path)
