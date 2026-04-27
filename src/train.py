import argparse
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import evaluate_models
from src.models import train_models

TARGET_COLUMN = "Class"
MODEL_OUTPUT_PATH = Path("models/fraud_pipeline.pkl")


def load_dataset(data_path: str) -> pd.DataFrame:
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'."
        )

    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    return df


def main(data_path: str) -> None:
    df = load_dataset(data_path)

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    models = train_models(X_train_res, y_train_res)
    results = evaluate_models(models, X_test, y_test)

    print("\nModel comparison with optimal threshold:\n")
    print(results.sort_values(by="pr_auc", ascending=False).to_string(index=False))

    best_row = results.sort_values(by="pr_auc", ascending=False).iloc[0]
    best_model = models[best_row["model"]]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", best_model),
    ])

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "threshold": best_row["threshold"],
    }, MODEL_OUTPUT_PATH)

    print(f"\nSaved pipeline with threshold: {best_row['threshold']:.4f}")
    print(f"Path: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    args = parser.parse_args()

    main(args.data_path)
