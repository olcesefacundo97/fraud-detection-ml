import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

from src.models import train_models
from src.evaluation import evaluate_models


def main(data_path):
    df = pd.read_csv(data_path)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # Handle imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    models = train_models(X_train_res, y_train_res)
    results = evaluate_models(models, X_test, y_test)

    print(results)

    # Save best model (example: Random Forest)
    best_model = models["Random Forest"]
    joblib.dump(best_model, "models/fraud_model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    args = parser.parse_args()

    main(args.data_path)
