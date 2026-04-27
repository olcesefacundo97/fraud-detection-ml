from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_models(X_train, y_train):
    models = {}

    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=10,
        eval_metric="logloss",
        use_label_encoder=False
    )
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb

    return models
