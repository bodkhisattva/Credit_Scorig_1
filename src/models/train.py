import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    RocCurveDisplay,
)

from src.models.pipeline import create_pipeline

TARGET_COL = "default.payment.next.month"
RANDOM_STATE = 42


def infer_feature_types(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    if "ID" in X.columns:
        X = X.drop(columns=["ID"])

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    return numeric_features, categorical_features


def main() -> None:
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    y_train = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    X_train = train_df.drop(columns=[TARGET_COL])
    X_test = test_df.drop(columns=[TARGET_COL])

    if "ID" in X_train.columns:
        X_train = X_train.drop(columns=["ID"])
        X_test = X_test.drop(columns=["ID"])

    numeric_features, categorical_features = infer_feature_types(train_df, TARGET_COL)

    # MLflow 
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Credit_Default_Prediction")

    with mlflow.start_run():
        pipeline = create_pipeline(numeric_features, categorical_features)
        param_distributions = {
            "classifier__n_estimators": [100, 200, 400],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__max_depth": [2, 3, 4],
            "classifier__subsample": [0.7, 0.85, 1.0],
        }

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=20,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
        )

        search.fit(X_train, y_train)

        model = search.best_estimator_

        # Логирование результатов
        mlflow.log_param("model_type", "GradientBoostingClassifier")
        mlflow.log_param("search_type", "RandomizedSearchCV")
        mlflow.log_param("n_iter", 20)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("n_train", len(train_df))
        mlflow.log_param("n_test", len(test_df))

        mlflow.log_metric("cv_best_roc_auc", search.best_score_)

        for param_name, param_value in search.best_params_.items():
            mlflow.log_param(param_name, param_value)

       # Оценка на тесте
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }

        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # ROC кривая
        Path("artifacts").mkdir(exist_ok=True)

        plt.figure()
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title("ROC curve (test)")
        plt.tight_layout()

        roc_path = Path("artifacts/roc_curve.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()

        mlflow.log_artifact(str(roc_path), artifact_path="plots")

        metrics_path = Path("artifacts/metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        Path("models").mkdir(exist_ok=True)

        model_path = Path("models/credit_default_model.joblib")
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(model, artifact_path="model")

        print("Best CV ROC-AUC:", search.best_score_)
        print("Test metrics:", metrics)
        print("Saved model:", model_path)


if __name__ == "__main__":
    main()
