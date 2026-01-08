from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    RocCurveDisplay,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)

from src.data_prep import DataPrepConfig, run_stage_1


@dataclass(frozen=True)
class MLflowConfig:
    raw_path: str = "data/raw/UCI_Credit_Card.csv"
    target_col: str = "default.payment.next.month"

    test_size: float = 0.2
    random_state: int = 42

    # MLflow
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "pd_model_experiments"

    cv: int = 5
    n_iter: int = 25  # per experiment


def _select_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    if "ID" in X.columns:
        X = X.drop(columns=["ID"])
    return X, y


def _infer_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )


def get_experiments(preprocessor: ColumnTransformer) -> List[Dict[str, Any]]:
    experiments: List[Dict[str, Any]] = []

    # 1) LogisticRegression
    experiments.append(
        dict(
            name="logreg",
            estimator=Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", LogisticRegression(max_iter=2000, solver="liblinear")),
                ]
            ),
            param_distributions={
                "model__penalty": ["l1", "l2"],
                "model__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
                "model__class_weight": [None, "balanced"],
            },
        )
    )

    # 2) GradientBoostingClassifier
    experiments.append(
        dict(
            name="gb",
            estimator=Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", GradientBoostingClassifier()),
                ]
            ),
            param_distributions={
                "model__n_estimators": [100, 200, 400],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.85, 1.0],
            },
        )
    )

    # 3) RandomForestClassifier
    experiments.append(
        dict(
            name="rf",
            estimator=Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
                ]
            ),
            param_distributions={
                "model__n_estimators": [300, 600, 900],
                "model__max_depth": [None, 6, 10, 14],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": [None, "balanced"],
            },
        )
    )

    # 4) ExtraTreesClassifier
    experiments.append(
        dict(
            name="extra_trees",
            estimator=Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", ExtraTreesClassifier(random_state=42, n_jobs=-1)),
                ]
            ),
            param_distributions={
                "model__n_estimators": [300, 600, 900],
                "model__max_depth": [None, 6, 10, 14],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": [None, "balanced"],
            },
        )
    )

    # 5) AdaBoostClassifier
    experiments.append(
        dict(
            name="adaboost",
            estimator=Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", AdaBoostClassifier(random_state=42)),
                ]
            ),
            param_distributions={
                "model__n_estimators": [100, 200, 400, 600],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
            },
        )
    )

    return experiments


def evaluate_and_save_roc(model, X_test: pd.DataFrame, y_test: pd.Series, out_path: Path) -> Dict[str, float]:
    # Скоры
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # ROC график
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC curve (test)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    return metrics


def run_mlflow_experiments(cfg: MLflowConfig) -> None:
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    df = run_stage_1(DataPrepConfig(raw_path=cfg.raw_path, target_col=cfg.target_col))
    X, y = _select_features(df, cfg.target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    num_cols, cat_cols = _infer_columns(X_train)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    experiments = get_experiments(preprocessor)

    for exp in experiments:
        run_name = exp["name"]
        estimator = exp["estimator"]
        param_distributions = exp["param_distributions"]

        with mlflow.start_run(run_name=run_name):
            # Логируем контекст эксперимента
            mlflow.log_param("algorithm", run_name)
            mlflow.log_param("cv", cfg.cv)
            mlflow.log_param("n_iter", cfg.n_iter)
            mlflow.log_param("test_size", cfg.test_size)
            mlflow.log_param("random_state", cfg.random_state)

            # Поиск гиперпараметров (ROC-AUC)
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_distributions,
                n_iter=cfg.n_iter,
                scoring="roc_auc",
                cv=cfg.cv,
                n_jobs=-1,
                refit=True,
                random_state=cfg.random_state,
                verbose=1,
            )

            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            # Логируем лучшие параметры и CV-метрику
            mlflow.log_metric("cv_best_roc_auc", float(search.best_score_))
            for k, v in search.best_params_.items():
                mlflow.log_param(k, v)

            roc_path = Path("artifacts") / f"roc_{run_name}.png"
            test_metrics = evaluate_and_save_roc(best_model, X_test, y_test, roc_path)

            for m_name, m_val in test_metrics.items():
                mlflow.log_metric(f"test_{m_name}", m_val)

            mlflow.log_artifact(str(roc_path), artifact_path="plots")

            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                registered_model_name=None,  
            )

            print(f"[{run_name}] best_cv_roc_auc={search.best_score_:.4f} test={test_metrics}")


if __name__ == "__main__":
    cfg = MLflowConfig(
        raw_path="data/raw/UCI_Credit_Card.csv",
        experiment_name="pd_model_experiments",
        n_iter=25,
        cv=5,
    )
    run_mlflow_experiments(cfg)