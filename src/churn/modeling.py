from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COLUMN = "customerID"


def load_and_clean(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    required = {ID_COLUMN, TARGET, "TotalCharges", "tenure", "MonthlyCharges"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data[ID_COLUMN].duplicated().any():
        raise ValueError("Customer IDs must be unique")

    data = data.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    target = data[TARGET].map({"No": 0, "Yes": 1})
    if target.isna().any():
        raise ValueError("Churn must contain only Yes/No values")
    features = data.drop(columns=[ID_COLUMN, TARGET])
    return features, target.astype(int)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def make_models(features: pd.DataFrame, random_state: int = 42) -> dict[str, Pipeline]:
    estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=2_000, class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=160, learning_rate=0.04, max_depth=2, random_state=random_state
        ),
    }
    return {
        name: Pipeline([("prep", build_preprocessor(features)), ("model", estimator)])
        for name, estimator in estimators.items()
    }


def top_fraction_metrics(y_true: pd.Series | np.ndarray, scores: np.ndarray, fraction: float = 0.20) -> dict[str, float]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    y = np.asarray(y_true, dtype=int)
    k = max(1, int(np.ceil(len(y) * fraction)))
    selected = np.argsort(scores)[::-1][:k]
    captured = int(y[selected].sum())
    total_positive = int(y.sum())
    precision = captured / k
    recall = captured / total_positive if total_positive else 0.0
    base_rate = float(y.mean())
    return {
        "customers_targeted": k,
        "churners_captured": captured,
        "precision": precision,
        "recall": recall,
        "lift": precision / base_rate if base_rate else 0.0,
    }

