from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans


MODEL_PATH = Path(__file__).resolve().parent / "automl_model.pkl"


@dataclass
class AutoMLMetadata:
    task_type: str                    # "regression" | "classification" | "clustering" | "unsupervised"
    target_column: Optional[str]
    feature_columns: list
    n_samples: int
    n_features: int
    model_type: str                   # e.g. "RandomForestRegressor"
    metrics: Dict[str, Any]


def infer_task_type(df: pd.DataFrame, target_column: Optional[str]) -> str:
    """
    Simple heuristic:
    - If target provided:
        - numeric → regression
        - categorical → classification
    - If no target → unsupervised (clustering + anomaly detection)
    """
    if target_column is None:
        return "unsupervised"

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    if pd.api.types.is_numeric_dtype(df[target_column]):
        return "regression"
    else:
        return "classification"


def build_preprocessor(df: pd.DataFrame, target_column: Optional[str]) -> Tuple[ColumnTransformer, list]:
    if target_column is not None and target_column in df.columns:
        X = df.drop(columns=[target_column])
    else:
        X = df.copy()

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return preprocessor, X.columns.tolist()


def train_supervised(
    df: pd.DataFrame, target_column: str, task_type: str
) -> Tuple[Pipeline, AutoMLMetadata]:
    y = df[target_column]
    X = df.drop(columns=[target_column])

    preprocessor, feature_columns = build_preprocessor(df, target_column)

    if task_type == "regression":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X, y)

    # Simple in-sample metrics (for demo; in real work use train/test split)
    y_pred = pipe.predict(X)

    if task_type == "regression":
        mse = float(mean_squared_error(y, y_pred))
        metrics = {"mse": mse}
    else:
        acc = float(accuracy_score(y, y_pred))
        metrics = {"accuracy": acc}

    meta = AutoMLMetadata(
        task_type=task_type,
        target_column=target_column,
        feature_columns=feature_columns,
        n_samples=int(len(df)),
        n_features=int(len(feature_columns)),
        model_type=type(model).__name__,
        metrics=metrics,
    )

    return pipe, meta


def train_unsupervised(df: pd.DataFrame) -> Tuple[Dict[str, Any], AutoMLMetadata]:
    # No target column → use all numeric features
    X = df.select_dtypes(include=["int64", "float64"]).copy()
    if X.empty:
        raise ValueError("No numeric columns found for unsupervised training.")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    n_clusters = min(5, max(2, X_scaled.shape[0] // 10))  # simple heuristic
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Anomaly detection
    iso = IsolationForest(contamination=0.05, random_state=42)
    anomaly_scores = iso.fit_predict(X_scaled)  # -1 outlier, 1 normal

    # Bundle unsupervised models
    unsupervised_bundle = {
        "scaler": scaler,
        "kmeans": kmeans,
        "isolation_forest": iso,
        "numeric_columns": X.columns.tolist(),
    }

    meta = AutoMLMetadata(
        task_type="unsupervised",
        target_column=None,
        feature_columns=X.columns.tolist(),
        n_samples=int(len(df)),
        n_features=int(X.shape[1]),
        model_type="KMeans+IsolationForest",
        metrics={},
    )

    return unsupervised_bundle, meta


def train_automl(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entrypoint:
    - Detects task type
    - Trains appropriate model(s)
    - Saves to ml/automl_model.pkl
    - Returns metadata summary
    """
    task_type = infer_task_type(df, target_column)

    if task_type in ["regression", "classification"]:
        model, meta = train_supervised(df, target_column, task_type)
        bundle = {
            "kind": "supervised",
            "task_type": task_type,
            "target_column": target_column,
            "model": model,
            "meta": asdict(meta),
        }
    else:
        unsupervised_bundle, meta = train_unsupervised(df)
        bundle = {
            "kind": "unsupervised",
            "task_type": "unsupervised",
            "target_column": None,
            "models": unsupervised_bundle,
            "meta": asdict(meta),
        }

    # Save model bundle
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    return {
        "model_path": str(MODEL_PATH),
        "meta": bundle["meta"],
    }


def load_automl_model() -> Optional[Dict[str, Any]]:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None
