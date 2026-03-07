from __future__ import annotations

from typing import Any, Dict

import joblib
import pandas as pd

from agent_core import PredictiveMaintenanceAgent
from features import apply_feature_pipeline

MODEL_PATH = "model.pkl"
LEAKAGE_COLS = {"UDI", "TWF", "HDF", "PWF", "OSF", "RNF"}
LABEL_COLS = ("label", "Machine failure", "failure")

# Columns that serve as row identifiers / sort keys — not useful as features
_SORT_CANDIDATES = ("UDI", "date", "timestamp", "datetime", "time", "Date")
# Columns that can be used for group-wise rolling features
_GROUP_CANDIDATES = ("Type", "device", "machine", "unit", "asset", "sensor_id")


def load_artifact(model_path: str = MODEL_PATH) -> Dict[str, Any]:
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise ValueError("Model artifact must be a dict with keys: model, feature_cols, threshold, medians.")
    return artifact


def _drop_label_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [col for col in LABEL_COLS if col in df.columns]
    return df.drop(columns=cols_to_drop) if cols_to_drop else df


def _detect_sort_col(df: pd.DataFrame) -> str | None:
    for col in _SORT_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _detect_group_key(df: pd.DataFrame) -> str | None:
    for col in _GROUP_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _prepare_features(df: pd.DataFrame, artifact: Dict[str, Any]) -> pd.DataFrame:
    data = df.copy()
    data = _drop_label_cols(data)

    sort_col = _detect_sort_col(data)
    if sort_col:
        try:
            data = data.sort_values(sort_col).reset_index(drop=True)
        except Exception:
            pass

    drop_cols = [c for c in LEAKAGE_COLS if c in data.columns]
    if drop_cols:
        data = data.drop(columns=drop_cols)

    group_key = _detect_group_key(data)
    X = apply_feature_pipeline(data, group_key=group_key)

    feature_cols = list(artifact.get("feature_cols", []))
    if not feature_cols:
        raise ValueError("Model artifact missing feature_cols.")

    X = X.reindex(columns=feature_cols)

    medians = artifact.get("medians")
    if medians is not None:
        medians = pd.Series(medians)
        X = X.fillna(medians)

    X = X.fillna(0.0)
    return X


def predict(df: pd.DataFrame, model_path: str = MODEL_PATH) -> pd.DataFrame:
    artifact = load_artifact(model_path)
    model = artifact.get("model")
    if model is None:
        raise ValueError("Model artifact missing model.")

    X = _prepare_features(df, artifact)
    probs = model.predict_proba(X)[:, 1]

    agent = PredictiveMaintenanceAgent.from_model_artifact(artifact)
    decisions = [agent.decide(float(p)) for p in probs]

    out = df.copy()
    out["failure_risk"] = probs
    out["prediction"] = [d.label for d in decisions]
    out["action"] = [d.action for d in decisions]
    out["reason"] = [d.reason for d in decisions]
    out["threshold_used"] = agent.threshold
    return out
