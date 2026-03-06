import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inference import _drop_label_cols, _prepare_features, load_artifact, predict

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.pkl")


def _make_sensor_df(n=30):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "UDI": range(1, n + 1),
        "Product ID": [f"M{i}" for i in range(n)],
        "Type": ["L"] * 10 + ["M"] * 10 + ["H"] * 10,
        "Air temperature [K]": rng.normal(300, 2, n),
        "Process temperature [K]": rng.normal(310, 1, n),
        "Rotational speed [rpm]": rng.normal(1500, 100, n),
        "Torque [Nm]": rng.normal(40, 5, n),
        "Tool wear [min]": np.linspace(0, 200, n),
    })


class TestDropLabelCols:
    def test_drops_machine_failure(self):
        df = _make_sensor_df()
        df["Machine failure"] = 0
        out = _drop_label_cols(df)
        assert "Machine failure" not in out.columns

    def test_drops_label(self):
        df = _make_sensor_df()
        df["label"] = 0
        out = _drop_label_cols(df)
        assert "label" not in out.columns

    def test_no_label_col_unchanged(self):
        df = _make_sensor_df()
        out = _drop_label_cols(df)
        assert set(out.columns) == set(df.columns)

    def test_drops_only_first_match(self):
        df = _make_sensor_df()
        df["label"] = 0
        df["Machine failure"] = 0
        out = _drop_label_cols(df)
        # Only one should be dropped (the first match: "label")
        assert "label" not in out.columns


class TestLoadArtifact:
    def test_returns_dict(self):
        artifact = load_artifact(MODEL_PATH)
        assert isinstance(artifact, dict)

    def test_has_required_keys(self):
        artifact = load_artifact(MODEL_PATH)
        for key in ("model", "feature_cols", "threshold", "medians"):
            assert key in artifact, f"Missing key: {key}"

    def test_threshold_in_range(self):
        artifact = load_artifact(MODEL_PATH)
        assert 0.0 <= artifact["threshold"] <= 1.0

    def test_feature_cols_nonempty(self):
        artifact = load_artifact(MODEL_PATH)
        assert len(artifact["feature_cols"]) > 0


class TestPredict:
    def test_returns_dataframe(self):
        df = _make_sensor_df()
        out = predict(df, model_path=MODEL_PATH)
        assert isinstance(out, pd.DataFrame)

    def test_output_has_required_columns(self):
        df = _make_sensor_df()
        out = predict(df, model_path=MODEL_PATH)
        for col in ("failure_risk", "prediction", "action", "reason", "threshold_used"):
            assert col in out.columns, f"Missing output column: {col}"

    def test_same_row_count(self):
        df = _make_sensor_df()
        out = predict(df, model_path=MODEL_PATH)
        assert len(out) == len(df)

    def test_failure_risk_in_range(self):
        df = _make_sensor_df()
        out = predict(df, model_path=MODEL_PATH)
        assert out["failure_risk"].between(0.0, 1.0).all()

    def test_prediction_is_binary(self):
        df = _make_sensor_df()
        out = predict(df, model_path=MODEL_PATH)
        assert set(out["prediction"].unique()).issubset({0, 1})

    def test_action_values_are_valid(self):
        df = _make_sensor_df()
        out = predict(df, model_path=MODEL_PATH)
        valid_actions = {"NO_ACTION", "CREATE_MAINTENANCE_ALERT"}
        assert set(out["action"].unique()).issubset(valid_actions)

    def test_threshold_consistent(self):
        df = _make_sensor_df()
        out = predict(df, model_path=MODEL_PATH)
        assert out["threshold_used"].nunique() == 1

    def test_label_col_stripped_from_input(self):
        df = _make_sensor_df()
        df["Machine failure"] = 0
        out = predict(df, model_path=MODEL_PATH)
        assert len(out) == len(df)

    def test_handles_missing_type_column(self):
        df = _make_sensor_df().drop(columns=["Type"])
        out = predict(df, model_path=MODEL_PATH)
        assert len(out) == len(df)
