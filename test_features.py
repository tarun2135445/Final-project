import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features import clean_columns, add_time_features, apply_feature_pipeline


def _make_df(n=20):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "UDI": range(1, n + 1),
        "Type": ["L"] * (n // 2) + ["M"] * (n // 2),
        "Air temperature [K]": rng.normal(300, 2, n),
        "Torque [Nm]": rng.normal(40, 5, n),
        "Tool wear [min]": np.linspace(0, 200, n),
    })


class TestCleanColumns:
    def test_removes_special_chars(self):
        df = pd.DataFrame({"col [K]": [1], "normal": [2]})
        out = clean_columns(df)
        assert "col__K_" in out.columns or all(c.isidentifier() or "_" in c for c in out.columns)

    def test_no_empty_column_names(self):
        df = pd.DataFrame({"[K]": [1]})
        out = clean_columns(df)
        assert all(len(c) > 0 for c in out.columns)

    def test_double_underscores_collapsed(self):
        df = pd.DataFrame({"a  b": [1]})
        out = clean_columns(df)
        assert "__" not in list(out.columns)[0]


class TestAddTimeFeatures:
    def test_output_has_more_columns(self):
        df = _make_df()
        out = add_time_features(df)
        assert out.shape[1] > df.shape[1]

    def test_same_row_count(self):
        df = _make_df()
        out = add_time_features(df)
        assert len(out) == len(df)

    def test_lag_columns_created(self):
        df = _make_df()
        out = add_time_features(df)
        lag_cols = [c for c in out.columns if "_lag" in c]
        assert len(lag_cols) > 0

    def test_rolling_mean_columns_created(self):
        df = _make_df()
        out = add_time_features(df)
        roll_cols = [c for c in out.columns if "_roll" in c and "_mean" in c]
        assert len(roll_cols) > 0

    def test_group_key_produces_same_rows(self):
        df = _make_df()
        out = add_time_features(df, group_key="Type")
        assert len(out) == len(df)

    def test_no_all_nan_columns(self):
        df = _make_df()
        out = add_time_features(df)
        numeric = out.select_dtypes(include="number")
        assert not numeric.isnull().all().any()


class TestApplyFeaturePipeline:
    def test_returns_dataframe(self):
        df = _make_df()
        out = apply_feature_pipeline(df)
        assert isinstance(out, pd.DataFrame)

    def test_no_object_columns(self):
        df = _make_df()
        out = apply_feature_pipeline(df, group_key="Type")
        assert len(out.select_dtypes(include="object").columns) == 0

    def test_column_names_are_valid_identifiers(self):
        df = _make_df()
        out = apply_feature_pipeline(df)
        for col in out.columns:
            assert col.replace("_", "").isalnum(), f"Invalid column name: {col}"
