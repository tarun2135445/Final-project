import re
import numpy as np
import pandas as pd


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names for XGBoost (no brackets or special chars)."""
    def clean(name: str) -> str:
        name = re.sub(r"[^0-9A-Za-z_]+", "_", str(name))
        name = re.sub(r"__+", "_", name).strip("_")
        return name or "f"

    out = df.copy()
    out.columns = [clean(c) for c in out.columns]
    return out


def _slope(values: np.ndarray) -> float:
    """Return slope of a simple linear fit over the window."""
    x = np.arange(len(values))
    if len(values) < 2:
        return 0.0
    coef = np.polyfit(x, values, 1)
    return float(coef[0])


def add_time_features(df: pd.DataFrame, group_key: str | None = None) -> pd.DataFrame:
    """Add rolling stats, lags, and simple trend features per group (time-ordered)."""
    windows = (3, 5, 10)
    lags = (1, 5, 10)

    feat_df = df.copy()
    numeric_cols = feat_df.select_dtypes(include=["number", "float", "int"]).columns

    if group_key and group_key in feat_df.columns:
        grouped = feat_df.groupby(group_key, sort=False)
    else:
        grouped = [(None, feat_df)]

    out_frames = []
    for _, g in grouped:
        new_cols = {}
        for col in numeric_cols:
            series = g[col]
            for w in windows:
                roll = series.rolling(w, min_periods=1)
                roll_mean = roll.mean()
                roll_std = roll.std().fillna(0.0)
                roll_min = roll.min()
                roll_max = roll.max()
                roll_rms = roll.apply(lambda x: np.sqrt(np.mean(np.square(x))), raw=True)
                roll_slope = roll.apply(_slope, raw=True)

                new_cols[f"{col}_roll{w}_mean"] = roll_mean
                new_cols[f"{col}_roll{w}_std"] = roll_std
                new_cols[f"{col}_roll{w}_min"] = roll_min
                new_cols[f"{col}_roll{w}_max"] = roll_max
                new_cols[f"{col}_roll{w}_rms"] = roll_rms
                new_cols[f"{col}_roll{w}_slope"] = roll_slope
                new_cols[f"{col}_diff_roll{w}"] = series - roll_mean

            for lag in lags:
                new_cols[f"{col}_lag{lag}"] = series.shift(lag)

        new_feats = pd.DataFrame(new_cols, index=g.index)
        out_frames.append(pd.concat([g, new_feats], axis=1))

    feat_df = pd.concat(out_frames, axis=0).sort_index()
    return feat_df


def fill_with_train_stats(X_train: pd.DataFrame, X_other: pd.DataFrame):
    """Fill NaNs using training medians (numeric) and mode (categorical encoded as dummies)."""
    medians = X_train.median()
    X_train_filled = X_train.fillna(medians)
    X_other_filled = X_other.fillna(medians)
    return X_train_filled, X_other_filled, medians


def apply_feature_pipeline(df: pd.DataFrame, group_key: str | None = None) -> pd.DataFrame:
    """Apply time features + one-hot encoding + column sanitization."""
    X = add_time_features(df, group_key=group_key)
    X = pd.get_dummies(X, drop_first=True)
    X = clean_columns(X)
    return X
