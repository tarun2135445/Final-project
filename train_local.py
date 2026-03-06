import numpy as np

import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from features import apply_feature_pipeline, fill_with_train_stats

MODEL_OUT = "model.pkl"
RNG = np.random.default_rng(42)


def _choose_threshold(probs: np.ndarray, y_true: np.ndarray, target_recall: float = 0.98) -> float:
    """Pick a recall-first threshold: smallest threshold meeting target recall (fewer FN)."""
    if y_true.sum() == 0:
        return 0.5

    pos_scores = probs[y_true == 1]
    # Threshold at lower quantile of positive scores to keep high recall, with a small safety margin.
    base_t = float(np.quantile(pos_scores, max(0.0, 1 - target_recall)))
    base_t = max(0.0, min(1.0, base_t - 0.02))
    return base_t


def _rolling_splits(n_rows: int, min_train_frac: float = 0.6, test_frac: float = 0.1, max_folds: int = 3):
    """Generate walk-forward train/test index slices."""
    min_train = max(int(n_rows * min_train_frac), 1)
    test_size = max(int(n_rows * test_frac), 1)
    start = min_train
    folds = []
    while start + test_size <= n_rows and len(folds) < max_folds:
        folds.append((slice(0, start), slice(start, start + test_size)))
        start += test_size
    if not folds:
        folds.append((slice(0, min_train), slice(min_train, n_rows)))
    return folds




def _stress_test_noise(model, X: pd.DataFrame, y: pd.Series, threshold: float, medians: pd.Series):
    """Simple noise stress: missingness, spikes, drift."""
    noisy = X.copy().astype(float)
    # Random missing values.
    mask = RNG.random(noisy.shape) < 0.01
    noisy = noisy.mask(mask)
    # Spikes on a few columns.
    spike_cols = noisy.columns[: min(5, noisy.shape[1])]
    noisy.loc[:, spike_cols] = noisy.loc[:, spike_cols] + RNG.normal(0, noisy.loc[:, spike_cols].std(), size=noisy.loc[:, spike_cols].shape)
    # Linear drift on first column.
    if noisy.shape[0]:
        drift = np.linspace(0, 0.5, noisy.shape[0])
        noisy.iloc[:, 0] = noisy.iloc[:, 0] + drift
    noisy = noisy.fillna(medians)
    probs = model.predict_proba(noisy)[:, 1]
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y, preds, labels=[0, 1])
    return cm


def main():
    df = pd.read_csv("sensor_train.csv")

    # Accept either a generic "label" column or the dataset's "Machine failure" column.
    for label_col in ("label", "Machine failure"):
        if label_col in df.columns:
            break
    else:
        raise KeyError("No label column found. Expected 'label' or 'Machine failure'.")

    # Sort chronologically (UDI is a monotonically increasing ID in this dataset) to avoid
    # leaking future patterns into training.
    sort_col = "UDI" if "UDI" in df.columns else None
    if sort_col:
        df = df.sort_values(sort_col).reset_index(drop=True)

    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])

    # Remove pure identifiers and leakage-y failure component flags.
    leakage_cols = {"UDI", "TWF", "HDF", "PWF", "OSF", "RNF"}
    drop_cols = [c for c in leakage_cols if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    group_key = "Type" if "Type" in X.columns else None
    X = apply_feature_pipeline(X, group_key=group_key)

    folds = _rolling_splits(len(X), min_train_frac=0.6, test_frac=0.1, max_folds=4)
    fold_results = []

    for fold_num, (train_idx, test_idx) in enumerate(folds, start=1):
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train_raw, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Calibration holdout from tail of train.
        cal_start = max(int(len(X_train_raw) * 0.9), len(X_train_raw) - 1)
        X_fit_raw, X_cal_raw = X_train_raw.iloc[:cal_start], X_train_raw.iloc[cal_start:]
        y_fit, y_cal = y_train_raw.iloc[:cal_start], y_train_raw.iloc[cal_start:]

        # Impute using train medians.
        X_fit, X_cal, medians = fill_with_train_stats(X_fit_raw, X_cal_raw)
        X_test = X_test_raw.fillna(medians)

        # Imbalance handling.
        pos = (y_fit == 1).sum()
        neg = (y_fit == 0).sum()
        scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

        base_model = XGBClassifier(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        )

        base_model.fit(X_fit, y_fit)

        if len(X_cal) and y_cal.nunique() == 2:
            calibrator = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
            calibrator.fit(X_cal, y_cal)
            model_for_eval = calibrator
            cal_probs = calibrator.predict_proba(X_cal)[:, 1]
            threshold = _choose_threshold(cal_probs, y_cal, target_recall=0.98)
        else:
            model_for_eval = base_model
            train_probs = base_model.predict_proba(X_fit)[:, 1]
            threshold = _choose_threshold(train_probs, y_fit, target_recall=0.98)

        probs = model_for_eval.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)

        ap = average_precision_score(y_test, probs)
        roc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        baseline_preds = np.zeros(len(y_test), dtype=int)
        baseline_ap = average_precision_score(y_test, baseline_preds)
        baseline_cm = confusion_matrix(y_test, baseline_preds, labels=[0, 1])

        fold_results.append(
            {
                "fold": fold_num,
                "threshold": threshold,
                "ap": ap,
                "roc": roc,
                "cm": cm,
                "fp": int(fp),
                "fn": int(fn),
                "baseline_ap": baseline_ap,
                "baseline_cm": baseline_cm,
                "model": model_for_eval,
                "medians": medians,
            }
        )

        print(f"\nFold {fold_num}: train={len(X_train_raw)}, test={len(X_test)} threshold={threshold:.3f}")
        print(f"PR-AUC (AP): {ap:.4f} | ROC-AUC: {roc:.4f}")
        print("Confusion matrix [ [TN FP] [FN TP] ]:")
        print(cm)
        print(f"False negatives (missed failures): {fn} | False positives: {fp}")
        print("Baseline confusion matrix (always 0):")
        print(baseline_cm)

        # Logistic regression baseline comparison.
        lr_baseline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ])
        lr_baseline.fit(X_fit, y_fit)
        lr_probs = lr_baseline.predict_proba(X_test)[:, 1]
        lr_ap = average_precision_score(y_test, lr_probs)
        lr_roc = roc_auc_score(y_test, lr_probs)
        print(f"  Logistic Regression → AP: {lr_ap:.4f} | ROC-AUC: {lr_roc:.4f}")
        print(f"  XGBoost             → AP: {ap:.4f} | ROC-AUC: {roc:.4f}")

    # Summary across folds.
    mean_ap = np.mean([f["ap"] for f in fold_results])
    mean_fn = np.mean([f["fn"] for f in fold_results])
    print(f"\nMean AP across folds: {mean_ap:.4f} | Mean FN: {mean_fn:.2f}")

    # Train final model on all-but-tail calibration slice and calibrate + save.
    cal_start_full = max(int(len(X) * 0.9), len(X) - 1)
    X_fit_full_raw, X_cal_full_raw = X.iloc[:cal_start_full], X.iloc[cal_start_full:]
    y_fit_full, y_cal_full = y.iloc[:cal_start_full], y.iloc[cal_start_full:]
    X_fit_full, X_cal_full, medians_full = fill_with_train_stats(X_fit_full_raw, X_cal_full_raw)

    pos_full = (y_fit_full == 1).sum()
    neg_full = (y_fit_full == 0).sum()
    scale_pos_weight_full = float(neg_full / pos_full) if pos_full > 0 else 1.0

    final_base = XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight_full,
    )
    final_base.fit(X_fit_full, y_fit_full)

    if len(X_cal_full) and y_cal_full.nunique() == 2:
        final_model = CalibratedClassifierCV(final_base, method="sigmoid", cv="prefit")
        final_model.fit(X_cal_full, y_cal_full)
        cal_probs_full = final_model.predict_proba(X_cal_full)[:, 1]
        final_threshold = _choose_threshold(cal_probs_full, y_cal_full, target_recall=0.98)
    else:
        final_model = final_base
        train_probs_full = final_model.predict_proba(X_fit_full)[:, 1]
        final_threshold = _choose_threshold(train_probs_full, y_fit_full, target_recall=0.98)

    print(f"\nFinal model threshold (>=0.98 recall target): {final_threshold:.3f}")

    # Stress test on the last fold's test set.
    last_fold = fold_results[-1]
    stress_cm = _stress_test_noise(last_fold["model"], X.iloc[folds[-1][1]].fillna(last_fold["medians"]), y.iloc[folds[-1][1]], last_fold["threshold"], last_fold["medians"])
    print("Noise stress test confusion matrix [ [TN FP] [FN TP] ]:")
    print(stress_cm)

    # Optional SHAP summary for explainability.
    try:
        import shap  # type: ignore

        sample = X_fit_full.iloc[-200:].fillna(medians_full)
        explainer = shap.TreeExplainer(final_base)
        shap_vals = explainer.shap_values(sample)
        mean_contrib = pd.Series(np.abs(shap_vals).mean(axis=0), index=sample.columns)
        top = mean_contrib.sort_values(ascending=False).head(10)
        print("\nTop 10 features by mean |SHAP|:")
        print(top.to_string())
    except Exception as e:
        print(f"SHAP summary skipped: {e}")

    joblib.dump(
        {
            "model": final_model,
            "feature_cols": list(X.columns),
            "threshold": final_threshold,
            "medians": medians_full,
        },
        MODEL_OUT,
    )
    print(f"Saved: {MODEL_OUT}")

if __name__ == "__main__":
    main()
