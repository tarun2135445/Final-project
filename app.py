import io
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from inference import predict, load_artifact

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {
  --bg: #f5f7fb;
  --panel: #ffffff;
  --panel-alt: #f1f5f9;
  --accent: #1f5eff;
  --accent-2: #00a2ff;
  --text: #0f172a;
  --muted: #6b7280;
  --border: #e5e7eb;
  --success: #16a34a;
  --danger: #dc2626;
  --warning: #d97706;
}

html, body, [class*="css"]  {
  font-family: 'IBM Plex Sans', sans-serif;
  color: var(--text);
}

.stApp {
  background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%) fixed;
}

section[data-testid="stSidebar"] {
  background: #f8fafc;
  border-right: 1px solid var(--border);
}

h1, h2, h3 {
  letter-spacing: 0.2px;
  color: var(--text);
}

.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}

.note {
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  color: #1e3a8a;
  border-radius: 12px;
  padding: 12px 14px;
}

.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 12px;
  background: rgba(31,94,255,0.08);
  color: var(--accent);
  border: 1px solid rgba(31,94,255,0.2);
}

.kpi {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
  box-shadow: 0 10px 20px rgba(15, 23, 42, 0.06);
}

.kpi-label {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
}

.kpi-value {
  font-size: 28px;
  font-weight: 700;
  margin-top: 4px;
}

.kpi-sub {
  font-size: 12px;
  color: var(--muted);
  margin-top: 2px;
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--panel);
  font-size: 13px;
  font-weight: 600;
}

.chart-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
  box-shadow: 0 10px 20px rgba(15, 23, 42, 0.06);
}

.stButton>button {
  background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
  color: white;
  border: none;
  border-radius: 10px;
  padding: 0.6rem 1.2rem;
  font-weight: 700;
}

/* Table header contrast + sticky header */
[data-testid="stDataFrame"] thead th {
  color: #0f172a !important;
  background: #f8fafc !important;
  border-bottom: 1px solid var(--border) !important;
  position: sticky !important;
  top: 0 !important;
  z-index: 2 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("# Predictive Maintenance Dashboard")
st.markdown("<span class='badge'>Clean enterprise view</span>", unsafe_allow_html=True)
st.caption("Upload sensor data to generate risk scores and maintenance actions.")

with st.sidebar:
    st.markdown("## Run Setup")
    default_model_path = os.getenv("MODEL_PATH", "model.pkl")
    model_path = st.text_input("Model path", value=default_model_path)
    uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])
    run = st.button("Run Inference")

st.markdown(
    """
<div class="note">
<strong>Note</strong>: Time-based features use rolling windows and lags, so predictions are most reliable on
chronologically ordered batches rather than single rows. If your data has <code>UDI</code>, it will be auto-sorted.
</div>
""",
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
_LABEL_COLS = ("label", "Machine failure", "failure")
_GROUP_CANDIDATES = ("Type", "device", "machine", "unit", "asset", "sensor_id")


# ── Helpers ────────────────────────────────────────────────────────────────────
def _detect_group_col(df: pd.DataFrame) -> str | None:
    for col in _GROUP_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _validate_upload(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (blocking_errors, warnings). Errors stop inference; warnings are displayed."""
    errors: list[str] = []
    warnings: list[str] = []

    if df.empty:
        errors.append("The uploaded file is empty.")
        return errors, warnings

    if len(df.select_dtypes(include="number").columns) == 0:
        errors.append(
            "No numeric columns found. The model requires sensor readings (numeric data)."
        )

    return errors, warnings


def _ground_truth_col(df: pd.DataFrame) -> str | None:
    for col in _LABEL_COLS:
        if col in df.columns:
            return col
    return None


def _confusion_matrix_html(cm: np.ndarray) -> str:
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return f"""
<div style="overflow-x:auto;">
<table style="border-collapse:collapse;font-size:14px;margin:16px auto;text-align:center;">
  <tr>
    <th style="padding:10px;"></th>
    <th style="padding:10px;color:#6b7280;font-weight:600;">Predicted Normal</th>
    <th style="padding:10px;color:#6b7280;font-weight:600;">Predicted Failure</th>
  </tr>
  <tr>
    <th style="padding:10px;color:#6b7280;font-weight:600;text-align:right;">Actual Normal</th>
    <td style="padding:18px 28px;background:#dcfce7;border-radius:8px;font-weight:700;font-size:22px;">
      {tn}<br><span style="font-size:11px;font-weight:400;color:#166534;">True Negative</span>
    </td>
    <td style="padding:18px 28px;background:#fee2e2;border-radius:8px;font-weight:700;font-size:22px;">
      {fp}<br><span style="font-size:11px;font-weight:400;color:#991b1b;">False Positive</span>
    </td>
  </tr>
  <tr>
    <th style="padding:10px;color:#6b7280;font-weight:600;text-align:right;">Actual Failure</th>
    <td style="padding:18px 28px;background:#fee2e2;border-radius:8px;font-weight:700;font-size:22px;">
      {fn}<br><span style="font-size:11px;font-weight:400;color:#991b1b;">False Negative</span>
    </td>
    <td style="padding:18px 28px;background:#dcfce7;border-radius:8px;font-weight:700;font-size:22px;">
      {tp}<br><span style="font-size:11px;font-weight:400;color:#166534;">True Positive</span>
    </td>
  </tr>
</table>
</div>
<p style="text-align:center;color:#6b7280;font-size:12px;margin-top:4px;">
  Recall: <strong>{recall:.1%}</strong> &nbsp;|&nbsp; Precision: <strong>{precision:.1%}</strong>
</p>
"""


def _get_feature_importances(artifact: dict) -> pd.Series | None:
    model = artifact.get("model")
    feature_cols = artifact.get("feature_cols", [])
    if model is None or not feature_cols:
        return None
    try:
        base = (
            model.calibrated_classifiers_[0].estimator
            if hasattr(model, "calibrated_classifiers_")
            else model
        )
        importances = pd.Series(base.feature_importances_, index=feature_cols)
        return importances.sort_values(ascending=False).head(20)
    except Exception:
        return None


# ── Main logic ─────────────────────────────────────────────────────────────────
if run and uploaded is None:
    st.warning("Upload a CSV file to run predictions.")

if uploaded is not None and run:
    data = pd.read_csv(uploaded)

    # ── Validation ──────────────────────────────────────────────────────────
    val_errors, val_warnings = _validate_upload(data)
    for w in val_warnings:
        st.warning(w)
    if val_errors:
        for e in val_errors:
            st.error(e)
        st.stop()

    try:
        results = predict(data, model_path=model_path)
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        st.stop()

    failure_count = int((results["prediction"] == 1).sum())
    total = len(results)
    mean_risk = float(results["failure_risk"].mean()) if total else 0.0
    threshold_used = float(results["threshold_used"].iloc[0]) if total else 0.0

    if mean_risk >= threshold_used:
        status_label, status_color, status_bg = "Elevated risk", "#dc2626", "rgba(220,38,38,0.12)"
    elif mean_risk >= threshold_used * 0.7:
        status_label, status_color, status_bg = "Watch", "#d97706", "rgba(217,119,6,0.12)"
    else:
        status_label, status_color, status_bg = "Normal", "#16a34a", "rgba(22,163,74,0.12)"

    st.markdown(
        f"""
<div class='card' style='margin-top: 12px;'>
  <div style='display:flex; align-items:center; justify-content:space-between; gap:12px;'>
    <div>
      Loaded <strong>{total:,}</strong> rows. Threshold <strong>{threshold_used:.3f}</strong>.
      Predicted failures <strong>{failure_count:,}</strong>.
    </div>
    <div class='status-pill' style='border-color:{status_color}; color:{status_color}; background:{status_bg};'>
      {status_label}
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── KPI Cards ────────────────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""
<div class='kpi'>
  <div class='kpi-label'>Total Rows</div>
  <div class='kpi-value'>{total:,}</div>
  <div class='kpi-sub'>Rows processed</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
<div class='kpi'>
  <div class='kpi-label'>Predicted Failures</div>
  <div class='kpi-value' style='color:{status_color};'>{failure_count:,}</div>
  <div class='kpi-sub'>Actionable alerts</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f"""
<div class='kpi'>
  <div class='kpi-label'>Mean Risk</div>
  <div class='kpi-value'>{mean_risk:.3f}</div>
  <div class='kpi-sub'>Average score</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""
<div class='kpi'>
  <div class='kpi-label'>Threshold</div>
  <div class='kpi-value'>{threshold_used:.3f}</div>
  <div class='kpi-sub'>Model setting</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_overview, tab_eval, tab_fi, tab_data = st.tabs(
        ["Overview", "Evaluation", "Feature Importance", "Raw Data"]
    )

    # ── Overview tab ─────────────────────────────────────────────────────────
    with tab_overview:
        charts_left, charts_right = st.columns((2, 1))

        with charts_left:
            st.markdown("### Risk Distribution")
            bins = np.linspace(0.0, 1.0, 11)
            counts, edges = np.histogram(results["failure_risk"], bins=bins)
            labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(len(edges) - 1)]
            hist_df = pd.DataFrame({"risk_bucket": labels, "count": counts})
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.bar_chart(hist_df.set_index("risk_bucket"))
            st.markdown("</div>", unsafe_allow_html=True)

        with charts_right:
            group_col = _detect_group_col(results)
            st.markdown(f"### Failure Rate by {group_col or 'Group'}")
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            if group_col:
                type_counts = (
                    results.groupby(group_col)["prediction"]
                    .agg(["sum", "count"])
                    .reset_index()
                )
                type_counts["rate"] = type_counts["sum"] / type_counts["count"]
                st.bar_chart(type_counts.set_index(group_col)["rate"])
            else:
                st.info("No group column found for breakdown.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Evaluation tab ────────────────────────────────────────────────────────
    with tab_eval:
        gt_col = _ground_truth_col(data)
        if gt_col is None:
            st.info(
                "No ground-truth label column found in the uploaded file. "
                "Upload a CSV that includes **'Machine failure'** or **'label'** "
                "to see evaluation metrics."
            )
        else:
            y_true = data[gt_col].astype(int).values
            y_prob = results["failure_risk"].values
            y_pred = results["prediction"].values

            try:
                roc = roc_auc_score(y_true, y_prob)
                ap = average_precision_score(y_true, y_prob)
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(
                        f"""
<div class='kpi'>
  <div class='kpi-label'>ROC-AUC</div>
  <div class='kpi-value'>{roc:.4f}</div>
  <div class='kpi-sub'>Area under ROC curve</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        f"""
<div class='kpi'>
  <div class='kpi-label'>PR-AUC (AP)</div>
  <div class='kpi-value'>{ap:.4f}</div>
  <div class='kpi-sub'>Average precision score</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                with m3:
                    st.markdown(
                        f"""
<div class='kpi'>
  <div class='kpi-label'>Recall</div>
  <div class='kpi-value' style='color:#16a34a;'>{recall:.1%}</div>
  <div class='kpi-sub'>Failures caught</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                with m4:
                    st.markdown(
                        f"""
<div class='kpi'>
  <div class='kpi-label'>Precision</div>
  <div class='kpi-value'>{precision:.1%}</div>
  <div class='kpi-sub'>Alert accuracy</div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### Confusion Matrix")
                st.markdown(_confusion_matrix_html(cm), unsafe_allow_html=True)

            except Exception as exc:
                st.error(f"Could not compute evaluation metrics: {exc}")

    # ── Feature Importance tab ────────────────────────────────────────────────
    with tab_fi:
        st.markdown("### Top 20 Feature Importances")
        st.caption("XGBoost gain score — total information gain from all splits on each feature.")
        try:
            artifact = load_artifact(model_path)
            importances = _get_feature_importances(artifact)
            if importances is not None:
                fi_df = importances.reset_index()
                fi_df.columns = ["feature", "importance"]
                st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
                st.bar_chart(fi_df.set_index("feature"))
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Could not extract feature importances from the model artifact.")
        except Exception as exc:
            st.error(f"Could not load model for feature importances: {exc}")

    # ── Raw Data tab ──────────────────────────────────────────────────────────
    with tab_data:
        st.markdown("### Detailed Results")
        st.dataframe(results, use_container_width=True, height=520)

        buffer = io.StringIO()
        results.to_csv(buffer, index=False)
        st.download_button(
            label="Download CSV",
            data=buffer.getvalue(),
            file_name="predictions.csv",
            mime="text/csv",
        )
