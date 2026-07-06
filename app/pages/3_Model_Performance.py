import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path

from style import load_css

load_css()

from data_loader import load_model_comparison
from config import PROJECT_TITLE

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="Model Performance",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Model Performance")

st.markdown(
    """
    Evaluate the Machine Learning models used for predicting
    late delivery risk in APL Logistics.
    """
)

st.divider()

# ----------------------------------------------------
# LOAD MODEL COMPARISON
# ----------------------------------------------------

comparison = load_model_comparison()

st.subheader("🏆 Model Comparison")

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

st.divider()

# ----------------------------------------------------
# BEST MODEL
# ----------------------------------------------------

best = comparison.iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Best Model", best["Model"])

c2.metric("Accuracy", f"{best['Accuracy']:.3f}")

c3.metric("Precision", f"{best['Precision']:.3f}")

c4.metric("Recall", f"{best['Recall']:.3f}")

c5.metric("ROC-AUC", f"{best['ROC-AUC']:.3f}")

st.divider()

# ----------------------------------------------------
# MODEL VISUALIZATIONS
# ----------------------------------------------------

evaluation_path = Path("outputs/evaluation")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Confusion Matrix")

    cm = Image.open(evaluation_path / "confusion_matrix.png")

    st.image(cm, use_container_width=True)

with col2:

    st.subheader("ROC Curve")

    roc = Image.open(evaluation_path / "roc_curve.png")

    st.image(roc, use_container_width=True)

st.divider()

st.subheader("Precision Recall Curve")

pr = Image.open(
    evaluation_path / "precision_recall_curve.png"
)

st.image(pr, use_container_width=True)

st.divider()

# ----------------------------------------------------
# CLASSIFICATION REPORT
# ----------------------------------------------------

st.subheader("Classification Report")

report = open(
    evaluation_path / "classification_report.txt",
    "r",
    encoding="utf-8"
).read()

st.code(report)

st.divider()

# ----------------------------------------------------
# BUSINESS INSIGHTS
# ----------------------------------------------------

st.subheader("📌 Business Insights")

st.success(
"""
• Logistic Regression achieved the best overall performance.

• ROC-AUC above 0.80 indicates strong discrimination capability.

• High precision reduces false delay alerts.

• Good recall ensures most late deliveries are detected.

• The model is suitable for operational deployment as an
early warning system.
"""
)