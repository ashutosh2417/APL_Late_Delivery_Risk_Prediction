"""
Model Evaluation
Machine Learning-Based Late Delivery Risk Prediction
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ----------------------------------------------------
# Create folder
# ----------------------------------------------------

os.makedirs("outputs/evaluation", exist_ok=True)

# ----------------------------------------------------
# Load model
# ----------------------------------------------------

model = joblib.load("models/best_model.pkl")

X_test = joblib.load("models/X_test.pkl")

y_test = joblib.load("models/y_test.pkl")

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# ----------------------------------------------------
# Metrics
# ----------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc = roc_auc_score(y_test, y_prob)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc:.4f}")

# ----------------------------------------------------
# Classification Report
# ----------------------------------------------------

report = classification_report(y_test, y_pred)

with open(
    "outputs/evaluation/classification_report.txt",
    "w"
) as file:

    file.write(report)

print("✓ Classification Report Saved")

# ----------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.tight_layout()

plt.savefig(
    "outputs/evaluation/confusion_matrix.png",
    dpi=300
)

plt.close()

print("✓ Confusion Matrix Saved")

# ----------------------------------------------------
# ROC Curve
# ----------------------------------------------------

RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.tight_layout()

plt.savefig(
    "outputs/evaluation/roc_curve.png",
    dpi=300
)

plt.close()

print("✓ ROC Curve Saved")

# ----------------------------------------------------
# Precision Recall Curve
# ----------------------------------------------------

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_prob
)

plt.tight_layout()

plt.savefig(
    "outputs/evaluation/precision_recall_curve.png",
    dpi=300
)

plt.close()

print("✓ Precision Recall Curve Saved")

print("\nEvaluation Completed Successfully")