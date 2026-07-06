"""
Model Training
Machine Learning-Based Late Delivery Risk Prediction
"""

import time
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =====================================================
# Load Data
# =====================================================

X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")

y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")

print("=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# =====================================================
# Models
# =====================================================

models = {

    "Logistic Regression":

        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

    "Random Forest":

        RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
),

    "Gradient Boosting":

        GradientBoostingClassifier(
            random_state=42
        ),

    "XGBoost":

XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

}

results = []

best_auc = 0
best_model = None
best_name = None

# =====================================================
# Train Models
# =====================================================

for name, model in models.items():

    print(f"\nTraining {name}...")

    start = time.time()

    model.fit(X_train, y_train)

    end = time.time()

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    roc = roc_auc_score(y_test, y_prob)

    results.append({

        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc,
        "Training Time (sec)": round(end - start, 2)

    })

    print(f"Accuracy : {accuracy:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")

    joblib.dump(
        model,
        f"models/{name.lower().replace(' ','_')}.pkl"
    )

    if roc > best_auc:

        best_auc = roc
        best_model = model
        best_name = name

# =====================================================
# Save Best Model
# =====================================================

joblib.dump(best_model, "models/best_model.pkl")

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="ROC-AUC",
    ascending=False
)

comparison.to_csv(
    "outputs/metrics/model_comparison.csv",
    index=False
)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison)

print("\nBest Model :", best_name)

print("ROC-AUC :", round(best_auc,4))

print("\nAll models saved successfully.")