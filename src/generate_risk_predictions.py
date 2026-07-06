"""
Generate Risk Predictions (Fast Batch Version)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# ----------------------------------------------------
# PROJECT ROOT
# ----------------------------------------------------
sys.path.append(os.path.abspath("."))

# ----------------------------------------------------
# LOAD MODEL & PREPROCESSOR
# ----------------------------------------------------
model = joblib.load("models/best_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
print("=" * 60)
print("Loading dataset...")
print("=" * 60)

df = pd.read_csv("data/processed_data.csv")

X = df.drop(columns=["Late_delivery_risk"]).copy()

# ----------------------------------------------------
# CREATE ENGINEERED FEATURES (ONLY IF MISSING)
# ----------------------------------------------------

if "Shipping Pressure Index" not in X.columns:
    X["Shipping Pressure Index"] = (
        X["Order Item Quantity"]
        / X["Days for shipment (scheduled)"].replace(0, 1)
    )

if "Mode Risk Flag" not in X.columns:
    X["Mode Risk Flag"] = np.where(
        X["Shipping Mode"] == "Standard Class",
        1,
        0,
    )

if "Order Complexity Score" not in X.columns:
    X["Order Complexity Score"] = (
        X["Order Item Quantity"]
        * X["Product Price"]
        * (1 + X["Order Item Discount Rate"])
    )

print("=" * 60)
print("Transforming features...")
print("=" * 60)

X_processed = preprocessor.transform(X)

print("=" * 60)
print("Generating predictions...")
print("=" * 60)

predictions = model.predict(X_processed)

probabilities = model.predict_proba(X_processed)[:, 1]

# ----------------------------------------------------
# SAVE RESULTS
# ----------------------------------------------------

df["Predicted_Risk"] = predictions

df["Risk_Probability"] = probabilities

df["Risk_Category"] = pd.cut(
    df["Risk_Probability"],
    bins=[0, 0.40, 0.70, 1],
    labels=["Low", "Medium", "High"],
    include_lowest=True,
)

os.makedirs("outputs", exist_ok=True)

output_file = "outputs/risk_predictions.csv"

df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("✅ Risk prediction file created successfully!")
print(f"Saved to: {output_file}")
print("=" * 60)