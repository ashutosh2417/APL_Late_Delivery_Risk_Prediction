"""
Prediction Module
APL Logistics - Late Delivery Risk Prediction
"""

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


def create_features(df):
    """
    Automatically create engineered features required by the model.
    """

    # Shipping Pressure Index
    df["Shipping Pressure Index"] = (
        df["Order Item Quantity"] /
        df["Days for shipment (scheduled)"].replace(0, 1)
    )

    # Mode Risk Flag
    df["Mode Risk Flag"] = np.where(
        df["Shipping Mode"] == "Standard Class",
        1,
        0
    )

    # Order Complexity Score
    df["Order Complexity Score"] = (
        df["Order Item Quantity"] *
        df["Product Price"] *
        (1 + df["Order Item Discount Rate"])
    )

    return df


def predict_risk(input_df):
    """
    Predict late delivery risk.
    """

    input_df = create_features(input_df.copy())

    transformed = preprocessor.transform(input_df)

    prediction = model.predict(transformed)[0]

    probability = model.predict_proba(transformed)[0][1]

    return prediction, probability