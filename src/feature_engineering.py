"""
Feature Engineering
Machine Learning-Based Late Delivery Risk Prediction
"""

import numpy as np
import pandas as pd

from config import DATA_PATH, PROCESSED_DATA_PATH


def load_data():
    """Load dataset"""
    return pd.read_csv(DATA_PATH, encoding="latin1")


def create_shipping_pressure(df):
    """Shipping Pressure Index"""

    df["Shipping Pressure Index"] = (
        df["Order Item Quantity"] /
        df["Days for shipment (scheduled)"].replace(0, 1)
    )

    return df


def create_mode_risk_flag(df):
    """Shipping Mode Risk Flag"""

    df["Mode Risk Flag"] = np.where(
        df["Shipping Mode"] == "Standard Class",
        1,
        0
    )

    return df


def create_order_complexity(df):
    """Order Complexity Score"""

    df["Order Complexity Score"] = (
        df["Order Item Quantity"] *
        df["Product Price"] *
        (1 + df["Order Item Discount Rate"])
    )

    return df


def create_regional_congestion(df):
    """Regional Congestion Indicator"""

    congestion = (
        df.groupby("Order Region")["Late_delivery_risk"]
        .mean()
    )

    df["Regional Congestion"] = (
        df["Order Region"].map(congestion)
    )

    return df


def drop_unnecessary_columns(df):
    """Drop ID and personal information"""

    columns = [

        "Customer Fname",
        "Customer Lname",
        "Customer Street",
        "Customer Zipcode",

        "Customer Id",
        "Order Customer Id",

        "Latitude",
        "Longitude",

        "Category Id",
        "Department Id"

    ]

    return df.drop(columns=columns)


def remove_data_leakage(df):
    """Remove leakage columns"""

    leakage = [

        "Days for shipping (real)",
        "Delivery Status"

    ]

    return df.drop(columns=leakage)


def save_dataset(df):
    df.to_csv(PROCESSED_DATA_PATH, index=False)


def main():

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    df = load_data()

    print("✓ Dataset Loaded")

    df = create_shipping_pressure(df)
    print("✓ Shipping Pressure Index")

    df = create_mode_risk_flag(df)
    print("✓ Mode Risk Flag")

    df = create_order_complexity(df)
    print("✓ Order Complexity Score")

    df = create_regional_congestion(df)
    print("✓ Regional Congestion")

    df = drop_unnecessary_columns(df)
    print("✓ Unnecessary Columns Removed")

    df = remove_data_leakage(df)
    print("✓ Leakage Columns Removed")

    save_dataset(df)

    print("\nProcessed Dataset Shape :", df.shape)

    print("\nDataset Saved Successfully")


if __name__ == "__main__":
    main()