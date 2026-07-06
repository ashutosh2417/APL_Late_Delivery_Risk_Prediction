"""
Exploratory Data Analysis
Machine Learning-Based Late Delivery Risk Prediction
"""

import pandas as pd

from config import DATA_PATH, SUMMARY_PATH

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH, encoding="latin1")

print("=" * 70)
print("APL LOGISTICS DATASET SUMMARY")
print("=" * 70)

print(f"\nDataset Shape : {df.shape}")

print("\nMissing Values")

print(df.isnull().sum())

print("\nDuplicate Rows")

print(df.duplicated().sum())

print("\nData Types")

print(df.dtypes)

print("\nStatistical Summary")

print(df.describe(include="all"))

# --------------------------------------------------
# Save Summary
# --------------------------------------------------

with open(f"{SUMMARY_PATH}/dataset_summary.txt", "w", encoding="utf-8") as file:

    file.write("APL Logistics Dataset Summary\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Shape : {df.shape}\n\n")

    file.write("Missing Values\n")
    file.write(str(df.isnull().sum()))
    file.write("\n\n")

    file.write("Duplicate Rows\n")
    file.write(str(df.duplicated().sum()))
    file.write("\n\n")

    file.write("Data Types\n")
    file.write(str(df.dtypes))
    file.write("\n\n")

print("\n✅ Dataset summary saved successfully.")