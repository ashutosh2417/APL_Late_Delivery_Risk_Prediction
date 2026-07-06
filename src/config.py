"""
Project Configuration
Machine Learning-Based Late Delivery Risk Prediction
"""

import os

# ==========================
# DATA PATHS
# ==========================

DATA_PATH = "data/APL_Logistics.csv"
PROCESSED_DATA_PATH = "data/processed_data.csv"

# ==========================
# OUTPUT PATHS
# ==========================

CHART_PATH = "reports/images"
SUMMARY_PATH = "outputs/summaries"
TABLE_PATH = "outputs/tables"
MODEL_PATH = "models"

EVALUATION_PATH = "outputs/evaluation"
EXPLAINABILITY_PATH = "outputs/explainability"

# ==========================
# CREATE FOLDERS
# ==========================

os.makedirs(CHART_PATH, exist_ok=True)
os.makedirs(SUMMARY_PATH, exist_ok=True)
os.makedirs(TABLE_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

os.makedirs(EVALUATION_PATH, exist_ok=True)
os.makedirs(EXPLAINABILITY_PATH, exist_ok=True)