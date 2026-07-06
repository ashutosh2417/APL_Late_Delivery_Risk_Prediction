"""
Data Preprocessing
Machine Learning-Based Late Delivery Risk Prediction
"""

import joblib
import pandas as pd

from config import PROCESSED_DATA_PATH

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# Load Processed Dataset
# ============================================================

df = pd.read_csv(PROCESSED_DATA_PATH)

print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# ============================================================
# Split Features and Target
# ============================================================

X = df.drop("Late_delivery_risk", axis=1)

y = df["Late_delivery_risk"]

print("✓ Features and Target Separated")

# ============================================================
# Identify Column Types
# ============================================================

categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()

numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

print(f"✓ Categorical Columns : {len(categorical_columns)}")
print(f"✓ Numerical Columns   : {len(numerical_columns)}")

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("✓ Train Test Split Completed")

# ============================================================
# Preprocessing Pipeline
# ============================================================

numeric_transformer = Pipeline(

    steps=[

        ("scaler", StandardScaler())

    ]

)

categorical_transformer = Pipeline(

    steps=[

        ("encoder",
         OneHotEncoder(
             handle_unknown="ignore"
         ))

    ]

)

preprocessor = ColumnTransformer(

    transformers=[

        ("num",
         numeric_transformer,
         numerical_columns),

        ("cat",
         categorical_transformer,
         categorical_columns)

    ]

)

# ============================================================
# Fit Preprocessor
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("✓ Data Transformed")

# ============================================================
# Save Feature Names
# ============================================================

feature_names = preprocessor.get_feature_names_out()

joblib.dump(feature_names, "models/feature_names.pkl")

print("✓ Feature Names Saved")

# ============================================================
# Save Objects
# ============================================================

joblib.dump(preprocessor, "models/preprocessor.pkl")

joblib.dump(X_train_processed, "models/X_train.pkl")

joblib.dump(X_test_processed, "models/X_test.pkl")

joblib.dump(y_train, "models/y_train.pkl")

joblib.dump(y_test, "models/y_test.pkl")

print("✓ Preprocessor Saved")

print("\n")
print("=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Training Samples :", X_train_processed.shape)

print("Testing Samples  :", X_test_processed.shape)