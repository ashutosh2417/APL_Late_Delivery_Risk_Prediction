"""
Data Loader
Loads all datasets and model artifacts for the Streamlit dashboard.
"""

import pandas as pd
import joblib
import streamlit as st


@st.cache_data
def load_processed_data():
    return pd.read_csv("data/processed_data.csv")


@st.cache_data
def load_model_comparison():
    return pd.read_csv("outputs/metrics/model_comparison.csv")


@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")


@st.cache_resource
def load_preprocessor():
    return joblib.load("models/preprocessor.pkl")
    

@st.cache_data
def load_risk_predictions():
    return pd.read_csv("outputs/risk_predictions.csv")
