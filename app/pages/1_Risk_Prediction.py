import streamlit as st
import pandas as pd

from style import load_css

load_css()

from data_loader import load_processed_data
from prediction import predict_risk

# -------------------------
# Load Data
# -------------------------

df = load_processed_data()

st.title("🎯 Order-Level Risk Prediction")

st.markdown(
    "Select an order from the processed dataset and predict its late delivery risk."
)

# -------------------------
# Select Order
# -------------------------

order_index = st.number_input(
    "Select Order Index",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1,
)

selected = df.iloc[[order_index]]

features = selected.drop(columns=["Late_delivery_risk"])

actual = int(selected["Late_delivery_risk"].iloc[0])

st.subheader("Selected Order")

st.dataframe(features)

# -------------------------
# Prediction
# -------------------------

if st.button("🔮 Predict Late Delivery Risk"):

    prediction, probability = predict_risk(features)

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Late Delivery Probability",
            f"{probability*100:.2f}%"
        )

    with col2:

        if probability < 0.40:
            risk = "🟢 Low"

        elif probability < 0.70:
            risk = "🟡 Medium"

        else:
            risk = "🔴 High"

        st.metric("Risk Category", risk)

    st.write("### Actual Label")

    st.write(actual)

    st.write("### Model Prediction")

    st.write(prediction)

    st.write("### Recommendation")

    if probability >= 0.70:

        st.error(
            "Prioritize shipment, notify customer and consider faster shipping."
        )

    elif probability >= 0.40:

        st.warning(
            "Monitor shipment and keep customer informed."
        )

    else:

        st.success(
            "Shipment is likely to arrive on time."
        )