import streamlit as st

from style import load_css

load_css()

st.set_page_config(
    page_title="About Project",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About the Project")

st.markdown("""
## 🚚 Machine Learning-Based Late Delivery Risk Prediction

This project was developed to help **APL Logistics (KWE Group)** identify shipments that are likely to be delivered late before dispatch.

The solution combines **Business Analytics, Machine Learning, and Interactive Dashboards** to support data-driven logistics decisions.
""")

st.divider()

# -------------------------------------------------------
# BUSINESS PROBLEM
# -------------------------------------------------------

st.header("🎯 Business Problem")

st.write("""
Late deliveries increase operational costs, reduce customer satisfaction,
and negatively impact supply chain performance.

This project predicts the probability of shipment delays before dispatch,
allowing logistics managers to take proactive actions.
""")

# -------------------------------------------------------
# OBJECTIVES
# -------------------------------------------------------

st.header("🎯 Project Objectives")

st.markdown("""
- Predict late delivery risk using Machine Learning
- Identify high-risk shipments
- Support proactive logistics planning
- Improve customer satisfaction
- Reduce operational delays
- Provide an interactive decision-support dashboard
""")

# -------------------------------------------------------
# DATASET
# -------------------------------------------------------

st.header("📊 Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Records", "180,519")
    st.metric("Features", "32")

with col2:
    st.metric("Target Variable", "Late_delivery_risk")
    st.metric("Domain", "Supply Chain & Logistics")

st.divider()

# -------------------------------------------------------
# MACHINE LEARNING
# -------------------------------------------------------

st.header("🤖 Machine Learning Models")

st.markdown("""
The following Machine Learning models were developed and evaluated:

- Logistic Regression ✅ (Best Model)
- Random Forest
- Gradient Boosting
- XGBoost
""")

st.success("""
Best Performing Model:
**Logistic Regression**
""")

# -------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------

st.header("⚙️ Feature Engineering")

st.markdown("""
Custom engineered features include:

- Shipping Pressure Index
- Mode Risk Flag
- Order Complexity Score
- Regional Congestion
""")

# -------------------------------------------------------
# TECHNOLOGY STACK
# -------------------------------------------------------

st.header("💻 Technology Stack")

st.markdown("""
### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Plotly
- Matplotlib

### Machine Learning

- Scikit-learn
- XGBoost

### Dashboard

- Streamlit
""")

# -------------------------------------------------------
# BUSINESS IMPACT
# -------------------------------------------------------

st.header("📈 Business Value")

st.markdown("""
This solution helps logistics managers:

- Reduce shipment delays
- Improve operational efficiency
- Prioritize high-risk shipments
- Allocate resources effectively
- Enhance customer satisfaction
""")

# -------------------------------------------------------
# FUTURE ENHANCEMENTS
# -------------------------------------------------------

st.header("🚀 Future Enhancements")

st.markdown("""
Possible future improvements include:

- Real-time shipment monitoring
- Live API integration
- Weather data integration
- Route optimization
- Explainable AI (SHAP)
- Cloud deployment
""")

st.divider()

st.success(
    "Developed as part of the Business Analytics Internship Project."
)