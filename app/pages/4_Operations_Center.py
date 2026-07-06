import streamlit as st

from style import load_css

load_css()

from data_loader import load_risk_predictions

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="Operations Center",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Operations Center")

st.markdown("""
Identify shipments requiring immediate operational attention
using predicted late delivery risk.
""")

st.divider()

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------

df = load_risk_predictions()

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

threshold = st.sidebar.slider(
    "Risk Threshold",
    min_value=0.40,
    max_value=1.00,
    value=0.70,
    step=0.05,
)

# -----------------------------------------------------
# FILTER
# -----------------------------------------------------

high_risk = df[
    df["Risk_Probability"] >= threshold
].copy()

# -----------------------------------------------------
# KPI
# -----------------------------------------------------

c1, c2, c3 = st.columns(3)

c1.metric(
    "High Risk Orders",
    len(high_risk)
)

c2.metric(
    "Average Risk",
    f"{high_risk['Risk_Probability'].mean()*100:.2f}%"
)

c3.metric(
    "Threshold",
    f"{threshold:.2f}"
)

st.divider()

# -----------------------------------------------------
# TABLE
# -----------------------------------------------------

st.subheader("📋 Priority Action Queue")

display_columns = [
    "Risk_Probability",
    "Risk_Category",
    "Shipping Mode",
    "Market",
    "Customer Segment",
    "Order Region",
    "Sales",
]

st.dataframe(
    high_risk[display_columns].sort_values(
        by="Risk_Probability",
        ascending=False,
    ),
    use_container_width=True,
)

# -----------------------------------------------------
# DOWNLOAD
# -----------------------------------------------------

csv = high_risk.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download High-Risk Orders",
    data=csv,
    file_name="high_risk_orders.csv",
    mime="text/csv",
)

st.divider()

# -----------------------------------------------------
# RECOMMENDATIONS
# -----------------------------------------------------

st.subheader("🚛 Recommended Operational Actions")

st.info("""
### 🔴 High Risk
- Prioritize shipment
- Notify customer immediately
- Consider Express Shipping
- Escalate to logistics supervisor

### 🟡 Medium Risk
- Monitor shipment closely
- Prepare contingency plan

### 🟢 Low Risk
- Continue normal operations
""")