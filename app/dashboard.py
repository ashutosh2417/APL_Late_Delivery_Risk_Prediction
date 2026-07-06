import streamlit as st
import plotly.express as px

from style import load_css

load_css()

st.markdown("""
<div style="
background: linear-gradient(90deg,#0F172A,#1E3A8A);
padding:25px;
border-radius:15px;
margin-bottom:25px;
">

<h1 style="
color:white;
margin:0;
">
🚚 APL Logistics
</h1>

<h3 style="
color:#CBD5E1;
margin-top:8px;
">
Machine Learning-Based Late Delivery Risk Prediction
</h3>

<p style="
color:#E2E8F0;
font-size:18px;
">
Business Analytics Internship Project
</p>

</div>
""", unsafe_allow_html=True)

from data_loader import (
    load_processed_data,
    load_model_comparison,
)

from config import *

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="APL Logistics Dashboard",
    page_icon="🚚",
    layout="wide",
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = load_processed_data()
metrics = load_model_comparison()

# ----------------------------
# PROFESSIONAL HEADER
# ----------------------------

st.markdown("""
<div style="...">
...
</div>
""", unsafe_allow_html=True)

# 👇 ADD THE PROJECT HIGHLIGHTS HERE

st.markdown("## 📌 Project Highlights")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.info("📦 **180,519 Orders Analyzed**")

with c2:
    st.success("🤖 **Best Model:** Logistic Regression")

with c3:
    st.warning("🎯 **ROC-AUC:** 0.8156")

with c4:
    st.info("⚡ **Interactive Analytics Dashboard**")

st.divider()


# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

total_orders = len(df)

late_orders = int(df["Late_delivery_risk"].sum())

late_rate = late_orders / total_orders * 100

best_model = metrics.iloc[0]["Model"]

roc_auc = metrics.iloc[0]["ROC-AUC"]

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("📦 Total Orders", f"{total_orders:,}")

c2.metric("🚨 Late Deliveries", f"{late_orders:,}")

c3.metric("📈 Delay Rate", f"{late_rate:.2f}%")

c4.metric("🏆 Best Model", best_model)

c5.metric("🎯 ROC-AUC", f"{roc_auc:.4f}")

st.divider()

# --------------------------------------------------
# CHARTS
# --------------------------------------------------

left, right = st.columns(2)

with left:

    fig = px.pie(
        df,
        names="Late_delivery_risk",
        title="Late Delivery Distribution",
        hole=0.45,
    )

    st.plotly_chart(fig, use_container_width=True)

with right:

    shipping = (
        df["Shipping Mode"]
        .value_counts()
        .reset_index()
    )

    shipping.columns = ["Shipping Mode", "Orders"]

    fig = px.bar(
        shipping,
        x="Shipping Mode",
        y="Orders",
        title="Shipping Mode Distribution",
    )
fig.update_layout(
    template="plotly_white",
    title_x=0.5,
    title_font_size=20,
    margin=dict(l=20, r=20, t=60, b=20),
    legend_title_text=""
)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)

with left:

    market = (
        df["Market"]
        .value_counts()
        .reset_index()
    )

    market.columns = ["Market", "Orders"]

    fig = px.bar(
        market,
        x="Market",
        y="Orders",
        title="Market Distribution",
    )
fig.update_layout(
    template="plotly_white",
    title_x=0.5,
    title_font_size=20,
    margin=dict(l=20, r=20, t=60, b=20),
    legend_title_text=""
)
st.plotly_chart(fig, use_container_width=True)

with right:

    segment = (
        df["Customer Segment"]
        .value_counts()
        .reset_index()
    )

    segment.columns = ["Customer Segment", "Orders"]

    fig = px.bar(
        segment,
        x="Customer Segment",
        y="Orders",
        title="Customer Segment Distribution",
    )
fig.update_layout(
    template="plotly_white",
    title_x=0.5,
    title_font_size=20,
    margin=dict(l=20, r=20, t=60, b=20),
    legend_title_text=""
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------
# DATASET SUMMARY
# --------------------------------------------------

st.subheader("Dataset Summary")

a, b, c = st.columns(3)

a.metric("Rows", f"{df.shape[0]:,}")

b.metric("Columns", df.shape[1])

c.metric(
    "Target",
    "Late_delivery_risk",
)

st.success(
    "Use the pages in the left sidebar to explore predictions, analytics, model performance, and operations."
)
st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748B; padding:20px;">
        <h4>🚚 APL Logistics – Late Delivery Risk Prediction</h4>
        <p>
            Developed by <strong>Ashutosh Sharma</strong><br>
            MBA (Business Analytics) | Amity University Online<br>
            Business Analytics Internship Project • 2026
        </p>
    </div>
    """,
    unsafe_allow_html=True
)