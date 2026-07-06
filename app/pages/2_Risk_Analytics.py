import streamlit as st
import pandas as pd
import plotly.express as px

from style import load_css

load_css()

from data_loader import load_processed_data

st.set_page_config(page_title="Risk Analytics", layout="wide")

df = load_processed_data()

st.title("📊 Risk Analytics")

st.markdown(
    "Analyze late delivery risk across shipping modes, markets, customer segments and regions."
)

# --------------------------------------------------
# Sidebar Filters
# --------------------------------------------------

shipping = st.sidebar.multiselect(
    "Shipping Mode",
    sorted(df["Shipping Mode"].unique()),
    default=sorted(df["Shipping Mode"].unique())
)

market = st.sidebar.multiselect(
    "Market",
    sorted(df["Market"].unique()),
    default=sorted(df["Market"].unique())
)

segment = st.sidebar.multiselect(
    "Customer Segment",
    sorted(df["Customer Segment"].unique()),
    default=sorted(df["Customer Segment"].unique())
)

filtered = df[
    (df["Shipping Mode"].isin(shipping)) &
    (df["Market"].isin(market)) &
    (df["Customer Segment"].isin(segment))
]

# --------------------------------------------------
# KPIs
# --------------------------------------------------

k1, k2, k3 = st.columns(3)

k1.metric("Orders", len(filtered))

k2.metric(
    "Late Deliveries",
    int(filtered["Late_delivery_risk"].sum())
)

k3.metric(
    "Late Delivery %",
    f"{filtered['Late_delivery_risk'].mean()*100:.2f}%"
)

st.divider()

# --------------------------------------------------
# Shipping Mode
# --------------------------------------------------

left, right = st.columns(2)

with left:

    fig = px.histogram(
        filtered,
        x="Shipping Mode",
        color="Late_delivery_risk",
        barmode="group",
        title="Late Deliveries by Shipping Mode"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:

    fig = px.histogram(
        filtered,
        x="Market",
        color="Late_delivery_risk",
        title="Late Deliveries by Market"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Customer Segment
# --------------------------------------------------

left, right = st.columns(2)

with left:

    fig = px.histogram(
        filtered,
        x="Customer Segment",
        color="Late_delivery_risk",
        title="Late Deliveries by Customer Segment"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:

    fig = px.histogram(
        filtered,
        x="Order Region",
        color="Late_delivery_risk",
        title="Late Deliveries by Region"
    )

    st.plotly_chart(fig, use_container_width=True)