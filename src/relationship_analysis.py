"""
Relationship Analysis
"""

import pandas as pd

from config import DATA_PATH
from visualization import save_grouped_bar_chart

# Load dataset
df = pd.read_csv(DATA_PATH, encoding="latin1")

print("=" * 60)
print("Relationship Analysis")
print("=" * 60)

relationships = {

    "Shipping Mode":
        "16_risk_shipping_mode.png",

    "Market":
        "17_risk_market.png",

    "Customer Segment":
        "18_risk_customer_segment.png",

    "Order Region":
        "19_risk_order_region.png"

}

for column, filename in relationships.items():

    result = pd.crosstab(
        df[column],
        df["Late_delivery_risk"]
    )

    save_grouped_bar_chart(
        result,
        f"Late Delivery Risk by {column}",
        column,
        "Number of Orders",
        filename
    )

    print(f"✓ {column}")

print("\n✅ Relationship Analysis Completed Successfully.")