"""
Categorical Analysis
"""

import pandas as pd

from config import DATA_PATH
from visualization import save_bar_chart

# Load Dataset
df = pd.read_csv(DATA_PATH, encoding="latin1")

print("=" * 60)
print("Categorical Analysis")
print("=" * 60)

# Dictionary of charts
categorical_columns = {
    "Late_delivery_risk": ("Late Delivery Risk Distribution", "01_late_delivery_risk.png"),
    "Delivery Status": ("Delivery Status Distribution", "02_delivery_status.png"),
    "Shipping Mode": ("Shipping Mode Distribution", "03_shipping_mode.png"),
    "Customer Segment": ("Customer Segment Distribution", "04_customer_segment.png"),
    "Market": ("Market Distribution", "05_market_distribution.png"),
    "Order Status": ("Order Status Distribution", "06_order_status.png"),
    "Department Name": ("Department Distribution", "07_department_distribution.png"),
    "Category Name": ("Category Distribution", "08_category_distribution.png")
}

for column, (title, filename) in categorical_columns.items():

    counts = df[column].value_counts()

    save_bar_chart(
        counts,
        title,
        column,
        "Count",
        filename
    )

    print(f"✓ {column}")

print("\n✅ All categorical charts generated successfully.")