"""
Numerical Analysis
"""

import pandas as pd

from config import DATA_PATH
from visualization import save_histogram

# Load Dataset
df = pd.read_csv(DATA_PATH, encoding="latin1")

print("=" * 60)
print("Numerical Analysis")
print("=" * 60)

numerical_columns = {

    "Sales":
    ("Sales Distribution",
     "09_sales_distribution.png"),

    "Benefit per order":
    ("Benefit per Order Distribution",
     "10_benefit_distribution.png"),

    "Product Price":
    ("Product Price Distribution",
     "11_product_price_distribution.png"),

    "Order Profit Per Order":
    ("Order Profit Distribution",
     "12_order_profit_distribution.png"),

    "Order Item Quantity":
    ("Order Quantity Distribution",
     "13_order_quantity_distribution.png"),

    "Days for shipment (scheduled)":
    ("Scheduled Shipping Days",
     "14_scheduled_shipping_days.png"),

    "Days for shipping (real)":
    ("Actual Shipping Days",
     "15_actual_shipping_days.png")
}

for column, (title, filename) in numerical_columns.items():

    save_histogram(
        df[column],
        title,
        column,
        "Frequency",
        filename
    )

    print(f"✓ {column}")

print("\n✅ Numerical Analysis Completed Successfully.")