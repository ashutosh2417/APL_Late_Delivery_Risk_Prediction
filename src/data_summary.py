import pandas as pd

# Load dataset
df = pd.read_csv("data/APL_Logistics.csv", encoding="latin1")

print("=" * 60)
print("COLUMN NAMES")
print("=" * 60)

for i, col in enumerate(df.columns, start=1):
    print(f"{i}. {col}")