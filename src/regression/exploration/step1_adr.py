import pandas as pd

df = pd.read_csv("data/H2.csv", na_values=["NULL"])

print("Shape (rows, cols):", df.shape)
print()
print(df["ADR"].describe())
print()
print("Rows with ADR <= 0:", (df["ADR"] <= 0).sum())
print("Rows with ADR > 500:", (df["ADR"] > 500).sum())