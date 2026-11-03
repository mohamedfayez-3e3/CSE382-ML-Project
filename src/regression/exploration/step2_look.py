import pandas as pd

df = pd.read_csv("data/H2.csv", na_values=["NULL"])

print("=== Rows with ADR > 500 ===")
print(df[df["ADR"] > 500][["ADR", "IsCanceled", "ArrivalDateMonth", "Adults", "ReservedRoomType", "MarketSegment"]])
print()

zero = df[df["ADR"] == 0]
print("=== Rows with ADR == 0 ===")
print("Count:", len(zero))
print()
print("By MarketSegment:")
print(zero["MarketSegment"].value_counts())
print()
print("By IsCanceled:")
print(zero["IsCanceled"].value_counts())
print()
print("By DepositType:")
print(zero["DepositType"].value_counts())