from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path.cwd().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "H2.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "audit"

PLOT_DIR = PROJECT_ROOT / "outputs" / "eda" / "figures"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)







print("\nDATASET OVERVIEW")

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nCOLUMN NAMES")
print(df.columns.tolist())





print("\nDATA TYPES")

print(df.dtypes)




lost_data = pd.DataFrame({
    "lost_count": df.isna().sum(),
    "lost_percent": df.isna().mean() * 100
})

print("\nMISSING VALUES ==>")
print(lost_data)



duplicate_count = df.duplicated().sum()

print("number of douplicate rows ==>: \n", duplicate_count)





# Summary for numerical columns
numerical_summary = df.describe(include="number")

# Summary for categorical columns
categorical_summary = df.describe(include=["str", "object", "category"])

print("\nNUMERICAL SUMMARY")
print(numerical_summary)

print("\nCATEGORICAL SUMMARY")
print(categorical_summary)




unusual_data = {
    "Negative ADR": (df["ADR"] < 0).sum(),
    "Negative LeadTime": (df["LeadTime"] < 0).sum(),
    "Negative Adults": (df["Adults"] < 0).sum(),
    "Negative Children": (df["Children"] < 0).sum(),
    "Negative Babies": (df["Babies"] < 0).sum(),
    "Zero Guests": (
        (df["Adults"] + df["Children"] + df["Babies"]) == 0
    ).sum(),
    "ADR Above 1000": (df["ADR"] > 1000).sum()
}

unusual_data = pd.Series(unusual_data)

print("\nUNUSUAL VALUES")
print(unusual_data)




cancellation_counts = df["IsCanceled"].value_counts()

print("\nCANCELLATION DISTRIBUTION")
print(cancellation_counts)




df["LeadTimeGroup"] = pd.cut(
    df["LeadTime"],
    bins=[-1, 7, 30, 90, 180, float("inf")],
    labels=["0-7", "8-30", "31-90", "91-180", "180+"]
)





cancellation_by_leadtime = (
    df.groupby("LeadTimeGroup", observed=False)["IsCanceled"]
    .agg(["mean", "count"])
)

cancellation_by_leadtime["cancellation_rate"] = (
    cancellation_by_leadtime["mean"] * 100
)

print(cancellation_by_leadtime)








cancellation_by_leadtime["cancellation_rate"].plot(
    kind="bar",
    title="Cancellation Rate by Lead Time",
    xlabel="Lead Time Group",
    ylabel="Cancellation Rate (%)"
)

plt.tight_layout()
plt.show()





lost_data.to_csv(OUTPUT_DIR / "lost_data.csv")