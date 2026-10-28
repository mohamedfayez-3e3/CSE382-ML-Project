
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# 1.project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "H2.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "audit"
PLOT_DIR = PROJECT_ROOT / "outputs" / "eda" / "figures"


def main():
    # 2.check if the data is there or not
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # 3.load data
    df = pd.read_csv(DATA_PATH)

    # 4.basic dataset information
    print("\n DATASET OVERVIEW ==>")
    print(f"Rows ==> {df.shape[0]}")
    print(f"Columns ==> {df.shape[1]}")
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\n DATA TYPES ==>")
    print(df.dtypes.to_string())

    # 5.lost data
    lost_data = pd.DataFrame({
        "lost_count": df.isna().sum(),
        "lost_percent": df.isna().mean() * 100
    })

    lost_data = lost_data.sort_values(
        "lost_count", ascending=False
    )

    print("\n lost data ==>")
    print(lost_data.to_string())

    # 6.duplicate data
    duplicate_count = int(df.duplicated().sum())

    print("\n duplicate data ==>")
    print(f"duplicate rows ==> {duplicate_count}")


    # 7.numerical and categorical summaries
    numerical_summary = df.describe(include="number")
    categorical_summary = df.describe(
        include=["str", "object", "category"]
    )

    print("\n NUMERICAL SUMMARY ==>")
    print(numerical_summary.to_string())

    print("\n CATEGORICAL SUMMARY ==>")
    print(categorical_summary.to_string())

    # 8.check unusual values
    unusual_data = {
        "Negative ADR": df["ADR"] < 0,
        "Negative LeadTime": df["LeadTime"] < 0,
        "Negative Adults": df["Adults"] < 0,
        "Negative Children": df["Children"] < 0,
        "Negative Babies": df["Babies"] < 0,
        "Zero Guests": (
            (df["Adults"] == 0)
            & (df["Children"].fillna(0) == 0)
            & (df["Babies"] == 0)
        ),
        "ADR Above 1000": df["ADR"] > 1000
    }

    unusual_summary = pd.Series({
        name: int(values.sum())
        for name, values in unusual_data.items()
    }, name="count")

    print("\n unusual values ==>")
    print(unusual_summary.to_string())

    # 9.IsCanceled distribution
    canceled_count = df["IsCanceled"].value_counts().sort_index()
    canceled_percent = (
        df["IsCanceled"].value_counts(normalize=True).sort_index() * 100
    )

    canceled_summary = pd.DataFrame({
        "count": canceled_count,
        "percent": canceled_percent
    })

    print("\n is canceled ==>")
    print(canceled_summary.to_string())

    # 10.save audit results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    column_summary = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(dtype) for dtype in df.dtypes],
        "lost_count": df.isna().sum().values,
        "lost_percent": df.isna().mean().values * 100
    })

    column_summary.to_csv(
        OUTPUT_DIR / "column_summary.csv",
        index=False
    )

    lost_data.to_csv(OUTPUT_DIR / "lost_data.csv")
    unusual_summary.to_csv(OUTPUT_DIR / "unusual_data.csv")
    canceled_summary.to_csv(OUTPUT_DIR / "is_canceled_summary.csv")

    # 11.save audit report
    report_path = OUTPUT_DIR / "audit_report.txt"

    with open(report_path, "w", encoding="utf-8") as report:
        report.write("DATASET AUDIT REPORT\n")
        report.write("====================\n")
        report.write(f"Rows: {df.shape[0]}\n")
        report.write(f"Columns: {df.shape[1]}\n")
        report.write(f"Duplicate rows: {duplicate_count}\n")

        report.write("\nCOLUMN SUMMARY\n")
        report.write(column_summary.to_string(index=False))

        report.write("\n\nLOST DATA\n")
        report.write(lost_data.to_string())

        report.write("\n\nNUMERICAL SUMMARY\n")
        report.write(numerical_summary.to_string())

        report.write("\n\nCATEGORICAL SUMMARY\n")
        report.write(categorical_summary.to_string())

        report.write("\n\nUNUSUAL VALUES\n")
        report.write(unusual_summary.to_string())

        report.write("\n\nIS CANCELED DISTRIBUTION\n")
        report.write(canceled_summary.to_string())

    # 12.create plots
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df["IsCanceled"].value_counts().sort_index().plot(
        kind="bar", rot=0
    )
    plt.title("Booking Cancellation Distribution")
    plt.xlabel("IsCanceled")
    plt.ylabel("Number of Bookings")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cancellation_distribution.png", dpi=150)
    plt.close()

    df["LeadTime"].plot(kind="hist", bins=40)
    plt.title("Lead Time Distribution")
    plt.xlabel("LeadTime (days)")
    plt.ylabel("Number of Bookings")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "lead_time_distribution.png", dpi=150)
    plt.close()

    df.loc[df["ADR"].between(0, 500), "ADR"].plot(
        kind="hist", bins=40
    )
    plt.title("ADR Distribution (0 to 500)")
    plt.xlabel("ADR")
    plt.ylabel("Number of Bookings")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "adr_distribution.png", dpi=150)
    plt.close()



    # 13.analyze cancellation relationships

    # Cancellation rate by lead time
    df["LeadTimeGroup"] = pd.cut(
        df["LeadTime"],
        bins=[-1, 7, 30, 90, 180, 365, float("inf")],
        labels=[
            "0-7 days",
            "8-30 days",
            "31-90 days",
            "91-180 days",
            "181-365 days",
            "Over 365 days"
        ]
    )

    leadtime_analysis = df.groupby(
        "LeadTimeGroup", observed=False
    )["IsCanceled"].agg(["mean", "count"])

    leadtime_analysis["cancellation_rate"] = (
        leadtime_analysis["mean"] * 100
    )

    print("\n CANCELLATION BY LEAD TIME ==>")
    print(leadtime_analysis.to_string())

    leadtime_analysis.to_csv(
        OUTPUT_DIR / "cancellation_by_leadtime.csv"
    )

    leadtime_analysis["cancellation_rate"].plot(kind="bar")
    plt.title("Cancellation Rate by Lead Time")
    plt.xlabel("Lead Time Group")
    plt.ylabel("Cancellation Rate (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "cancellation_by_leadtime.png", dpi=150
    )
    plt.close()

    # Cancellation rate by deposit type
    deposit_analysis = df.groupby(
        "DepositType", observed=True
    )["IsCanceled"].agg(["mean", "count"])

    deposit_analysis["cancellation_rate"] = (
        deposit_analysis["mean"] * 100
    )

    print("\n CANCELLATION BY DEPOSIT TYPE ==>")
    print(deposit_analysis.to_string())

    deposit_analysis.to_csv(
        OUTPUT_DIR / "cancellation_by_deposit.csv"
    )

    deposit_analysis["cancellation_rate"].sort_values().plot(
        kind="bar"
    )
    plt.title("Cancellation Rate by Deposit Type")
    plt.xlabel("Deposit Type")
    plt.ylabel("Cancellation Rate (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "cancellation_by_deposit.png", dpi=150
    )
    plt.close()

    # Cancellation rate by market segment
    market_analysis = df.groupby(
        "MarketSegment", observed=True
    )["IsCanceled"].agg(["mean", "count"])

    market_analysis["cancellation_rate"] = (
        market_analysis["mean"] * 100
    )

    print("\n CANCELLATION BY MARKET SEGMENT ==>")
    print(market_analysis.sort_values(
        "cancellation_rate", ascending=False
    ).to_string())

    market_analysis.to_csv(
        OUTPUT_DIR / "cancellation_by_market_segment.csv"
    )

    market_analysis["cancellation_rate"].sort_values().plot(
        kind="bar"
    )
    plt.title("Cancellation Rate by Market Segment")
    plt.xlabel("Market Segment")
    plt.ylabel("Cancellation Rate (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "cancellation_by_market_segment.png", dpi=150
    )
    plt.close()


    print(f"\nAudit files saved in: {OUTPUT_DIR}")
    print(f"Plots saved in: {PLOT_DIR}")
    print("The original dataset was not modified.")


if __name__ == "__main__":
    main()
