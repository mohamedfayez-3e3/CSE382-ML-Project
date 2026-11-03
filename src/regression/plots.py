import os

import matplotlib
matplotlib.use("Agg")  # save to file without opening a window
import matplotlib.pyplot as plt

from src.regression.data import load_config, load_regression_data

OUT_DIR = "outputs/regression"


def plot_adr_distribution(y_train):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_train, bins=50, edgecolor="white")
    median = y_train.median()
    ax.axvline(median, color="red", linestyle="--", label=f"Median = {median:.1f}")
    ax.set_xlabel("ADR (average daily rate)")
    ax.set_ylabel("Number of bookings")
    ax.set_title("Distribution of ADR (training set, after cleaning)")
    ax.legend()
    path = os.path.join(OUT_DIR, "adr_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def plot_adr_by_month(X_train, y_train):
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    monthly = y_train.groupby(X_train["ArrivalDateMonth"]).mean().reindex(month_order)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(monthly.index, monthly.values, edgecolor="white")
    ax.set_xlabel("Arrival month")
    ax.set_ylabel("Mean ADR")
    ax.set_title("Mean ADR by arrival month (training set)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    path = os.path.join(OUT_DIR, "adr_by_month.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def plot_adr_by_room_type(X_train, y_train):
    by_room = y_train.groupby(X_train["ReservedRoomType"]).mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(by_room.index.astype(str), by_room.values, edgecolor="white")
    ax.set_xlabel("Reserved room type")
    ax.set_ylabel("Mean ADR")
    ax.set_title("Mean ADR by reserved room type (training set)")
    path = os.path.join(OUT_DIR, "adr_by_room_type.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def main():
    cfg = load_config()
    X_train, X_test, y_train, y_test, log = load_regression_data(cfg)
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_adr_distribution(y_train)
    plot_adr_by_month(X_train, y_train)
    plot_adr_by_room_type(X_train, y_train)


if __name__ == "__main__":
    main()