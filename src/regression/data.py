import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_regression_data(cfg):
    """Load H2, apply regression-only cleaning, and split into train/test."""
    spec = cfg["features"]["regression"]
    num_cols, cat_cols, target = spec["numerical"], spec["categorical"], spec["target"]
    features = num_cols + cat_cols

    df = pd.read_csv(cfg["paths"]["dataset"], na_values=["NULL"])
    log = {"rows_start": len(df)}

    # Remove unrealistic prices (zero-priced bookings and the 5400 outlier)
    df = df[(df[target] > 0) & (df[target] <= 1000)]
    log["rows_after_adr_cleaning"] = len(df)

    # Remove exact duplicates so identical rows cannot fall in both train and test
    df = df.drop_duplicates(subset=features + [target])
    log["rows_after_duplicates"] = len(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target],
        test_size=cfg["split"]["test_size"],
        random_state=cfg["seed"],
    )
    log["train_shape"] = X_train.shape
    log["test_shape"] = X_test.shape
    return X_train, X_test, y_train, y_test, log