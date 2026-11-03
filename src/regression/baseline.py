import os

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from src.regression.data import load_config, load_regression_data
from src.regression.preprocessor import build_regression_preprocessor


def main():
    cfg = load_config()
    spec = cfg["features"]["regression"]
    X_train, X_test, y_train, y_test, log = load_regression_data(cfg)
    for key, value in log.items():
        print(f"{key}: {value}")

    preprocessor = build_regression_preprocessor(spec["numerical"], spec["categorical"])

    cv = KFold(n_splits=5, shuffle=True, random_state=cfg["seed"])
    scoring = {"MAE": "neg_mean_absolute_error",
               "RMSE": "neg_root_mean_squared_error",
               "R2": "r2"}

    rows = []
    for name, model in [("Dummy (mean)", DummyRegressor(strategy="mean")),
                        ("Linear Regression", LinearRegression())]:
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        rows.append({
            "model": name,
            "CV_MAE": -res["test_MAE"].mean(),
            "CV_RMSE": -res["test_RMSE"].mean(),
            "CV_R2": res["test_R2"].mean(),
            "CV_R2_std": res["test_R2"].std(),
        })

    results = pd.DataFrame(rows).round(3)
    print(results.to_string(index=False))

    os.makedirs("outputs", exist_ok=True)
    results.to_csv("outputs/regression_baseline_cv.csv", index=False)
    print("Saved: outputs/regression_baseline_cv.csv")


if __name__ == "__main__":
    main()