import json
import math
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.regression.data import load_config, load_regression_data
from src.regression.preprocessor import build_regression_preprocessor

CV_PATH = "outputs/regression_cv_results.csv"
TEST_PATH = "outputs/regression_test_result.csv"
SCORING = {"MAE": "neg_mean_absolute_error", "MSE": "neg_mean_squared_error",
           "RMSE": "neg_root_mean_squared_error", "R2": "r2"}


def get_models(seed):
    # name: (baseline model, model to tune, search space)
    return {
        "Linear Regression": (LinearRegression(), Ridge(),
                              {"model__alpha": [0.1, 1, 10, 100, 1000]}),
        "Random Forest": (RandomForestRegressor(random_state=seed, n_jobs=-1),
                          RandomForestRegressor(random_state=seed, n_jobs=-1),
                          {"model__n_estimators": [100, 200],
                           "model__max_depth": [10, 20, None],
                           "model__min_samples_leaf": [1, 3, 5]}),
        "XGBoost": (XGBRegressor(random_state=seed, n_jobs=-1),
                    XGBRegressor(random_state=seed, n_jobs=-1),
                    {"model__n_estimators": [200, 400],
                     "model__max_depth": [3, 5, 7],
                     "model__learning_rate": [0.05, 0.1, 0.2],
                     "model__subsample": [0.8, 1.0]}),
    }


def make_pipe(model, spec):
    prep = build_regression_preprocessor(spec["numerical"], spec["categorical"])
    return Pipeline([("prep", prep), ("model", model)])


def cv_scores(pipe, X, y, cv):
    r = cross_validate(pipe, X, y, cv=cv, scoring=SCORING)
    return {"CV_MAE": -r["test_MAE"].mean(), "CV_MSE": -r["test_MSE"].mean(),
            "CV_RMSE": -r["test_RMSE"].mean(), "CV_R2": r["test_R2"].mean(),
            "CV_R2_std": r["test_R2"].std()}


def run_cv(n_iter):
    cfg = load_config()
    spec, seed = cfg["features"]["regression"], cfg["seed"]
    X_train, _, y_train, _, _ = load_regression_data(cfg)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for name, (base, tuned, grid) in get_models(seed).items():
        print("===", name)
        s = cv_scores(make_pipe(base, spec), X_train, y_train, cv)
        rows.append({"model": name, "config": "baseline", **s, "best_params": "{}"})
        search = RandomizedSearchCV(
            make_pipe(tuned, spec), grid,
            n_iter=min(n_iter, math.prod(len(v) for v in grid.values())),
            scoring="neg_root_mean_squared_error", cv=cv, random_state=seed)
        search.fit(X_train, y_train)
        s = cv_scores(search.best_estimator_, X_train, y_train, cv)
        rows.append({"model": name, "config": "tuned", **s,
                     "best_params": json.dumps(search.best_params_)})
        print(search.best_params_)
    res = pd.DataFrame(rows)
    res[["CV_MAE", "CV_MSE", "CV_RMSE", "CV_R2", "CV_R2_std"]] = res[
        ["CV_MAE", "CV_MSE", "CV_RMSE", "CV_R2", "CV_R2_std"]].round(3)
    res.to_csv(CV_PATH, index=False)
    print(res.drop(columns="best_params").to_string(index=False))
    print("Saved:", CV_PATH)


def run_final():
    cfg = load_config()
    spec, seed = cfg["features"]["regression"], cfg["seed"]
    best = pd.read_csv(CV_PATH).sort_values("CV_RMSE").iloc[0]
    print("Selected by CV:", best["model"], best["config"])
    base, tuned, _ = get_models(seed)[best["model"]]
    pipe = make_pipe(base if best["config"] == "baseline" else tuned, spec)
    pipe.set_params(**json.loads(best["best_params"]))
    X_train, X_test, y_train, y_test, _ = load_regression_data(cfg)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    out = pd.DataFrame([{"model": best["model"], "config": best["config"],
                         "Test_MAE": round(mean_absolute_error(y_test, pred), 3),
                         "Test_MSE": round(mse, 3),
                         "Test_RMSE": round(math.sqrt(mse), 3),
                         "Test_R2": round(r2_score(y_test, pred), 3)}])
    out.to_csv(TEST_PATH, index=False)
    print(out.to_string(index=False))
    print("Saved:", TEST_PATH)


if __name__ == "__main__":
    if "final" in sys.argv:
        run_final()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10
        run_cv(n)