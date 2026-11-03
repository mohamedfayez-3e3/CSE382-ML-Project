import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

cfg = yaml.safe_load(open("config.yaml"))
spec = cfg["features"]["regression"]
num_cols, cat_cols, target = spec["numerical"], spec["categorical"], spec["target"]

df = pd.read_csv(cfg["paths"]["dataset"], na_values=["NULL"])
print("Rows at start:", len(df))

# 1) Regression-only cleaning: remove unrealistic prices
df = df[(df[target] > 0) & (df[target] <= 1000)]
print("Rows after ADR cleaning:", len(df))

# 2) Remove exact duplicates (same features and same ADR)
#    so identical rows cannot land in both train and test
features = num_cols + cat_cols
df = df.drop_duplicates(subset=features + [target])
print("Rows after removing duplicates:", len(df))

# 3) Split into features (X) and target (y)
X = df[features]
y = df[target]

# 4) Train/test split, using the seed and test size from config.yaml
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=cfg["split"]["test_size"],
    random_state=cfg["seed"],
)
print("Train:", X_train.shape, "| Test:", X_test.shape)

# 5) Preprocessing: fill missing values, scale numbers, one-hot encode categories
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])

X_train_ready = preprocessor.fit_transform(X_train)   # learn from train, then transform
X_test_ready = preprocessor.transform(X_test)          # only transform, never learn

dense = X_train_ready.toarray() if hasattr(X_train_ready, "toarray") else X_train_ready
print("Before:", X_train.shape, "-> After:", X_train_ready.shape)
print("Missing values left in train:", int(np.isnan(dense).sum()))
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(name, model):
    model.fit(X_train_ready, y_train)
    for split, Xs, ys in [("train", X_train_ready, y_train), ("test", X_test_ready, y_test)]:
        pred = model.predict(Xs)
        print(f"{name:20s} {split:5s} "
              f"MAE={mean_absolute_error(ys, pred):7.2f}  "
              f"RMSE={np.sqrt(mean_squared_error(ys, pred)):7.2f}  "
              f"R2={r2_score(ys, pred):6.3f}")

evaluate("Dummy (mean)", DummyRegressor(strategy="mean"))
evaluate("Linear Regression", LinearRegression())