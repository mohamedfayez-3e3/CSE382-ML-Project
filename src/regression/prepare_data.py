import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

cfg = yaml.safe_load(open("config.yaml"))
spec = cfg["features"]["regression"]

df = pd.read_csv(cfg["paths"]["dataset"], na_values=["NULL"])
print("Rows at start:", len(df))

# 1) Regression-only cleaning: remove unrealistic prices
df = df[(df["ADR"] > 0) & (df["ADR"] <= 1000)]
print("Rows after ADR cleaning:", len(df))

# 2) Remove exact duplicates (same features and same ADR)
#    so identical rows cannot land in both train and test
features = spec["numerical"] + spec["categorical"]
df = df.drop_duplicates(subset=features + [spec["target"]])
print("Rows after removing duplicates:", len(df))

# 3) Split into features (X) and target (y)
X = df[features]
y = df[spec["target"]]

# 4) Train/test split, using the seed and test size from config.yaml
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=cfg["split"]["test_size"],
    random_state=cfg["seed"],
)
print("Train:", X_train.shape, "| Test:", X_test.shape)