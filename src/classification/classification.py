"""Classification: predict hotel booking cancellation (IsCanceled)."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs" / "classification"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Load data
matches = list(ROOT.glob("data/**/H2.csv"))
if not matches:
    raise FileNotFoundError("H2.csv not found inside the data/ folder")
df = pd.read_csv(matches[0])
df.columns = df.columns.str.strip()
cols = {c.lower(): c for c in df.columns}

# 2. Target + remove leakage columns (they reveal the answer)
target = cols["iscanceled"]
leaky = [cols[c] for c in ("reservationstatus", "reservationstatusdate") if c in cols]
X = df.drop(columns=[target] + leaky)
y = df[target]

# 3. Split (stratified keeps the 58/42 class ratio in both parts)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Preprocessing
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

# 5. Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
}

# 6. Train + evaluate
results, best_name, best_auc, best_model = {}, None, -1, None
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }
    print(name, results[name])
    if results[name]["roc_auc"] > best_auc:
        best_name, best_auc, best_model = name, results[name]["roc_auc"], pipe

# 7. Save
results["best_model"] = best_name
(OUT / "metrics.json").write_text(json.dumps(results, indent=2))
joblib.dump(best_model, OUT / "best_model.joblib")
print("Best model:", best_name)