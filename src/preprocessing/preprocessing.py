import pandas as pd
from src.config import ROOT, load_config, Task
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def load_data(path) -> pd.DataFrame:
    """Reads dataframe from the dataset."""
    df = pd.read_csv(path, na_values=["NULL"])
    return df

def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Deletes useless/harmful rows."""
    df = df.copy() # safety first

    # NAs
    df["Children"] = (df["Children"].fillna(0)).astype("int64")
    df["Country"] = df["Country"].fillna("Unknown")

    # duplicates
    # df = df.drop_duplicates() # idk what i should do with them

    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds extra features to use."""
    df = df.copy() # safety first
    return df

def get_feature_sets(task: Task, cfg, df = None):
    """Returns target, numerical_features, categorical_features"""
    
    spec = cfg["features"][task.value]
    target, num, cat = spec["target"], spec["numerical"], spec["categorical"]

    if target in num + cat:
        raise ValueError(f"Target {target} used as a feature")
    
    if df is not None:
        missing = [c for c in num + cat if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")

    return target, num, cat

def split_data(df, task: Task, cfg):
    """Returns X_train, X_test, y_train, y_test."""
    target, num, cat = get_feature_sets(task, cfg, df)
    X, y = df[num + cat], df[target]

    return train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["seed"],
        stratify=y if task == Task.CLASSIFICATION else None,
    )

def build_regression_preprocessor(cfg) -> ColumnTransformer:
    """Returns preprocessing pipeline for the regression model."""
        
    target, num, cat = get_feature_sets(Task.REGRESSION, cfg)

    return ColumnTransformer([])

def build_classification_preprocessor(cfg) -> ColumnTransformer:
    """Returns preprocessing pipeline for the classification model."""
    
    target, num, cat = get_feature_sets(Task.CLASSIFICATION, cfg)

    return ColumnTransformer([])

def build_clustering_preprocessor(cfg, scale: bool = True) -> ColumnTransformer:
    """Returns preprocessing pipeline for the clustering model."""

    _, num, cat = get_feature_sets(Task.CLUSTERING, cfg)

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        [("num", num_pipe, num), ("cat", cat_pipe, cat)],
        remainder="drop",
    )

def build_preprocessor(task: Task) -> ColumnTransformer:
    """Returns a pipeline for your task"""
    cfg = load_config()
    if task == Task.REGRESSION:
        return build_regression_preprocessor(cfg)
    elif task == Task.CLASSIFICATION:
        return build_classification_preprocessor(cfg)
    elif task == Task.CLUSTERING:
        return build_clustering_preprocessor(cfg)
    raise ValueError(f"Unknown task: {task}")    

def get_preprocessed_data(task: Task):
    """
    For supervised tasks: returns (X_train, X_test, y_train, y_test)
    For unsupervised tasks: return df[selected_features]
    """
    cfg = load_config()
    df = load_data(ROOT / cfg["paths"]["dataset"])
    df = clean_rows(df)
    df = add_features(df)
    if task == Task.CLUSTERING:
        _, num, cat = get_feature_sets(Task.CLUSTERING, cfg, df)
        return df[num + cat]
    else:
        return split_data(df, task, cfg)

def main():
    build_preprocessor(Task.CLUSTERING)
    t = get_preprocessed_data(Task.CLUSTERING)
    if type(t) != pd.DataFrame:
        raise ValueError("wtf")
    print(t.info())

if __name__ == "__main__":
    main()