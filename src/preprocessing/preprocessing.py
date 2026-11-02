from pathlib import Path
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)

def load_data(path):
    df = pd.read_csv(path, na_values=["NULL"])
    return df

def preprocess():
    cfg = load_config()
    dataset_path = ROOT / cfg["paths"]["dataset"]
    df = load_data(dataset_path)
    return df

if __name__ == "__main__":
    df = preprocess()
    df.info()