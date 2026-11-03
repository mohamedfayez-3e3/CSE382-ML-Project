from pathlib import Path
import yaml
from typing import Final
from enum import Enum

ROOT: Final = Path(__file__).resolve().parents[1]
def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)

class Task(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
