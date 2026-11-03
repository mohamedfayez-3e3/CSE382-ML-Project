import pandas as pd, yaml

cfg = yaml.safe_load(open("config.yaml"))
df = pd.read_csv("data/H2.csv", na_values=["NULL"])
spec = cfg["features"]["regression"]

cols = [spec["target"]] + spec["numerical"] + spec["categorical"] + spec["drop"]
print("أعمدة مش موجودة:", [c for c in cols if c not in df.columns])
print("أعمدة الملف:", list(df.columns))