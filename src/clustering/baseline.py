import src.preprocessing as pp
from src.config import *
from src.evaluation.metrics import clustering_metrics
import pandas as pd
from sklearn.cluster import KMeans

def main():
    cfg = load_config()
    df = pd.DataFrame(pp.get_preprocessed_data(Task.CLUSTERING))
    pipeline = pp.build_preprocessor(Task.CLUSTERING)

    X = pipeline.fit_transform(df)
    km = KMeans(
        n_clusters=4,
        n_init=10,
        random_state=cfg["seed"]
    )
    labels = km.fit_predict(X)

    metrics = clustering_metrics(X, labels)
    print(metrics)

if __name__ == "__main__":
    main()