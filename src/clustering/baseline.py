import src.preprocessing as pp
from src.config import *
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def main():
    cfg = load_config()
    df = pd.DataFrame(pp.get_preprocessed_data(Task.CLUSTERING))
    pipeline = pp.build_preprocessor(Task.CLUSTERING)
    X = pipeline.fit_transform(df)
    km = KMeans(n_clusters=4, n_init=10, random_state=cfg["seed"])
    labels = km.fit_predict(X)

    print(silhouette_score(X, labels, sample_size=10000, random_state=cfg["seed"]))
    print(davies_bouldin_score(X, labels))

if __name__ == "__main__":
    main()