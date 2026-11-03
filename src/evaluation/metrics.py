from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.config import load_config

def clustering_metrics(X, labels, sample_size=10000):
    """Return internal clustering metrics for a labeled feature matrix."""

    seed = load_config()["seed"]
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters <= 1:
        # if only one cluster:
        return {"silhouette": None, "davies_bouldin": None, "n_clusters": n_clusters}

    return {
        "silhouette": silhouette_score(X, labels, sample_size=sample_size, random_state=seed),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "n_clusters": n_clusters,
    }