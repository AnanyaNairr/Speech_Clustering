
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors

from config import DATA_DIR, FIGURES_DIR, TABLES_DIR, RANDOM_SEED

PCA_PATH = DATA_DIR / "pca_features.csv"
METRICS_PATH = TABLES_DIR / "clustering_metrics.csv"
LABELS_PATH = DATA_DIR / "cluster_assignments.csv"
DBSCAN_PATH = TABLES_DIR / "dbscan_parameter_search.csv"


def evaluate_clustering(X, clusters, true_labels):
    """Calculate internal and external clustering evaluation metrics."""
    clusters = np.asarray(clusters)
    unique_clusters = set(clusters)
    non_noise_mask = clusters != -1
    assigned_clusters = clusters[non_noise_mask]
    X_assigned = X[non_noise_mask]

    n_clusters = len(set(assigned_clusters))
    noise_ratio = float(np.mean(clusters == -1))

    silhouette = np.nan
    db_index = np.nan
    ch_index = np.nan

    # Internal metrics need at least two clusters and fewer clusters
    # than the number of samples being evaluated.
    if 2 <= n_clusters < len(X_assigned):
        try:
            silhouette = silhouette_score(
                X_assigned, assigned_clusters
            )
            db_index = davies_bouldin_score(
                X_assigned, assigned_clusters
            )
            ch_index = calinski_harabasz_score(
                X_assigned, assigned_clusters
            )
        except ValueError:
            pass

    # External metrics are calculated only for evaluation.
    ari = adjusted_rand_score(true_labels, clusters)
    nmi = normalized_mutual_info_score(true_labels, clusters)

    return {
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "silhouette_score": silhouette,
        "davies_bouldin_score": db_index,
        "calinski_harabasz_score": ch_index,
        "adjusted_rand_index": ari,
        "normalized_mutual_information": nmi,
    }


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not PCA_PATH.exists():
        raise FileNotFoundError(
            f"{PCA_PATH} not found. Run 03_prepare_features.py first."
        )

    df = pd.read_csv(PCA_PATH)

    if "word" not in df.columns:
        raise ValueError("The PCA file must contain the 'word' column.")

    # Word labels are reserved for evaluation only.
    true_labels = df["word"].astype(str).to_numpy()

    pca_columns = [
        col for col in df.columns
        if col.startswith("PC") and col[2:].isdigit()
    ]

    if len(pca_columns) < 2:
        raise ValueError("At least two PCA component columns are required.")

    X = df[pca_columns].to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError("PCA features contain non-finite values.")

    print("Dataset shape:", X.shape)
    print("Number of word classes (evaluation only):",
          len(np.unique(true_labels)))

    all_metrics = []
    assignments = pd.DataFrame({
        "path": df["path"] if "path" in df.columns else np.arange(len(df)),
        "word": true_labels,
    })
    dbscan_records = []

    # ---------------------------------------------------------
    # 1. K-Means
    # ---------------------------------------------------------
    print("\nRunning K-Means...")

    k_values = range(2, min(21, len(X)))

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            n_init=20,
        )
        clusters = model.fit_predict(X)

        scores = evaluate_clustering(X, clusters, true_labels)
        all_metrics.append({
            "algorithm": "KMeans",
            "parameters": f"k={k}",
            **scores,
        })
        assignments[f"kmeans_k{k}"] = clusters

    # ---------------------------------------------------------
    # 2. Agglomerative Clustering
    # ---------------------------------------------------------
    print("Running Agglomerative Clustering...")

    for k in k_values:
        model = AgglomerativeClustering(n_clusters=k)
        clusters = model.fit_predict(X)

        scores = evaluate_clustering(X, clusters, true_labels)
        all_metrics.append({
            "algorithm": "Agglomerative",
            "parameters": f"k={k}",
            **scores,
        })
        assignments[f"agglomerative_k{k}"] = clusters

    # ---------------------------------------------------------
    # 3. DBSCAN
    # Estimate useful eps values from k-nearest-neighbor distances.
    # ---------------------------------------------------------
    print("Searching DBSCAN parameters...")

    for min_samples in [5, 10, 15]:
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors.fit(X)
        distances, _ = neighbors.kneighbors(X)

        k_distances = distances[:, -1]
        eps_values = np.quantile(
            k_distances, [0.80, 0.85, 0.90, 0.95, 0.98]
        )
        eps_values = np.unique(np.round(eps_values, 6))

        for eps in eps_values:
            model = DBSCAN(
                eps=float(eps),
                min_samples=min_samples,
            )
            clusters = model.fit_predict(X)

            scores = evaluate_clustering(X, clusters, true_labels)

            params = f"eps={eps:.6f}, min_samples={min_samples}"

            record = {
                "algorithm": "DBSCAN",
                "parameters": params,
                **scores,
            }
            all_metrics.append(record)
            dbscan_records.append(record)

            assignments[
                f"dbscan_eps{eps:.6f}_min{min_samples}"
            ] = clusters

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_PATH, index=False)
    assignments.to_csv(LABELS_PATH, index=False)
    pd.DataFrame(dbscan_records).to_csv(DBSCAN_PATH, index=False)

    # ---------------------------------------------------------
    # Plot K-Means silhouette scores
    # ---------------------------------------------------------
    kmeans_df = metrics_df[
        metrics_df["algorithm"] == "KMeans"
    ].copy()

    kmeans_df["k"] = (
        kmeans_df["parameters"].str.extract(r"k=(\d+)").astype(int)
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        kmeans_df["k"],
        kmeans_df["silhouette_score"],
        marker="o",
    )
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("K-Means: Silhouette Score by Cluster Count")
    plt.xticks(kmeans_df["k"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "kmeans_silhouette_scores.png",
        dpi=200,
    )
    plt.close()

    # ---------------------------------------------------------
    # Print useful summaries
    # ---------------------------------------------------------
    print("\nBest K-Means result by silhouette score:")
    print(
        kmeans_df.sort_values(
            "silhouette_score", ascending=False
        ).head(5)[
            [
                "parameters",
                "n_clusters",
                "silhouette_score",
                "davies_bouldin_score",
                "adjusted_rand_index",
                "normalized_mutual_information",
            ]
        ].to_string(index=False)
    )

    valid_df = metrics_df.dropna(subset=["silhouette_score"])
    print("\nTop 10 results by silhouette score:")
    print(
        valid_df.sort_values(
            "silhouette_score", ascending=False
        ).head(10)[
            [
                "algorithm",
                "parameters",
                "n_clusters",
                "noise_ratio",
                "silhouette_score",
                "davies_bouldin_score",
                "adjusted_rand_index",
                "normalized_mutual_information",
            ]
        ].to_string(index=False)
    )

    print("\nClustering analysis complete.")
    print("Metrics:", METRICS_PATH)
    print("Assignments:", LABELS_PATH)
    print("DBSCAN search:", DBSCAN_PATH)
    print("Silhouette plot:",
          FIGURES_DIR / "kmeans_silhouette_scores.png")


if __name__ == "__main__":
    main()
