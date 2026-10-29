
import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.neighbors import NearestNeighbors

from config import DATA_DIR, TABLES_DIR


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_DIR / "pca_features.csv")

    pc_columns = [
        c for c in df.columns
        if c.startswith("PC") and c[2:].isdigit()
    ]

    X = df[pc_columns].to_numpy()
    words = df["word"].astype(str).to_numpy()

    # Estimate useful eps values from nearest-neighbour distances.
    min_samples_values = [5, 10, 15]
    eps_values = set()

    for min_samples in min_samples_values:
        nn = NearestNeighbors(
            n_neighbors=min_samples
        ).fit(X)

        distances, _ = nn.kneighbors(X)
        kth_distances = np.sort(distances[:, -1])

        for q in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
            eps_values.add(
                float(np.quantile(kth_distances, q))
            )

    results = []

    for min_samples in min_samples_values:
        for eps in sorted(eps_values):
            model = DBSCAN(
                eps=eps,
                min_samples=min_samples
            )
            labels = model.fit_predict(X)

            non_noise = labels != -1
            unique_clusters = set(labels[non_noise])
            n_clusters = len(unique_clusters)
            noise_ratio = float((~non_noise).mean())

            # Silhouette is calculated only on non-noise points.
            # It is undefined if fewer than 2 clusters remain.
            silhouette = np.nan

            if n_clusters >= 2 and non_noise.sum() > n_clusters:
                silhouette = silhouette_score(
                    X[non_noise], labels[non_noise]
                )

            results.append({
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters_excluding_noise": n_clusters,
                "noise_ratio": noise_ratio,
                "silhouette_non_noise": silhouette,
                "ARI_all_samples": adjusted_rand_score(
                    words, labels
                ),
                "NMI_all_samples": normalized_mutual_info_score(
                    words, labels
                )
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        TABLES_DIR / "dbscan_comparison.csv",
        index=False
    )

    print("\nDBSCAN configurations with 2+ clusters and at most 50% noise:")
    candidates = result_df[
        (result_df["n_clusters_excluding_noise"] >= 2)
        & (result_df["noise_ratio"] <= 0.50)
    ]

    if candidates.empty:
        print("No configurations met these conditions.")
    else:
        print(
            candidates.sort_values(
                "silhouette_non_noise",
                ascending=False
            ).head(10).round(4).to_string(index=False)
        )

    print("\nSaved: outputs/tables/dbscan_comparison.csv")


if __name__ == "__main__":
    main()
