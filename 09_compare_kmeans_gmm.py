
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

from config import DATA_DIR, TABLES_DIR, RANDOM_SEED


def evaluate(name, clusters, X, words):
    return {
        "algorithm": name,
        "silhouette": silhouette_score(X, clusters),
        "ARI_evaluation_only": adjusted_rand_score(
            words, clusters
        ),
        "NMI_evaluation_only": normalized_mutual_info_score(
            words, clusters
        )
    }


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_DIR / "pca_features.csv")
    pc_columns = [
        c for c in df.columns
        if c.startswith("PC") and c[2:].isdigit()
    ]

    X = df[pc_columns].to_numpy()
    words = df["word"].astype(str).to_numpy()

    results = []
    assignments = pd.DataFrame({"word": words})

    kmeans = KMeans(
        n_clusters=13,
        random_state=RANDOM_SEED,
        n_init=20
    )
    km_labels = kmeans.fit_predict(X)

    results.append(
        evaluate("KMeans k=13", km_labels, X, words)
    )
    assignments["kmeans_cluster"] = km_labels

    for covariance in ["diag", "spherical"]:
        gmm = GaussianMixture(
            n_components=13,
            covariance_type=covariance,
            random_state=RANDOM_SEED,
            n_init=3,
            reg_covar=1e-5
        )

        gmm_labels = gmm.fit_predict(X)

        results.append(
            evaluate(
                f"GMM {covariance} k=13",
                gmm_labels,
                X,
                words
            )
        )

        assignments[f"gmm_{covariance}_cluster"] = gmm_labels

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        TABLES_DIR / "kmeans_gmm_comparison.csv",
        index=False
    )

    assignments.to_csv(
        DATA_DIR / "kmeans_gmm_assignments.csv",
        index=False
    )

    print("\nAlgorithm comparison:")
    print(results_df.round(4).to_string(index=False))
    print("\nSaved table: outputs/tables/kmeans_gmm_comparison.csv")
    print("Saved assignments: data/kmeans_gmm_assignments.csv")


if __name__ == "__main__":
    main()
