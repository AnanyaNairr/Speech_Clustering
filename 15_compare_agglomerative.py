
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

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

    results = []
    assignments = pd.DataFrame({"word": words})

    for linkage in ["ward", "complete", "average"]:
        model = AgglomerativeClustering(
            n_clusters=13,
            linkage=linkage
        )

        clusters = model.fit_predict(X)

        results.append({
            "algorithm": f"Agglomerative {linkage}",
            "silhouette": silhouette_score(X, clusters),
            "ARI_evaluation_only": adjusted_rand_score(
                words, clusters
            ),
            "NMI_evaluation_only": normalized_mutual_info_score(
                words, clusters
            )
        })

        assignments[f"{linkage}_cluster"] = clusters

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        TABLES_DIR / "agglomerative_comparison.csv",
        index=False
    )

    assignments.to_csv(
        DATA_DIR / "agglomerative_assignments.csv",
        index=False
    )

    print("\nAgglomerative clustering comparison:")
    print(results_df.round(4).to_string(index=False))
    print("\nSaved: outputs/tables/agglomerative_comparison.csv")


if __name__ == "__main__":
    main()
