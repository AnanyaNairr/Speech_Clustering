
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

from config import DATA_DIR, TABLES_DIR, RANDOM_SEED


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_DIR / "features.csv")
    words = df["word"].astype(str).to_numpy()

    excluded = {"word", "file_path", "filename", "label"}
    feature_columns = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in excluded
    ]

    X = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    valid = X.notna().all(axis=1) & df["word"].notna()
    X = X.loc[valid].to_numpy()
    y = words[valid.to_numpy()]

    X_scaled = StandardScaler().fit_transform(X)

    representations = {
        "All standardized features": X_scaled
    }

    for variance in [0.80, 0.90, 0.95, 0.99]:
        pca = PCA(
            n_components=variance,
            random_state=RANDOM_SEED
        )
        X_pca = pca.fit_transform(X_scaled)

        representations[
            f"PCA {int(variance * 100)}%"
        ] = X_pca

    results = []

    for name, X_rep in representations.items():
        model = KMeans(
            n_clusters=13,
            random_state=RANDOM_SEED,
            n_init=20
        )
        clusters = model.fit_predict(X_rep)

        results.append({
            "representation": name,
            "dimensions": X_rep.shape[1],
            "silhouette": silhouette_score(
                X_rep, clusters
            ),
            "ARI_evaluation_only": adjusted_rand_score(
                y, clusters
            ),
            "NMI_evaluation_only": normalized_mutual_info_score(
                y, clusters
            ),
            "inertia": model.inertia_
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        TABLES_DIR / "representation_comparison.csv",
        index=False
    )

    print("\nRepresentation comparison:")
    print(
        results_df.round(4).to_string(index=False)
    )

    print(
        "\nSaved:",
        TABLES_DIR / "representation_comparison.csv"
    )


if __name__ == "__main__":
    main()
