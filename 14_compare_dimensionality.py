
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

from config import DATA_DIR, TABLES_DIR, RANDOM_SEED


def evaluate(name, X, words):
    clusters = KMeans(
        n_clusters=13,
        random_state=RANDOM_SEED,
        n_init=20
    ).fit_predict(X)

    return {
        "representation": name,
        "dimensions": X.shape[1],
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

    df = pd.read_csv(DATA_DIR / "features.csv")

    excluded = {"word", "file_path", "filename", "label"}
    columns = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in excluded
    ]

    data = df[columns].replace([np.inf, -np.inf], np.nan)
    valid = data.notna().all(axis=1) & df["word"].notna()

    X = StandardScaler().fit_transform(data.loc[valid])
    words = df.loc[valid, "word"].astype(str).to_numpy()

    results = []

    pca = PCA(
        n_components=0.95,
        random_state=RANDOM_SEED
    )
    X_pca = pca.fit_transform(X)
    results.append(evaluate("PCA 95%", X_pca, words))

    # Compare alternative reductions at the same dimension.
    n_components = X_pca.shape[1]

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=RANDOM_SEED
    )
    X_svd = svd.fit_transform(X)
    results.append(evaluate("Truncated SVD", X_svd, words))

    projection = GaussianRandomProjection(
        n_components=n_components,
        random_state=RANDOM_SEED
    )
    X_random = projection.fit_transform(X)
    results.append(evaluate("Random Projection", X_random, words))

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        TABLES_DIR / "dimensionality_comparison.csv",
        index=False
    )

    print("\nDimensionality comparison:")
    print(result_df.round(4).to_string(index=False))
    print("\nSaved: outputs/tables/dimensionality_comparison.csv")


if __name__ == "__main__":
    main()
