
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

    excluded = {"word", "file_path", "filename", "label"}

    numeric_columns = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in excluded
    ]

    numeric_data = df[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    valid = numeric_data.notna().all(axis=1) & df["word"].notna()

    numeric_data = numeric_data.loc[valid]
    words = df.loc[valid, "word"].astype(str).to_numpy()

    mfcc_columns = [
        c for c in numeric_columns
        if c.startswith("mfcc_") or c.startswith("delta_mfcc_")
    ]

    spectral_columns = [
        c for c in numeric_columns
        if c.startswith("spectral_")
        or "zero_crossing_rate" in c
        or c.startswith("rms")
    ]

    feature_sets = {
        "All features": numeric_columns,
        "MFCC and delta MFCC": mfcc_columns,
        "MFCC plus spectral": sorted(
            set(mfcc_columns + spectral_columns)
        )
    }

    results = []
    assignments = pd.DataFrame({"word": words})

    for name, columns in feature_sets.items():
        if not columns:
            print(f"Skipping {name}: no matching columns found.")
            continue

        X = numeric_data[columns].to_numpy()

        # Standardize each representation independently.
        X_scaled = StandardScaler().fit_transform(X)

        # Retain 95% of variance for each feature set.
        pca = PCA(
            n_components=0.95,
            random_state=RANDOM_SEED
        )
        X_pca = pca.fit_transform(X_scaled)

        model = KMeans(
            n_clusters=13,
            random_state=RANDOM_SEED,
            n_init=20
        )
        clusters = model.fit_predict(X_pca)

        results.append({
            "feature_set": name,
            "input_features": len(columns),
            "pca_dimensions": X_pca.shape[1],
            "variance_retained": pca.explained_variance_ratio_.sum(),
            "silhouette": silhouette_score(X_pca, clusters),
            "ARI_evaluation_only": adjusted_rand_score(
                words, clusters
            ),
            "NMI_evaluation_only": normalized_mutual_info_score(
                words, clusters
            )
        })

        assignments[f"{name}_cluster"] = clusters

        print(
            f"Completed {name}: "
            f"{len(columns)} input features, "
            f"{X_pca.shape[1]} PCA components"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        TABLES_DIR / "feature_set_comparison.csv",
        index=False
    )

    assignments.to_csv(
        DATA_DIR / "feature_set_assignments.csv",
        index=False
    )

    print("\nFeature-set comparison:")
    print(results_df.round(4).to_string(index=False))

    print("\nSaved table: outputs/tables/feature_set_comparison.csv")
    print("Saved assignments: data/feature_set_assignments.csv")


if __name__ == "__main__":
    main()
