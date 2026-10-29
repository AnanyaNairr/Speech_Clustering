
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

from config import DATA_DIR, TABLES_DIR, FIGURES_DIR, RANDOM_SEED


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_DIR / "pca_features.csv")

    pc_columns = [
        col for col in df.columns
        if col.startswith("PC") and col[2:].isdigit()
    ]

    X = df[pc_columns].to_numpy()
    words = df["word"].astype(str).to_numpy()

    results = []

    print("Testing different K-Means cluster counts...")

    for k in range(2, 21):
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            n_init=20
        )

        clusters = model.fit_predict(X)

        silhouette = silhouette_score(X, clusters)
        ari = adjusted_rand_score(words, clusters)
        nmi = normalized_mutual_info_score(words, clusters)

        results.append({
            "k": k,
            "silhouette": silhouette,
            "ARI_evaluation_only": ari,
            "NMI_evaluation_only": nmi,
            "inertia": model.inertia_
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        TABLES_DIR / "cluster_count_diagnostics.csv",
        index=False
    )

    print("\nTop 5 cluster counts by silhouette score:")
    print(
        results_df.sort_values(
            "silhouette", ascending=False
        ).head(5).round(4).to_string(index=False)
    )

    print("\nTop 5 by ARI (evaluation only):")
    print(
        results_df.sort_values(
            "ARI_evaluation_only", ascending=False
        ).head(5).round(4).to_string(index=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        results_df["k"],
        results_df["silhouette"],
        marker="o"
    )
    axes[0].set_title("Silhouette Score vs Cluster Count")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Silhouette score")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        results_df["k"],
        results_df["inertia"],
        marker="o"
    )
    axes[1].set_title("K-Means Elbow Curve")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Inertia")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "cluster_count_diagnostics.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Fit using acoustic features only.
    model = KMeans(
        n_clusters=13,
        random_state=RANDOM_SEED,
        n_init=20
    )
    clusters = model.fit_predict(X)

    # Word labels are used only to evaluate the resulting clusters.
    composition = pd.crosstab(
        pd.Series(clusters, name="cluster"),
        pd.Series(words, name="word")
    )

    composition.to_csv(
        TABLES_DIR / "kmeans13_word_distribution.csv"
    )

    normalized = composition.div(
        composition.sum(axis=0).replace(0, np.nan),
        axis=1
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(
        normalized.to_numpy(),
        aspect="auto",
        cmap="viridis"
    )

    ax.set_xticks(range(len(normalized.columns)))
    ax.set_xticklabels(
        normalized.columns, rotation=45, ha="right"
    )
    ax.set_yticks(range(len(normalized.index)))
    ax.set_yticklabels(normalized.index)
    ax.set_xlabel("Actual word (evaluation only)")
    ax.set_ylabel("K-Means cluster")
    ax.set_title("Word Distribution Across K-Means Clusters")
    fig.colorbar(image, ax=ax, label="Fraction of each word")
    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "kmeans13_word_distribution.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("\nDiagnostics complete.")
    print("Table: outputs/tables/cluster_count_diagnostics.csv")
    print("Table: outputs/tables/kmeans13_word_distribution.csv")
    print("Figures: outputs/figures/cluster_count_diagnostics.png")
    print("Figure: outputs/figures/kmeans13_word_distribution.png")


if __name__ == "__main__":
    main()
