
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from config import DATA_DIR, FIGURES_DIR, RANDOM_SEED

PCA_PATH = DATA_DIR / "pca_features.csv"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PCA_PATH)
    pc_columns = [
        col for col in df.columns
        if col.startswith("PC") and col[2:].isdigit()
    ]

    X = df[pc_columns].to_numpy()
    words = df["word"].astype(str)

    # Create a two-dimensional view for visualization only.
    pca_2d = PCA(n_components=2, random_state=RANDOM_SEED)
    X_2d = pca_2d.fit_transform(X)

    # Compare the true labels with unsupervised cluster assignments.
    clusters = KMeans(
        n_clusters=13,
        random_state=RANDOM_SEED,
        n_init=20,
    ).fit_predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    word_codes = pd.Categorical(words).codes
    axes[0].scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=word_codes, cmap="tab20",
        s=24, alpha=0.75,
    )
    axes[0].set_title("PCA View: Actual Word Labels")
    axes[0].set_xlabel("Principal component 1")
    axes[0].set_ylabel("Principal component 2")

    scatter = axes[1].scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=clusters, cmap="tab20",
        s=24, alpha=0.75,
    )
    axes[1].set_title("PCA View: K-Means (13 Clusters)")
    axes[1].set_xlabel("Principal component 1")
    axes[1].set_ylabel("Principal component 2")

    fig.suptitle(
        "Speech Feature Clustering: Labels vs. K-Means",
        fontsize=14,
    )
    fig.tight_layout()

    output_path = FIGURES_DIR / "cluster_visualization.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Visualization complete.")
    print("2D variance retained:", 
          f"{pca_2d.explained_variance_ratio_.sum():.2%}")
    print("Saved figure:", output_path)


if __name__ == "__main__":
    main()
