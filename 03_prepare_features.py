
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from config import DATA_DIR, OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR, RANDOM_SEED

FEATURES_PATH = DATA_DIR / "features.csv"
SCALED_PATH = DATA_DIR / "scaled_features.csv"
PCA_PATH = DATA_DIR / "pca_features.csv"
VARIANCE_PATH = TABLES_DIR / "pca_explained_variance.csv"
SCALER_PATH = OUTPUTS_DIR / "scaler_parameters.csv"


def main():
    for folder in [DATA_DIR, OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH)

    # Keep labels and file metadata separate from model features.
    metadata_columns = [
        col for col in ["path", "word", "label", "file", "filename"]
        if col in df.columns
    ]

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_columns:
        raise ValueError("No numerical feature columns were found.")

    X = df[numeric_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        raise ValueError("Missing or infinite values found. Validate features first.")

    # Standardize each feature to mean 0 and standard deviation 1.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaled_df = pd.DataFrame(
        X_scaled,
        columns=numeric_columns,
    )

    # Preserve word labels and metadata for analysis, not as model inputs.
    for col in metadata_columns:
        scaled_df.insert(len(scaled_df.columns), col, df[col].values)

    scaled_df.to_csv(SCALED_PATH, index=False)

    # PCA: retain enough components to explain 95% of the variance.
    pca = PCA(n_components=0.95, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)

    pca_columns = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
    pca_df = pd.DataFrame(X_pca, columns=pca_columns)

    for col in metadata_columns:
        pca_df[col] = df[col].values

    pca_df.to_csv(PCA_PATH, index=False)

    variance_df = pd.DataFrame({
        "component": pca_columns,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(
            pca.explained_variance_ratio_
        ),
    })
    variance_df.to_csv(VARIANCE_PATH, index=False)

    # Save scaler parameters so the same transformation can be reused later.
    pd.DataFrame({
        "feature": numeric_columns,
        "mean": scaler.mean_,
        "scale": scaler.scale_,
    }).to_csv(SCALER_PATH, index=False)

    # Scree plot.
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, len(pca_columns) + 1),
        np.cumsum(pca.explained_variance_ratio_),
        marker="o",
    )
    plt.axhline(0.95, linestyle="--", label="95% variance")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA Explained Variance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_explained_variance.png", dpi=200)
    plt.close()

    print("Feature preparation complete.")
    print("Original feature matrix:", X.shape)
    print("Scaled feature matrix:", X_scaled.shape)
    print("PCA feature matrix:", X_pca.shape)
    print(
        "Variance retained:",
        f"{np.sum(pca.explained_variance_ratio_):.2%}",
    )
    print("Scaled data:", SCALED_PATH)
    print("PCA data:", PCA_PATH)
    print("PCA variance table:", VARIANCE_PATH)
    print("PCA plot:", FIGURES_DIR / "pca_explained_variance.png")


if __name__ == "__main__":
    main()
