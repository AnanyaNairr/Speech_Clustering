
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from config import DATA_DIR, TABLES_DIR, FIGURES_DIR


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_DIR / "features.csv")

    excluded = {"word", "file_path", "filename", "label"}
    feature_columns = [
        col for col in df.select_dtypes(include=np.number).columns
        if col not in excluded
    ]

    X = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    valid = X.notna().all(axis=1) & df["word"].notna()

    X = X.loc[valid]
    y = df.loc[valid, "word"].astype(str)

    # Diagnostic only: labels are used to inspect feature usefulness,
    # not to train the unsupervised clustering model.
    scores, p_values = f_classif(X, y)

    diagnostics = pd.DataFrame({
        "feature": feature_columns,
        "f_score": scores,
        "p_value": p_values
    })

    diagnostics = diagnostics.replace(
        [np.inf, -np.inf], np.nan
    ).sort_values("f_score", ascending=False)

    diagnostics.to_csv(
        TABLES_DIR / "feature_diagnostics.csv",
        index=False
    )

    print("Number of valid samples:", len(X))
    print("Number of numerical features:", len(feature_columns))
    print("\nTop 15 features by ANOVA F-score (diagnostic only):")
    print(diagnostics.head(15).round(6).to_string(index=False))

    top = diagnostics.dropna(subset=["f_score"]).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"][::-1], top["f_score"][::-1])
    ax.set_xlabel("ANOVA F-score")
    ax.set_title("Acoustic Feature Diagnostic by Word")
    fig.tight_layout()

    output_path = FIGURES_DIR / "feature_diagnostics.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved table:", TABLES_DIR / "feature_diagnostics.csv")
    print("Saved figure:", output_path)


if __name__ == "__main__":
    main()
