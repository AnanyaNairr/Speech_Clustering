
import pandas as pd
from config import DATA_DIR, TABLES_DIR

def main():
    assignments = pd.read_csv(
        DATA_DIR / "kmeans_gmm_assignments.csv"
    )

    print("\nSamples per K-Means cluster:")
    print(
        assignments["kmeans_cluster"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    composition = pd.crosstab(
        assignments["kmeans_cluster"],
        assignments["word"]
    )

    composition.to_csv(
        TABLES_DIR / "final_cluster_composition.csv"
    )

    print("\nWord composition of K-Means clusters:")
    print(composition.to_string())

    print("\nSaved: outputs/tables/final_cluster_composition.csv")

if __name__ == "__main__":
    main()
