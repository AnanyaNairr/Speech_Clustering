
import pandas as pd
from config import DATA_DIR

def main():
    features = pd.read_csv(DATA_DIR / "features.csv")
    assignments = pd.read_csv(
        DATA_DIR / "kmeans_gmm_assignments.csv"
    )

    print("Labels in feature dataset:")
    print(sorted(features["word"].astype(str).unique()))

    print("\nLabels in cluster assignments:")
    print(sorted(assignments["word"].astype(str).unique()))

    print("\nFeature dataset label counts:")
    print(features["word"].value_counts().sort_index())

    print("\nAssignment label counts:")
    print(assignments["word"].value_counts().sort_index())

    print("\nNumber of feature rows:", len(features))
    print("Number of assignment rows:", len(assignments))

if __name__ == "__main__":
    main()
