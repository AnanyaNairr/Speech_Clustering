
import streamlit as st
import pandas as pd
from pathlib import Path

from config import DATA_DIR, OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR

st.set_page_config(
    page_title="Acoustic Feature Clustering",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Acoustic Feature Clustering")
st.caption(
    "INT396 | Hands-Free Accessibility Control Applications"
)

st.markdown(
    "Explore acoustic features, clustering algorithms, "
    "evaluation metrics, and cluster assignments."
)

# Load existing datasets
features_path = DATA_DIR / "features.csv"
assignments_path = DATA_DIR / "cluster_assignments.csv"
metrics_path = TABLES_DIR / "clustering_metrics.csv"

if not features_path.exists():
    st.error("features.csv not found. Run the feature extraction pipeline first.")
    st.stop()

features = pd.read_csv(features_path)

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Audio Samples", len(features))
col2.metric("Unique Words", features["word"].nunique())
col3.metric(
    "Acoustic Features",
    len([
        c for c in features.select_dtypes(include="number").columns
        if c not in ["label"]
    ])
)
col4.metric("Samples per Word", int(features["word"].value_counts().min()))

st.divider()

overview_tab, metrics_tab, clusters_tab, files_tab = st.tabs([
    "Overview",
    "Model Evaluation",
    "Cluster Analysis",
    "Downloads"
])

with overview_tab:
    st.subheader("Dataset Distribution")

    word_counts = (
        features["word"]
        .value_counts()
        .sort_index()
        .rename_axis("Word")
        .reset_index(name="Samples")
    )

    st.bar_chart(word_counts, x="Word", y="Samples")
    st.dataframe(word_counts, use_container_width=True)

    st.subheader("Project Pipeline")
    st.markdown(
        """
        1. Download and balance the speech dataset
        2. Extract acoustic features
        3. Standardize features
        4. Apply PCA
        5. Perform unsupervised clustering
        6. Evaluate cluster quality
        """
    )

with metrics_tab:
    st.subheader("Clustering Evaluation")

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics, use_container_width=True)

        numeric_cols = metrics.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            st.download_button(
                "Download Clustering Metrics",
                metrics.to_csv(index=False).encode("utf-8"),
                file_name="clustering_metrics.csv",
                mime="text/csv"
            )
    else:
        st.warning("Clustering metrics file not found.")

    
st.caption(
    "Research note: Clustering performance is exploratory. "
    "Cluster assignments represent acoustic similarity and "
    "should not be interpreted as validated speech recognition."
)



with clusters_tab:
    st.subheader("Cluster Analysis")

    if assignments_path.exists():
        assignments = pd.read_csv(assignments_path)

        st.dataframe(assignments.head(100), use_container_width=True)

        st.download_button(
            "Download Cluster Assignments",
            assignments.to_csv(index=False).encode("utf-8"),
            file_name="cluster_assignments.csv",
            mime="text/csv"
        )

        st.divider()
        st.subheader("Word Distribution Across Clusters")

        if "word" in assignments.columns:
            cluster_columns = [
                c for c in assignments.columns
                if c.lower() not in [
                    "word", "file_path", "filename", "label"
        ]
        and (
            "cluster" in c.lower()
            or pd.api.types.is_numeric_dtype(assignments[c])
        )
    ]


            if cluster_columns:
                selected_column = st.selectbox(
                    "Select cluster assignment",
                    cluster_columns
                )

                composition = pd.crosstab(
                    assignments[selected_column],
                    assignments["word"]
                )

                st.bar_chart(composition)
                st.dataframe(composition, use_container_width=True)

                st.download_button(
                    "Download Cluster Composition",
                    composition.to_csv().encode("utf-8"),
                    file_name="cluster_composition.csv",
                    mime="text/csv"
                )
            else:
                st.info("No cluster-label column was found.")
        else:
            st.info("Word labels are unavailable in this assignment file.")
    else:
        st.warning(
            "Cluster assignments not found. Run 04_cluster_analysis.py first."
        )


with files_tab:
    st.subheader("Saved Visualizations")

    figures = [
        "pca_explained_variance.png",
        "cluster_visualization.png",
        "kmeans_silhouette_scores.png",
        "cluster_count_diagnostics.png",
        "kmeans13_word_distribution.png",
        "feature_diagnostics.png"
    ]

    for filename in figures:
        image_path = FIGURES_DIR / filename

        if image_path.exists():
            st.markdown(f"**{filename}**")
            st.image(str(image_path), use_container_width=True)

    st.subheader("Download Feature Data")

    st.download_button(
        "Download Acoustic Features CSV",
        features.to_csv(index=False).encode("utf-8"),
        file_name="acoustic_features.csv",
        mime="text/csv"
    )

st.sidebar.header("About")
st.sidebar.write("INT396 Acoustic Feature Clustering")
st.sidebar.write("Dataset: 650 speech clips across 13 words.")
st.sidebar.write("Baseline: K-Means with 13 clusters.")
st.sidebar.write("PCA representation: 95% retained variance.")
