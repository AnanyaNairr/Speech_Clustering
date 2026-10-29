# Acoustic Feature Clustering for Hands-Free Accessibility Control Applications

**INT396 Academic Project**

An audio analysis project that extracts acoustic features from speech commands and applies unsupervised machine learning algorithms to explore similarities between spoken words. The project includes an interactive Streamlit dashboard for visualizing clustering results, comparing models, and exploring acoustic feature distributions.

## Project Overview

Voice commands can support hands-free interaction with digital systems. This project investigates whether acoustic features extracted from short speech recordings can reveal meaningful groupings using unsupervised clustering techniques.

The project uses the Google Speech Commands dataset and compares multiple clustering algorithms and feature representations.

**Note:** This is an exploratory clustering project, not a validated speech recognition system.

## Key Features

- Audio feature extraction using Python, NumPy, SciPy, and SoundFile.
- Acoustic feature analysis using MFCC-like coefficients, spectral characteristics, and zero-crossing rate.
- Feature scaling and dimensionality reduction using Principal Component Analysis (PCA).
- Comparison of K-Means, Gaussian Mixture Models (GMM), DBSCAN, and Agglomerative Clustering.
- Evaluation using Silhouette Score, Adjusted Rand Index (ARI), and Normalized Mutual Information (NMI).
- Cluster visualization and feature diagnostics.
- Interactive Streamlit dashboard for inspecting results and downloading analysis files.

## Dataset

The project uses a balanced subset of the **Google Speech Commands v0.02** dataset.

- **Total audio samples:** 650
- **Command categories:** 13
- **Samples per category:** 50

The selected speech commands are:

`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`, `forward`, `backward`, `follow`

The original audio dataset is not required in the GitHub repository if the processed feature and result files needed by the application are included.

## Methodology

1. **Data preparation:** Select a balanced subset of speech recordings.
2. **Feature extraction:** Extract numerical acoustic descriptors from each recording.
3. **Feature preprocessing:** Scale numerical features and prepare alternative feature representations.
4. **Dimensionality reduction:** Apply PCA to reduce dimensionality while preserving important variance.
5. **Clustering:** Apply and compare multiple unsupervised clustering algorithms.
6. **Evaluation:** Calculate clustering metrics and compare cluster assignments with known word labels for external evaluation.
7. **Visualization:** Generate figures, tables, and an interactive dashboard.

The known word labels are used for evaluating clustering outcomes, not as training targets for the unsupervised clustering algorithms.

## Technology Stack

- **Language:** Python
- **Data processing:** NumPy, Pandas, SciPy, SoundFile
- **Machine learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Dashboard:** Streamlit
- **Development environment:** Visual Studio Code

## Project Structure

```text
Speech_Clustering/
├── app.py
├── config.py
├── requirements.txt
├── 01_download_data.py
├── 02_extract_features.py
├── 03_prepare_features.py
├── 04_cluster_analysis.py
├── 05_visualize_clusters.py
├── 06_cluster_diagnostics.py
├── 07_feature_diagnostics.py
├── 08_compare_representations.py
├── 
