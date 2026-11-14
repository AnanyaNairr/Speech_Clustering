# Acoustic Feature Clustering for Hands-Free Accessibility Control Applications

An end-to-end machine learning system and interactive web application designed to analyze acoustic similarities among hands-free voice commands. Built to help assistive technology designers select voice commands that are easy for both humans and speech-recognition engines to distinguish, minimizing dangerous mixups (such as executing *"go"* instead of *"stop"*).

> **Pure Python Project:** This project runs entirely using standard Python (`.py` modules) and does **not require Jupyter Notebook**. It is fully compatible with Visual Studio Code, PyCharm, and standard terminal environments.

---

## 🌟 Key Features

1. **Pure Python Architecture (No Jupyter Notebook Required):**
   - Run the complete data science analysis, model evaluation, and plot generation directly using `python run_analysis.py`.
   - Generates 8 high-resolution publication figures into `outputs/figures/`.
   - Automatically compiles standalone HTML and publication-ready PDF reports into `outputs/reports/`.
2. **Interactive 5-Step Web Application:**
   - **Step 1: Load Data & Acoustic Command Studio:** Listen to authentic speech recordings from the Google Speech Commands dataset for all 10 core words (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`), cycle through different volunteer speakers, adjust playback speed (0.75x slow-motion), and watch real-time frequencies on a live Web Audio API oscilloscope.
   - **Step 2: Clean & Prepare:** 45 acoustic features extracted via Librosa (MFCC means/stds, Delta MFCCs, Spectral Centroid, Bandwidth, Rolloff, ZCR, RMS energy, duration), collinearity filtering ($|r| > 0.95$), and strict label isolation.
   - **Step 3: Choose Groups (K):** Side-by-side comparison of Euclidean ($L_2$ K-Means) and Manhattan ($L_1$ K-Medians) for $K = 2 \dots 10$ across WCSS (Elbow), Silhouette Score, and Davies-Bouldin Index with automated defensible recommendations and user override.
   - **Step 4: Explore Groups:** 2D and 3D PCA interactive projections, cluster acoustic sound profiles with audio previews, word-by-cluster heatmap, and confusable command pairs hazard ranking with **sequential back-to-back audio comparison**.
   - **Step 5: Report & Try It:** In-browser microphone voice recorder with real-time reactive VU meter, WAV upload, nearest-exemplar cluster matching, and download center.
3. **Accessibility-First Design:**
   - WCAG AA compliant contrast ratio ($\ge 4.5:1$).
   - Accessible keyboard navigation, visible focus indicators, and ARIA attributes.
   - Okabe-Ito color-blind safe palette across all charts.
   - Dark Mode and Light Mode support.
4. **100% Offline & Reproducible:**
   - Pre-extracted features for 2,000 balanced clips cached in `data/features.csv` for instantaneous execution without downloading the raw 1.5 GB audio archive.
   - Fixed random seeds ($42$) throughout all clustering initializations ($n_{\text{init}} \ge 10$) and 10-seed stability testing.

---

## 📂 Project Directory Structure

```
Speech_Clustering/
├── backend/                     # Modular Python ML & FastAPI backend
│   ├── analysis.py              # PCA 2D/3D, loadings, profiles, confusable pairs, ANOVA
│   ├── app.py                   # FastAPI server & REST API endpoints
│   ├── clustering.py            # Euclidean K-Means & custom Manhattan K-Medians (L1)
│   ├── feature_extractor.py     # 45-feature extraction using librosa & soundfile
│   ├── preprocessing.py         # Relevance review, correlation filtering, StandardScaler
│   ├── report_generator.py      # Automated PDF (ReportLab) & HTML report generator
│   └── try_it.py                # Live user audio matching & PCA projection
├── frontend/                    # Single-Page Web Application
│   ├── index.html               # Semantic, accessible HTML5 user interface
│   ├── css/
│   │   └── style.css            # Dark/light theme styles, WCAG AA contrast, responsive grid
│   └── js/
│       ├── app.js               # 5-step stepper, state management, API synchronization
│       ├── audio_recorder.js    # Live microphone recording & 16kHz WAV encoder
│       ├── audio_visualizer.js  # Live Web Audio API oscilloscope & command studio
│       ├── charts.js            # Plotly.js charts (Elbow, Silhouette, DB, PCA 2D/3D)
│       └── plotly.min.js        # Vendored offline Plotly.js library
├── data/                        # Datasets, model caches, and report mirrors
│   ├── features.csv             # 2,000 clips x 45 acoustic features
│   ├── features_clean.csv       # Preprocessed standardized features
│   ├── clustering_metrics.csv   # WCSS, Silhouette & DB metrics for K=2..10
│   ├── clustering_cache.joblib  # Fast binary model cache
│   ├── acoustic_clustering_report.html # Standalone print-ready report
│   └── acoustic_clustering_report.pdf  # Formatted publication-ready PDF report
├── outputs/                     # Generated figures and reports
│   ├── figures/                 # 8 publication-grade high-resolution charts
│   └── reports/                 # Auto-generated HTML & PDF reports
├── docs/                        # Research and documentation
│   └── final_written_summary.md # 1-page submission-ready report summary
├── run_analysis.py              # Pure Python Data Science analysis runner
├── run_server.py                # Local web server launcher script
├── run_pipeline.py              # Standalone batch pipeline runner
├── start.bat                    # One-click Windows launcher
├── start.ps1                    # One-click PowerShell launcher
├── requirements.txt             # Python dependencies (no Jupyter required)
├── .gitignore                   # Excludes raw audio clips & temporary caches
└── README.md                    # Project documentation
```

---

## 🚀 Installation & Execution Guide

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your computer.
- Visual Studio Code or any standard terminal.

### 2. Setup Virtual Environment & Install Dependencies
Open your terminal or VS Code integrated terminal (`Ctrl + ~`) in the project directory:

```bash
# Optional: Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

### 3. Execution Option A: Run Full Data Science Analysis (Terminal / VS Code)

Run the standalone Python analysis script to perform the end-to-end data science pipeline:

```bash
python run_analysis.py
```

#### What this script does:
1. Loads the 2,000 speech command clips (balanced across the 10 words).
2. Performs data cleaning, variance checks, and drops collinear feature pairs with $|r| > 0.95$.
3. Evaluates $K = 2 \dots 10$ using **Euclidean K-Means** and **Manhattan K-Medians** ($n_{\text{init}} = 10$).
4. Computes WCSS (Inertia), Silhouette Score, and Davies-Bouldin Index for all 18 configurations.
5. Projects the data to 2D and 3D using PCA and computes PC1/PC2 feature loadings.
6. Analyzes phonetic confusability, calculates cluster distribution overlap, and ranks safety hazards.
7. Evaluates algorithmic stability across 10 random seeds ($42 \dots 51$) using Adjusted Rand Index (ARI).
8. Exports **8 publication-quality figures** to `outputs/figures/`:
   - `01_correlation_heatmap.png` — Pairwise acoustic feature correlation heatmap.
   - `02_clustering_metrics_comparison.png` — Elbow, Silhouette, and Davies-Bouldin curves.
   - `03_pca_scree_plot.png` — Explained and cumulative variance per principal component.
   - `04_pca_clusters_and_words_2d.png` — 2D scatter plots colored by Cluster vs. Spoken Word.
   - `05_pca_feature_loadings.png` — Top positive and negative acoustic contributors to PC1.
   - `06_word_cluster_heatmap.png` — Command word distribution across clusters.
   - `07_confusable_pairs_matrix.png` — Pairwise confusability hazard matrix.
   - `08_key_feature_contributions_anova.png` — Top 10 acoustic discriminators by ANOVA F-score.
9. Compiles and saves both HTML and PDF reports to `outputs/reports/`.

---

### 4. Execution Option B: Run Interactive Web Application

Launch the local web application:

```bash
python run_server.py
```
*Alternatively, double-click `start.bat` on Windows or execute `./start.ps1` in PowerShell.*

Open your web browser and navigate to:
👉 **`http://127.0.0.1:8000`**

#### Web Application Walkthrough:
- **Step 1 (Load Data):** Review vocabulary statistics, click any word card (`STOP`, `GO`, `YES`, etc.) to hear volunteer recordings from disk, cycle speakers, and test slow-motion playback.
- **Step 2 (Clean & Prepare):** Review dropped collinear features, standardization, and inspect the interactive correlation heatmap.
- **Step 3 (Choose Groups):** Compare metrics across K=2..10 side-by-side, view automated recommendations, and override settings if desired.
- **Step 4 (Explore Groups):** Rotate 3D PCA scatter plots, review plain-English sound profiles, and click **"🔊 Compare Pair"** to hear confusable commands (like `GO` and `NO`) played back-to-back.
- **Step 5 (Report & Try It):** Speak a command into your laptop microphone (or upload a WAV file) to see which acoustic cluster your voice falls into in real time. Download the CSV tables, HTML report, and PDF report.

---

## 🔬 Key Empirical Findings

| Metric | Euclidean ($L_2$) K-Means | Manhattan ($L_1$) K-Medians |
| :--- | :---: | :---: |
| **Recommended Configuration** | **$K = 2$** (Composite Score: 0.800) | **$K = 2$** (Composite Score: 0.785) |
| **Silhouette Score** | **0.0768** (Peak cohesiveness) | **0.0841** |
| **Davies-Bouldin Index** | **3.1309** (Tightest separation) | **2.9878** |
| **WCSS / Inertia** | 82,500.8 | 65,643.6 |
| **10-Seed Stability (ARI)** | **Mean: 0.9720** (Min: 0.9525, Std: 0.0200) | Highly Stable |
| **Cross-Metric ARI ($L_2$ vs $L_1$)** | **0.7616** | Substantial convergence across distance metrics |

### Acoustic Cluster Profiles ($K = 2$):
- **Cluster 0 — Resonant Vowels & Nasals (39.1% of clips):**
  - Dominant words: `ON` (12.8%), `NO` (12.1%), `DOWN` (11.5%), `GO` (10.9%).
  - Acoustic signature: Negative spectral centroid deviation ($z = -0.605$), low zero-crossing rate ($z = -0.628$), low-frequency resonance.
- **Cluster 1 — Sibilants & Plosives (60.9% of clips):**
  - Dominant words: `YES` (12.8%), `RIGHT` (11.2%), `STOP` (10.9%), `LEFT` (10.4%).
  - Acoustic signature: Positive spectral centroid deviation ($z = +0.389$), elevated zero-crossing rate ($z = +0.403$), high-frequency frication (>3500 Hz).

### Critical Accessibility Safety Hazards:
- **`GO` $\leftrightarrow$ `NO` (Confusability = 0.929 | 98.6% Cluster Overlap):**
  - Both words feature the back-rounded diphthong `/oʊ/`. In assistive contexts (e.g. motorized wheelchair navigation), confusion between "go" and "no" poses a severe safety hazard.
- **`STOP` $\leftrightarrow$ `GO` (Acoustic Distance = 2.45):**
  - `STOP` starts with a sharp unvoiced `/s/` hiss followed by an abrupt stop gap and `/t/` plosive, placing it cleanly in Cluster 1, while `GO` resides in Cluster 0. In clear speech, they are acoustically separable. However, in dysarthric speech where initial consonants may be dropped, they risk acoustic convergence.

---

## 📋 Requirements & Dependencies

The project relies strictly on standard, open-source Python data science and web libraries:

```text
fastapi>=0.110.0
uvicorn>=0.28.0
python-multipart>=0.0.9
numpy>=1.26.0
pandas>=2.2.0
scipy>=1.12.0
scikit-learn>=1.4.0
librosa>=0.10.1
soundfile>=0.12.1
matplotlib>=3.8.0
seaborn>=0.13.0
reportlab>=4.1.0
joblib>=1.3.0
```
*(No Jupyter, IPython, or notebook extensions required.)*

---

## ⚖️ Responsible Use & Limitations

1. **Exploratory Clusters:** Clusters discovered here reflect unsupervised statistical groupings from one volunteer sample and must not be treated as permanent real-world phonetic categories.
2. **Demographic Bias:** The Google Speech Commands dataset predominantly features clear speech from healthy adult volunteers. Speakers with motor disabilities, dysarthria, tremors, or respiratory limitations may exhibit different acoustic patterns.
3. **Mandatory Human-in-the-Loop Validation:** Findings from this tool should guide initial vocabulary selection, but any hands-free accessibility system must undergo empirical field validation with actual intended users.

