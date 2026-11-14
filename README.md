# Acoustic Feature Clustering for Hands-Free Accessibility Control Applications

An end-to-end machine learning system and interactive web application designed to analyze acoustic similarities among hands-free voice commands. Built to help assistive technology designers select voice commands that are easy for both humans and speech-recognition engines to distinguish, minimizing dangerous mixups (such as executing *"go"* instead of *"stop"*).

---

## 📌 Project Overview

Acoustic Feature Clustering for Hands-Free Accessibility Control Applications is an end-to-end machine learning project designed to analyze acoustic similarities among voice commands and improve the safety and reliability of hands-free accessibility systems.

The project uses audio recordings from the Google Speech Commands Dataset to extract 45 acoustic features and perform unsupervised clustering using K-Means and K-Medians algorithms. Principal Component Analysis (PCA) is used for dimensionality reduction and visualization, while clustering evaluation metrics help assess the quality and stability of the generated clusters.

An interactive web application is also developed to enable audio auditioning, acoustic comparison, PCA exploration, and live microphone-based voice matching.

The primary objective is to identify potentially confusing voice commands and support the design of safer, more distinguishable command vocabularies for assistive technologies.

---

## 📂 Project Directory Structure

```text
Int396/
├── Dataset/
│   ├── yes/, no/, up/, down/, left/
│   ├── right/, on/, off/, stop/, go/
├── backend/
│   ├── app.py
│   ├── feature_extractor.py
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── analysis.py
│   ├── try_it.py
│   └── report_generator.py
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── app.js
│       ├── charts.js
│       ├── audio_recorder.js
│       └── plotly.min.js
├── data/
│   ├── features.csv
│   ├── features_clean.csv
│   ├── acoustic_clustering_report.html
│   └── acoustic_clustering_report.pdf
├── notebooks/
│   └── acoustic_feature_clustering.ipynb
├── docs/
│   └── final_written_summary.md
├── run_pipeline.py
├── run_server.py
├── build_notebook.py
├── requirements.txt
└── README.md
```

---

## 💻 Prerequisites

Ensure the following software and hardware requirements are available before running the project:

| Requirement | Version / Description |
|---|---|
| Python | 3.10 or higher |
| pip | Python package installer |
| Git | Required to clone the repository |
| Web Browser | Google Chrome, Microsoft Edge, or any modern browser |
| Microphone | Required for the live voice-matching feature |
| Dataset | Google Speech Commands Dataset containing the 10 selected command folders |

**Note:** A microphone is required only for the live voice-matching functionality. The remaining application features can be explored without microphone access.

---

## 📦 Dependencies

The project uses the following Python libraries and technologies:

| Dependency | Purpose |
|---|---|
| NumPy | Numerical computations |
| Pandas | Dataset handling and manipulation |
| Scikit-learn | Machine learning, clustering, PCA, and evaluation metrics |
| Librosa | Acoustic feature extraction |
| SoundFile | Audio file processing |
| SciPy | Scientific and statistical computations |
| FastAPI | Backend application development |
| Uvicorn | Running the FastAPI server |
| Matplotlib | Data visualization |
| ReportLab | PDF report generation |
| Jinja2 | HTML report templating |
| Plotly.js | Interactive frontend visualizations |

All Python dependencies are listed in the `requirements.txt` file.

---

## ⚙️ Installation

Follow these steps to set up the project locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/AnanyaNairr/Speech_Clustering.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Speech_Clustering
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 5: Install the Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Verify the Dataset

Ensure that the `Dataset/` directory contains the following command folders:

```text
Dataset/
├── yes/
├── no/
├── up/
├── down/
├── left/
├── right/
├── on/
├── off/
├── stop/
└── go/
```

Each folder should contain the corresponding WAV recordings.

---

## 🚀 Execution Instructions

The project supports three execution methods.

### 1. Run the Interactive Web Application

Start the FastAPI web server:

```bash
python run_server.py
```

Once the server starts successfully, open the following address in your browser:

**http://127.0.0.1:8000**

The application provides a guided five-step workflow:

1. Data Loading
2. Preprocessing
3. Clustering Validation
4. Acoustic Exploration
5. Try It Yourself

### 2. Run the Complete Machine Learning Pipeline (CLI)

To execute the standalone machine learning pipeline without launching the web application:

```bash
python run_pipeline.py
```

The pipeline performs:

- Audio dataset loading.
- Acoustic feature extraction.
- Feature preprocessing and standardization.
- K-Means and K-Medians clustering.
- Clustering evaluation.
- PCA-based dimensionality reduction.
- Cluster stability analysis.
- Confusable command pair identification.
- Automated HTML and PDF report generation.

**Available Command-Line Arguments:**

| Argument | Description | Default |
|---|---|---|
| `--samples 200` | Number of balanced audio clips per command | 200 |
| `--seed 42` | Random seed for reproducibility | 42 |
| `--force` | Re-extract acoustic features from scratch | Disabled |

**Examples:**

Run with 200 samples per command:

```bash
python run_pipeline.py --samples 200
```

Run with a different random seed:

```bash
python run_pipeline.py --seed 50
```

Force complete feature re-extraction:

```bash
python run_pipeline.py --force
```

### 3. Run the Jupyter Notebook

Launch the notebook:

```bash
jupyter notebook notebooks/acoustic_feature_clustering.ipynb
```

The notebook provides a step-by-step exploration of:

- Dataset exploration.
- Acoustic feature extraction.
- Data preprocessing.
- Clustering implementation.
- Model evaluation.
- PCA visualizations.
- Interpretation of experimental findings.

It also contains documented explanations, mathematical equations, charts, and pre-rendered outputs for reproducibility.

---

## 📊 Summary of Empirical Results

### Clustering Validation Comparison (`N=2,000` clips)

| **K** | **Distance / Algorithm** | **WCSS / Inertia** | **Silhouette Score** | **Davies-Bouldin Index** |
|---|---|---:|---:|---:|
| **K = 2** | **Euclidean (K-Means) [Recommended]** | **82,500.8** | **0.0768** | **3.1309** |
| K = 2 | Manhattan (K-Medians) | 65,643.6 | 0.0841 | 2.9878 |
| K = 3 | Euclidean (K-Means) | 78,646.2 | 0.0653 | 3.3100 |
| K = 3 | Manhattan (K-Medians) | 63,770.3 | 0.0746 | 4.2212 |
| K = 4 | Euclidean (K-Means) | 76,391.5 | 0.0477 | 3.3128 |
| K = 5 | Euclidean (K-Means) | 74,710.3 | 0.0421 | 3.3505 |
| K = 8 | Euclidean (K-Means) | 70,972.5 | 0.0345 | 3.2078 |
| K = 10 | Euclidean (K-Means) | 69,218.5 | 0.0383 | 3.2389 |

### Key Acoustic Findings

1. **`K=2` Partitioning:** Speech commands separate primarily into **continuous resonant vowels** (`on`, `no`, `down`, `go`) vs **high-frequency sibilants/plosives** (`yes`, `right`, `stop`, `left`).

2. **Safety-Critical Pairings:**
   - **`GO` vs `NO` (Confusability = 0.929):** High rhyme hazard (`/oʊ/`). Assistive systems should replace *"go"* with *"forward"* or *"drive"*.
   - **`STOP` vs `GO` (Distance = 2.45):** *"stop"* (Cluster 1) and *"go"* (Cluster 0) fall into opposite acoustic groups, providing strong phonetic separation in clean speech.

3. **Stability:** 10 random initializations (seeds 42–51) produced an **Adjusted Rand Index (ARI) of 0.9720**, confirming high algorithmic stability.
