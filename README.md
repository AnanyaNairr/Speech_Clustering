# Acoustic Feature Clustering for Hands-Free Accessibility Control Applications

An end-to-end machine learning system and interactive web application designed to analyze acoustic similarities among hands-free voice commands. Built to help assistive technology designers select voice commands that are easy for both humans and speech-recognition engines to distinguish, minimizing dangerous mixups (such as executing *"go"* instead of *"stop"*).

---

## 🌟 Key Features

1. **Interactive Audio Command Deck & Auditory Auditioning:**
   - **Step 1 Audio Deck:** Click any of the 10 core command cards (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`) to listen to authentic 1.0-second WAV recordings directly from the dataset.
   - **Speaker Shuffling (🔀 Next Voice):** Cycle through different volunteer speakers (accents, genders, speech speeds) for every command word.
   - **Master Tour Player (▶ Play All):** Sequentially plays all 10 vocabulary commands with real-time card focus and glowing highlights.
   - **Live Audio Equalizer & Waveform Spectrogram:** Animated 4-bar equalizer on active cards and a real-time reactive waveform display on the master HUD ribbon.
2. **Interactive Acoustic Analysis & Exploration:**
   - **A/B Confusable Pairs Audio Player (Step 4):** Click `🔊 Compare A vs B` on any confusable pair row (e.g. `GO` vs `NO`, `STOP` vs `GO`) to hear both clips played sequentially with visual timing indicators, exposing why rhyming formants (like `/oʊ/`) pose high confusion risks.
   - **PCA Scatter Point Audio Explorer (Step 4):** Click any data point among the 2,000 samples on the 2D PCA projection to open the *Clip Inspector*, revealing the spoken word, cluster assignment, speaker ID, exact coordinates, and a play button to hear that exact clip.
   - **Guided 5-Step Workflow:** Clean and intuitive progression from raw data loading to preprocessing, validation metrics, acoustic group exploration, and responsible-use reporting.
3. **Live Microphone Voice Matcher (Step 5):**
   - **Reactive Oscilloscope Visualizer:** HTML5 Canvas drawing real-time sine wave and spectrum glow during microphone recording.
   - **Dynamic Decibel (VU) Meter:** Visual volume indicator confirming speech energy capture.
   - **Instant Nearest-Exemplar Matching:** Projects the user's voice clip onto the PCA map, identifies the closest cluster, and ranks similarity to core commands.
4. **Accessible Design:**
   - WCAG AA compliant contrast ratio (> 4.5:1).
   - Accessible keyboard navigation, visible focus indicators, and ARIA attributes.
   - Okabe-Ito color-blind safe palette.
   - Instant Dark Mode and Light Mode toggle.
5. **100% Offline & Self-Contained:**
   - Local Plotly.js library included.
   - No external CDNs or cloud services required.
   - Fast sub-second cached pipeline startup.

---

## 📂 Project Directory Structure

```
Int396/
├── Dataset/                     # Google Speech Commands 10 core command folders
│   ├── yes/, no/, up/, down/, left/, right/, on/, off/, stop/, go/
├── backend/                     # Python FastAPI server & ML modules
│   ├── app.py                   # FastAPI application & REST endpoints
│   ├── feature_extractor.py     # 45-feature extraction using librosa & soundfile
│   ├── preprocessing.py         # Relevance review, correlation filtering, StandardScaler
│   ├── clustering.py            # Euclidean K-Means & Manhattan K-Medians, metrics & stability
│   ├── analysis.py              # PCA, loadings, cluster profiles, confusable pairs, ANOVA
│   ├── try_it.py                # Live audio processing for "Try it yourself" demo
│   └── report_generator.py      # Automated PDF (ReportLab) & HTML report generator
├── frontend/                    # Single-Page Web Application
│   ├── index.html               # Semantic, accessible HTML5 user interface
│   ├── css/
│   │   └── style.css            # Dark/light theme styles, WCAG AA contrast, responsive grid
│   └── js/
│       ├── app.js               # 5-step stepper, state management, API synchronization
│       ├── charts.js            # Plotly.js charts (Elbow, Silhouette, DB, PCA 2D/3D, Heatmaps)
│       ├── audio_recorder.js    # Live microphone recording & WAV processor
│       └── plotly.min.js        # Vendored offline Plotly.js library
├── data/                        # Output datasets, cache, and reports
│   ├── features.csv             # Extracted 45 acoustic features
│   ├── features_clean.csv       # Preprocessed features table
│   ├── acoustic_clustering_report.html # Standalone print-ready report
│   └── acoustic_clustering_report.pdf  # Formatted publication-ready PDF report
├── notebooks/                   # Jupyter Notebook submission
│   └── acoustic_feature_clustering.ipynb # Fully documented notebook with pre-rendered outputs
├── docs/                        # Research and documentation
│   └── final_written_summary.md # 1-page submission-ready report summary
├── run_pipeline.py              # Standalone CLI script running the complete ML pipeline
├── run_server.py                # Web server launcher script
├── build_notebook.py            # Script that built and pre-rendered the notebook
├── requirements.txt             # Project Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Installation
Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
Launch the web interface locally on your laptop:

```bash
python run_server.py
```
Open your browser and navigate to: **`http://127.0.0.1:8000`**

### 3. Run the Standalone Pipeline (CLI)
To run the full extraction, preprocessing, clustering comparison, PCA, stability, and report generation in batch mode:

```bash
python run_pipeline.py
```
*Options:*
- `--samples 200` : Set number of balanced clips per word (default: 200).
- `--seed 42` : Fixed random seed (default: 42).
- `--force` : Re-extract features from scratch.

### 4. Run the Jupyter Notebook
To inspect the step-by-step reproducible analysis with equations, charts, and markdown:

```bash
jupyter notebook notebooks/acoustic_feature_clustering.ipynb
```

---

## 📊 Summary of Empirical Results

### Clustering Validation Comparison ($N = 2,000$ clips)

| K | Distance / Algorithm | WCSS / Inertia | Silhouette Score | Davies-Bouldin Index |
| :---: | :---: | :---: | :---: | :---: |
| **K = 2** | **Euclidean (K-Means) [Recommended]** | **82,500.8** | **0.0768** | **3.1309** |
| K = 2 | Manhattan (K-Medians) | 65,643.6 | 0.0841 | 2.9878 |
| K = 3 | Euclidean (K-Means) | 78,646.2 | 0.0653 | 3.3100 |
| K = 3 | Manhattan (K-Medians) | 63,770.3 | 0.0746 | 4.2212 |
| K = 4 | Euclidean (K-Means) | 76,391.5 | 0.0477 | 3.3128 |
| K = 5 | Euclidean (K-Means) | 74,710.3 | 0.0421 | 3.3505 |
| K = 8 | Euclidean (K-Means) | 70,972.5 | 0.0345 | 3.2078 |
| K = 10 | Euclidean (K-Means) | 69,218.5 | 0.0383 | 3.2389 |

### Key Acoustic Findings
1. **$K = 2$ Partitioning:** Speech commands separate primarily into **continuous resonant vowels** (`on`, `no`, `down`, `go`) vs **high-frequency sibilants/plosives** (`yes`, `right`, `stop`, `left`).
2. **Safety-Critical Pairings:**
   - **`GO` vs `NO` (Confusability = 0.929):** High rhyme hazard (`/oʊ/`). Assistive systems should replace *"go"* with *"forward"* or *"drive"*.
   - **`STOP` vs `GO` (Distance = 2.45):** *"stop"* (Cluster 1) and *"go"* (Cluster 0) fall into opposite acoustic groups, providing strong phonetic separation in clean speech.
3. **Stability:** 10 random initializations (seeds 42–51) produced an **Adjusted Rand Index (ARI) of 0.9720**, confirming high algorithmic stability.

---

## ♿ Accessibility & Responsible Use

1. **Volunteer Dataset Limitations:** The Google Speech Commands dataset consists predominantly of clear speech from healthy adult volunteers. Ambient noise, regional accents, and vocal fatigue significantly alter acoustic characteristics.
2. **Under-representation of Dysarthric Speech:** People with motor disabilities (e.g., Cerebral Palsy, ALS, Stroke) frequently experience dysarthria, which alters formant frequencies, reduces breath support, and introduces vocal tremor. These baseline clusters **do not represent dysarthric speech patterns** and must be calibrated to individual end-users.
3. **Fail-Safe Design:** Voice-controlled assistive devices should never execute irreversible actions based on a single voice trigger without secondary dwell-time confirmation or multi-modal fail-safes.

