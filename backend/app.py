"""
FastAPI Server for Acoustic Feature Clustering Application.
Exposes REST endpoints for the guided 5-step accessibility workflow.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib

from .feature_extractor import (
    extract_dataset,
    CORE_WORDS,
)
from .preprocessing import preprocess_features
from .clustering import (
    run_clustering_comparison,
    evaluate_stability,
    KMeans,
    KMediansManhattan,
)
from .analysis import (
    compute_pca_analysis,
    compute_cluster_profiles,
    compute_word_cluster_heatmap,
    compute_confusable_pairs,
    compute_key_feature_contributions,
)
from .try_it import process_user_audio
from .report_generator import generate_html_report, generate_pdf_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
DATASET_DIR = os.path.join(ROOT_DIR, "Dataset")
FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "features_clean.csv")
PREP_LOG_FILE = os.path.join(DATA_DIR, "preprocessing_log.json")
REPORT_HTML_PATH = os.path.join(DATA_DIR, "acoustic_clustering_report.html")
REPORT_PDF_PATH = os.path.join(DATA_DIR, "acoustic_clustering_report.pdf")
CLUSTERING_CACHE_PATH = os.path.join(DATA_DIR, "clustering_cache.joblib")

os.makedirs(DATA_DIR, exist_ok=True)

WORD_METADATA = {
    "yes": {
        "ipa": "/jɛs/",
        "category": "Sibilant Fricative + Palatal Glide",
        "sound_desc": "High-pitched hiss onset with rising palatal formant",
        "accessibility_role": "Affirmation / Confirmation",
    },
    "no": {
        "ipa": "/noʊ/",
        "category": "Nasal + Resonant Diphthong",
        "sound_desc": "Warm nasal onset followed by elongated vowel resonance",
        "accessibility_role": "Rejection / Cancellation",
    },
    "up": {
        "ipa": "/ʌp/",
        "category": "Central Vowel + Bilabial Unvoiced Plosive",
        "sound_desc": "Short central vowel terminating in an abrupt lip stop",
        "accessibility_role": "Vertical Navigation (Ascend)",
    },
    "down": {
        "ipa": "/daʊn/",
        "category": "Voiced Alveolar Stop + Resonant Diphthong",
        "sound_desc": "Deep low-frequency glide with elongated resonant decay",
        "accessibility_role": "Vertical Navigation (Descend)",
    },
    "left": {
        "ipa": "/lɛft/",
        "category": "Lateral Approximant + Labiodental Fricative + Stop",
        "sound_desc": "Crisp multi-consonant cluster with abrupt burst",
        "accessibility_role": "Horizontal Navigation (Left)",
    },
    "right": {
        "ipa": "/raɪt/",
        "category": "Rhotic Glide + Diphthong + Alveolar Plosive",
        "sound_desc": "Sweeping rhotic onset resolving into a sharp terminal burst",
        "accessibility_role": "Horizontal Navigation (Right)",
    },
    "on": {
        "ipa": "/ɔːn/",
        "category": "Open Vowel + Resonant Nasal",
        "sound_desc": "Continuous vocal energy with smooth low harmonic decay",
        "accessibility_role": "System Power (Activate)",
    },
    "off": {
        "ipa": "/ɔːf/",
        "category": "Open Vowel + Unvoiced Labiodental Fricative",
        "sound_desc": "Warm vocal onset transitioning into white-noise hiss",
        "accessibility_role": "System Power (Deactivate)",
    },
    "stop": {
        "ipa": "/stɑːp/",
        "category": "Sibilant Fricative + Unvoiced Alveolar Plosive",
        "sound_desc": "Sharp sibilant 's' followed by sudden transient release",
        "accessibility_role": "Safety Halt (Critical Emergency)",
    },
    "go": {
        "ipa": "/ɡoʊ/",
        "category": "Voiced Velar Stop + Resonant Diphthong",
        "sound_desc": "Deep velar release sliding into an open resonant vowel",
        "accessibility_role": "Execution / Forward (Critical)",
    },
}

app = FastAPI(title="Acoustic Feature Clustering for Accessibility", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Cache in Memory
STATE = {
    "df_raw": None,
    "df_clean": None,
    "X_scaled": None,
    "scaler": None,
    "label_encoder": None,
    "prep_log": None,
    "retained_features": [],
    "clustering_results": None,
    "models_cache": None,
    "pca_model": None,
    "pca_data": None,
    "active_config": {"k": 5, "distance": "Euclidean"},
    "active_analysis": None,
}


def ensure_pipeline_initialized():
    """Initializes the data pipeline if cached files exist on disk."""
    if STATE["df_clean"] is not None and STATE["clustering_results"] is not None:
        return

    if os.path.exists(FEATURES_CSV):
        logger.info("Initializing pipeline from cached data...")
        df_raw = pd.read_csv(FEATURES_CSV)
        STATE["df_raw"] = df_raw
        df_clean, X_scaled, scaler, label_encoder, prep_log = preprocess_features(df_raw)
        STATE["df_clean"] = df_clean
        STATE["X_scaled"] = X_scaled
        STATE["scaler"] = scaler
        STATE["label_encoder"] = label_encoder
        STATE["prep_log"] = prep_log
        STATE["retained_features"] = prep_log["retained_features"]

        if os.path.exists(CLUSTERING_CACHE_PATH):
            logger.info("Loading precomputed clustering comparison from cache...")
            clustering_res, models_cache = joblib.load(CLUSTERING_CACHE_PATH)
        else:
            clustering_res, models_cache = run_clustering_comparison(X_scaled, k_min=2, k_max=10, n_init=10, random_state=42)
            joblib.dump((clustering_res, models_cache), CLUSTERING_CACHE_PATH)
        STATE["clustering_results"] = clustering_res
        STATE["models_cache"] = models_cache

        # Set default active configuration from automated suggestion
        rec_k = clustering_res["suggestion"]["recommended_k"]
        rec_dist = clustering_res["suggestion"]["recommended_distance"]
        STATE["active_config"] = {"k": rec_k, "distance": rec_dist}
        update_active_analysis(rec_k, rec_dist)


def update_active_analysis(k: int, distance: str):
    """Computes profiles, PCA, heatmap, confusable pairs, and stability for active configuration."""
    X_scaled = STATE["X_scaled"]
    df_clean = STATE["df_clean"]
    retained_features = STATE["retained_features"]

    dist_key = "euclidean" if distance.lower().startswith("eucl") else "manhattan"
    model_info = STATE["models_cache"][dist_key][k]
    labels = model_info["labels"]

    # PCA
    pca_model, pca_data = compute_pca_analysis(X_scaled, retained_features)
    STATE["pca_model"] = pca_model
    STATE["pca_data"] = pca_data

    # Profiles
    profiles = compute_cluster_profiles(df_clean, labels, retained_features)

    # Heatmap
    heatmap = compute_word_cluster_heatmap(df_clean, labels)

    # Confusable Pairs
    confusable = compute_confusable_pairs(df_clean, labels, X_scaled, retained_features)

    # Key Feature ANOVA
    f_contribs = compute_key_feature_contributions(X_scaled, labels, retained_features)

    # Stability (10 random seeds)
    stability = evaluate_stability(X_scaled, k=k, distance=distance, num_seeds=10, base_seed=42)

    STATE["active_config"] = {"k": k, "distance": distance}
    STATE["active_analysis"] = {
        "k": k,
        "distance": distance,
        "labels": [int(lbl) for lbl in labels],
        "profiles": profiles,
        "pca_data": pca_data,
        "heatmap": heatmap,
        "confusable_pairs": confusable,
        "feature_contributions": f_contribs,
        "stability": stability,
    }


@app.on_event("startup")
async def startup_event():
    try:
        ensure_pipeline_initialized()
    except Exception as e:
        logger.warning(f"Could not initialize pipeline at startup: {e}")


@app.get("/api/status")
def get_status():
    """Returns dataset presence, cache status, and core word list."""
    has_dataset = os.path.exists(DATASET_DIR)
    has_cache = os.path.exists(FEATURES_CSV)
    return {
        "dataset_path": DATASET_DIR,
        "has_dataset": has_dataset,
        "has_cache": has_cache,
        "core_words": CORE_WORDS,
        "pipeline_ready": STATE["active_analysis"] is not None,
    }


@app.post("/api/extract")
def run_extraction(
    samples_per_word: int = Form(200),
    force_recompute: bool = Form(False),
    seed: int = Form(42),
):
    """Extracts 45 acoustic features from dataset clips and runs preprocessing."""
    if not os.path.exists(DATASET_DIR):
        raise HTTPException(status_code=404, detail=f"Dataset directory not found at {DATASET_DIR}")

    df_raw, stats = extract_dataset(
        dataset_dir=DATASET_DIR,
        output_csv=FEATURES_CSV,
        samples_per_word=samples_per_word,
        words=CORE_WORDS,
        seed=seed,
        force_recompute=force_recompute,
    )
    STATE["df_raw"] = df_raw

    # Preprocess
    df_clean, X_scaled, scaler, label_encoder, prep_log = preprocess_features(df_raw)
    STATE["df_clean"] = df_clean
    STATE["X_scaled"] = X_scaled
    STATE["scaler"] = scaler
    STATE["label_encoder"] = label_encoder
    STATE["prep_log"] = prep_log
    STATE["retained_features"] = prep_log["retained_features"]

    # Run clustering comparison
    clustering_res, models_cache = run_clustering_comparison(X_scaled, k_min=2, k_max=10, n_init=10, random_state=seed)
    joblib.dump((clustering_res, models_cache), CLUSTERING_CACHE_PATH)
    STATE["clustering_results"] = clustering_res
    STATE["models_cache"] = models_cache

    rec_k = clustering_res["suggestion"]["recommended_k"]
    rec_dist = clustering_res["suggestion"]["recommended_distance"]
    update_active_analysis(rec_k, rec_dist)

    return {
        "status": "success",
        "stats": stats,
        "preprocessing_summary": {
            "initial_features": prep_log["initial_feature_count"],
            "retained_features": prep_log["final_feature_count"],
            "dropped_collinear": len(prep_log["dropped_collinear_features"]),
            "dropped_constant": len(prep_log["dropped_constant_features"]),
        },
        "suggestion": clustering_res["suggestion"],
    }


@app.get("/api/preprocessing")
def get_preprocessing_log():
    """Returns preprocessing decisions, variable relevance log, and correlation heatmap data."""
    ensure_pipeline_initialized()
    if STATE["prep_log"] is None:
        raise HTTPException(status_code=400, detail="Data has not been extracted or preprocessed yet.")

    return {
        "log": STATE["prep_log"],
        "sample_preview": STATE["df_clean"].head(10).to_dict(orient="records"),
    }


@app.get("/api/clustering")
def get_clustering_metrics():
    """Returns WCSS, Silhouette, and Davies-Bouldin metrics for K=2..10 across Euclidean & Manhattan."""
    ensure_pipeline_initialized()
    if STATE["clustering_results"] is None:
        raise HTTPException(status_code=400, detail="Clustering has not been computed yet.")

    return STATE["clustering_results"]


@app.post("/api/select_model")
def select_model(k: int = Form(...), distance: str = Form("Euclidean")):
    """Sets active K and distance configuration and computes full interpretation outputs."""
    ensure_pipeline_initialized()
    if STATE["clustering_results"] is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized.")

    if k < 2 or k > 10:
        raise HTTPException(status_code=400, detail="K must be between 2 and 10.")

    update_active_analysis(k, distance)
    return STATE["active_analysis"]


@app.get("/api/analysis")
def get_current_analysis():
    """Returns the current active analysis results."""
    ensure_pipeline_initialized()
    if STATE["active_analysis"] is None:
        raise HTTPException(status_code=400, detail="Analysis not ready.")

    # Attach words and file names metadata for visualization & interactive audio playback
    res = dict(STATE["active_analysis"])
    res["words"] = STATE["df_clean"]["word"].tolist()
    if "file_name" in STATE["df_clean"].columns:
        res["file_names"] = STATE["df_clean"]["file_name"].tolist()
    return res


@app.get("/api/audio/words_metadata")
def get_words_metadata():
    """Returns rich linguistic and acoustic metadata for all 10 core command words, plus sample clips."""
    ensure_pipeline_initialized()
    meta = {}
    for word in CORE_WORDS:
        entry = dict(WORD_METADATA.get(word, {}))
        entry["word"] = word
        entry["total_dataset_clips"] = 0
        entry["sample_files"] = []
        entry["sample_speakers"] = []

        # Find sampled files from df_clean or disk
        if STATE["df_clean"] is not None and "file_name" in STATE["df_clean"].columns:
            df_w = STATE["df_clean"][STATE["df_clean"]["word"] == word]
            filenames = df_w["file_name"].tolist()
            entry["sample_count"] = len(filenames)
            entry["sample_files"] = filenames[:15]
            speakers = list(dict.fromkeys([f.split("_")[0] for f in filenames if "_" in f]))
            entry["sample_speakers"] = speakers[:10]
        else:
            w_dir = os.path.join(DATASET_DIR, word)
            if os.path.isdir(w_dir):
                all_wavs = [os.path.basename(f) for f in os.listdir(w_dir) if f.endswith(".wav")]
                entry["sample_count"] = len(all_wavs)
                entry["sample_files"] = all_wavs[:15]
                entry["sample_speakers"] = [f.split("_")[0] for f in all_wavs[:10]]

        meta[word] = entry
    return {"words": meta, "core_words": CORE_WORDS}


@app.get("/api/audio/sample/{word}")
def get_sample_audio(word: str, index: int = Query(0), randomize: bool = Query(False)):
    """Serves a real 1-second WAV audio sample for the requested command word."""
    word = os.path.basename(word).lower()
    if word not in CORE_WORDS:
        raise HTTPException(status_code=404, detail=f"Word '{word}' is not one of the 10 core commands.")

    ensure_pipeline_initialized()
    file_list = []
    if STATE["df_clean"] is not None and "file_name" in STATE["df_clean"].columns:
        df_w = STATE["df_clean"][STATE["df_clean"]["word"] == word]
        file_list = df_w["file_name"].tolist()

    if not file_list:
        w_dir = os.path.join(DATASET_DIR, word)
        if os.path.isdir(w_dir):
            file_list = [f for f in os.listdir(w_dir) if f.endswith(".wav") and "_background_noise_" not in f]

    if not file_list:
        raise HTTPException(status_code=404, detail=f"No audio files found for word '{word}'.")

    if randomize:
        chosen_file = np.random.choice(file_list)
    else:
        chosen_file = file_list[index % len(file_list)]

    file_path = os.path.join(DATASET_DIR, word, chosen_file)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Audio clip '{chosen_file}' not found.")

    speaker_id = chosen_file.split("_")[0] if "_" in chosen_file else "unknown"
    return FileResponse(
        file_path,
        media_type="audio/wav",
        headers={
            "X-Word-Name": word,
            "X-File-Name": chosen_file,
            "X-Speaker-Id": speaker_id,
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.get("/api/audio/clip")
def get_clip_by_name(word: str = Query(...), file_name: str = Query(...)):
    """Streams a specific WAV audio file by word and filename."""
    word = os.path.basename(word).lower()
    file_name = os.path.basename(file_name)
    if word not in CORE_WORDS:
        raise HTTPException(status_code=400, detail="Invalid word name.")

    file_path = os.path.join(DATASET_DIR, word, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Audio file '{file_name}' not found.")

    speaker_id = file_name.split("_")[0] if "_" in file_name else "unknown"
    return FileResponse(
        file_path,
        media_type="audio/wav",
        headers={
            "X-Word-Name": word,
            "X-File-Name": file_name,
            "X-Speaker-Id": speaker_id,
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.post("/api/try_it")
async def try_it_endpoint(
    audio_file: UploadFile = File(...),
):
    """Processes live microphone recording or uploaded WAV file."""
    ensure_pipeline_initialized()
    if STATE["active_analysis"] is None:
        raise HTTPException(status_code=400, detail="Pipeline not ready. Please initialize data first.")

    audio_bytes = await audio_file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio payload received.")

    active_k = STATE["active_config"]["k"]
    active_dist = STATE["active_config"]["distance"]
    dist_key = "euclidean" if active_dist.lower().startswith("eucl") else "manhattan"
    centers = STATE["models_cache"][dist_key][active_k]["centers"]

    try:
        result = process_user_audio(
            audio_bytes=audio_bytes,
            active_model_centers=centers,
            distance_metric=active_dist,
            pca_model=STATE["pca_model"],
            df_clean=STATE["df_clean"],
            X_scaled=STATE["X_scaled"],
        )
        return result
    except Exception as e:
        logger.error(f"Error processing user audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/features_csv")
def download_features_csv():
    """Downloads raw extracted features.csv."""
    if not os.path.exists(FEATURES_CSV):
        raise HTTPException(status_code=404, detail="features.csv not generated yet.")
    return FileResponse(FEATURES_CSV, media_type="text/csv", filename="features.csv")


@app.get("/api/download/clean_csv")
def download_clean_csv():
    """Downloads preprocessed features_clean.csv."""
    if not os.path.exists(CLEAN_CSV):
        raise HTTPException(status_code=404, detail="features_clean.csv not generated yet.")
    return FileResponse(CLEAN_CSV, media_type="text/csv", filename="features_clean.csv")


@app.get("/api/download/metrics_csv")
def download_metrics_csv():
    """Downloads the clustering metrics comparison table as CSV."""
    ensure_pipeline_initialized()
    if STATE["clustering_results"] is None:
        raise HTTPException(status_code=400, detail="Clustering results not ready.")

    df_metrics = pd.DataFrame(STATE["clustering_results"]["comparison_table"])
    out_path = os.path.join(DATA_DIR, "clustering_metrics.csv")
    df_metrics.to_csv(out_path, index=False)
    return FileResponse(out_path, media_type="text/csv", filename="clustering_metrics.csv")


@app.get("/api/download/report_html")
def download_report_html():
    """Generates and downloads full HTML report."""
    ensure_pipeline_initialized()
    report_data = {
        "suggestion": STATE["clustering_results"]["suggestion"],
        "metrics_table": STATE["clustering_results"]["comparison_table"],
        "preprocessing_log": STATE["prep_log"],
        "pca_data": STATE["pca_data"],
        "profiles": STATE["active_analysis"]["profiles"],
        "confusable_pairs": STATE["active_analysis"]["confusable_pairs"],
        "stability": STATE["active_analysis"]["stability"],
    }
    generate_html_report(report_data, REPORT_HTML_PATH)
    return FileResponse(REPORT_HTML_PATH, media_type="text/html", filename="acoustic_clustering_report.html")


@app.get("/api/download/report_pdf")
def download_report_pdf():
    """Generates and downloads full PDF report."""
    ensure_pipeline_initialized()
    report_data = {
        "suggestion": STATE["clustering_results"]["suggestion"],
        "metrics_table": STATE["clustering_results"]["comparison_table"],
        "preprocessing_log": STATE["prep_log"],
        "pca_data": STATE["pca_data"],
        "profiles": STATE["active_analysis"]["profiles"],
        "confusable_pairs": STATE["active_analysis"]["confusable_pairs"],
        "stability": STATE["active_analysis"]["stability"],
    }
    generate_pdf_report(report_data, REPORT_PDF_PATH)
    return FileResponse(REPORT_PDF_PATH, media_type="application/pdf", filename="acoustic_clustering_report.pdf")


# Serve Frontend
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

