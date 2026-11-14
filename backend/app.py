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
import joblib
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

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

os.makedirs(DATA_DIR, exist_ok=True)

# Detailed Acoustic and Phonetic Metadata for the 10 Core Commands
WORD_METADATA = {
    "yes": {
        "word": "yes",
        "display": "YES",
        "ipa": "/jɛs/",
        "category": "Palatal Glide + Front Vowel + Fricative /s/",
        "acoustic_summary": "High-frequency sibilant hiss (>3500 Hz), elevated zero-crossing rate, bright spectral centroid.",
        "accessibility_role": "Primary confirmation / selection command in assistive interfaces.",
        "safety_notes": "Acoustically distinct from 'no', making binary yes/no decisions relatively robust."
    },
    "no": {
        "word": "no",
        "display": "NO",
        "ipa": "/noʊ/",
        "category": "Nasal Consonant /n/ + Back Diphthong /oʊ/",
        "acoustic_summary": "Low spectral centroid (~1150 Hz), rich low-frequency formant resonance, low zero-crossing rate.",
        "accessibility_role": "Primary cancellation / rejection command.",
        "safety_notes": "High acoustic confusability with 'go' due to shared back-rounded /oʊ/ vowel rhyme."
    },
    "up": {
        "word": "up",
        "display": "UP",
        "ipa": "/ʌp/",
        "category": "Open-Mid Central Vowel /ʌ/ + Bilabial Plosive /p/",
        "acoustic_summary": "Transient unvoiced lip-closure release, mid-range spectral energy concentration.",
        "accessibility_role": "Vertical navigational control and increment command.",
        "safety_notes": "Can be confused with 'on' or 'off' if terminal plosive is unreleased or masked by noise."
    },
    "down": {
        "word": "down",
        "display": "DOWN",
        "ipa": "/daʊn/",
        "category": "Alveolar Stop /d/ + Diphthong /aʊ/ + Nasal /n/",
        "acoustic_summary": "Pronounced low-frequency diphthong glide ending in nasal resonance, broad spectral bandwidth.",
        "accessibility_role": "Vertical navigational control and decrement command.",
        "safety_notes": "Low centroid and nasal coda place it in resonant cluster alongside 'on' and 'no'."
    },
    "left": {
        "word": "left",
        "display": "LEFT",
        "ipa": "/lɛft/",
        "category": "Lateral Liquid /l/ + Front Vowel + Fricative-Stop Cluster /ft/",
        "acoustic_summary": "Rapid transition into high-frequency unvoiced /f/ noise followed by /t/ plosive burst.",
        "accessibility_role": "Horizontal directional control / steering.",
        "safety_notes": "Crisp terminal consonant cluster provides high acoustic distinctiveness."
    },
    "right": {
        "word": "right",
        "display": "RIGHT",
        "ipa": "/raɪt/",
        "category": "Rhotic Approximant /r/ + Diphthong /aɪ/ + Alveolar Plosive /t/",
        "accessibility_role": "Horizontal directional control / steering.",
        "acoustic_summary": "Characteristic dipped third formant (F3) rising into sharp unvoiced /t/ burst.",
        "safety_notes": "Phonetically opposes 'left'; distinct spectral trajectory ensures good discriminability."
    },
    "on": {
        "word": "on",
        "display": "ON",
        "ipa": "/ɒn/",
        "category": "Open Back Vowel /ɒ/ + Alveolar Nasal /n/",
        "acoustic_summary": "Warm, resonant nasal murmur, low spectral centroid, sustained formant bandwidth.",
        "accessibility_role": "System activation / power toggle command.",
        "safety_notes": "Opposite toggle partner to 'off'; sharing an initial open vowel can cause toggle confusion."
    },
    "off": {
        "word": "off",
        "display": "OFF",
        "ipa": "/ɒf/",
        "category": "Open Back Vowel /ɒ/ + Labiodental Fricative /f/",
        "acoustic_summary": "Vowel resonance immediately diffusing into broadband turbulent unvoiced hiss.",
        "accessibility_role": "System deactivation / power shutoff command.",
        "safety_notes": "High-frequency fricative tail differentiates 'off' from 'on', but vocal fatigue can weaken the /f/."
    },
    "stop": {
        "word": "stop",
        "display": "STOP",
        "ipa": "/stɒp/",
        "category": "Sibilant /s/ + Unvoiced Stop /t/ + Vowel /ɒ/ + Bilabial /p/",
        "acoustic_summary": "Sharp initial high-frequency /s/ hiss (high ZCR >0.18), silent stop gap, followed by plosive burst.",
        "accessibility_role": "Critical emergency halt and motion cessation command.",
        "safety_notes": "Diametrically opposed to 'go'. High phonetic discriminability is vital to avoid catastrophic crashes."
    },
    "go": {
        "word": "go",
        "display": "GO",
        "ipa": "/ɡoʊ/",
        "category": "Voiced Velar Plosive /ɡ/ + Back Diphthong /oʊ/",
        "acoustic_summary": "Low-frequency voiced onset, resonant vocal tract harmonics, low spectral centroid (<1300 Hz).",
        "accessibility_role": "Execution, start, and forward motion command.",
        "safety_notes": "Rhymes strongly with 'no' (/oʊ/). Severe hazard in voice-controlled wheelchairs or robotic arms."
    }
}

# Cache for audio file paths per word
AUDIO_FILES_CACHE: Dict[str, List[str]] = {}

def get_word_audio_files(word: str) -> List[str]:
    """Returns sorted list of audio filenames for a given core word."""
    w = word.lower().strip()
    if w in AUDIO_FILES_CACHE and AUDIO_FILES_CACHE[w]:
        return AUDIO_FILES_CACHE[w]

    word_dir = os.path.join(DATASET_DIR, w)
    if os.path.exists(word_dir):
        files = sorted([f for f in os.listdir(word_dir) if f.lower().endswith(".wav") and not f.startswith(".")])
        AUDIO_FILES_CACHE[w] = files
        return files
    return []

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


CLUSTERING_CACHE_PATH = os.path.join(DATA_DIR, "clustering_cache.joblib")

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
            logger.info("Loading precomputed clustering results from joblib cache...")
            clustering_res, models_cache = joblib.load(CLUSTERING_CACHE_PATH)
        else:
            logger.info("Running clustering comparison...")
            clustering_res, models_cache = run_clustering_comparison(X_scaled, k_min=2, k_max=10, n_init=10, random_state=42)
            try:
                joblib.dump((clustering_res, models_cache), CLUSTERING_CACHE_PATH)
            except Exception as e:
                logger.warning(f"Could not save clustering cache: {e}")

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

    # Attach words metadata for visualization
    res = dict(STATE["active_analysis"])
    res["words"] = STATE["df_clean"]["word"].tolist()
    return res


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


# --- Audio Explorer & Interactive Playback Endpoints ---

@app.get("/api/audio/words_info")
def get_words_info():
    """
    Returns rich phonetic, acoustic, and dataset metadata for all 10 core command words,
    including sample audio counts and playback URLs.
    """
    words_data = []
    for w in CORE_WORDS:
        files = get_word_audio_files(w)
        meta = WORD_METADATA.get(w, {
            "word": w,
            "display": w.upper(),
            "ipa": f"/{w}/",
            "category": "Voice Command",
            "acoustic_summary": "16kHz 1-second speech sample.",
            "accessibility_role": "Hands-free voice control command.",
            "safety_notes": ""
        })
        words_data.append({
            "word": w,
            "display": meta["display"],
            "ipa": meta["ipa"],
            "category": meta["category"],
            "acoustic_summary": meta["acoustic_summary"],
            "accessibility_role": meta["accessibility_role"],
            "safety_notes": meta.get("safety_notes", ""),
            "total_clips": len(files),
            "sample_url": f"/api/audio/sample/{w}?speaker_idx=0",
            "sample_speakers_count": min(len(files), 50),
        })
    return {"words": words_data}


@app.get("/api/audio/sample/{word}")
def get_audio_sample(
    word: str,
    speaker_idx: int = Query(0, ge=0),
    random_sample: bool = Query(False),
):
    """
    Streams a genuine 1.0-second 16kHz WAV audio recording for the requested core word.
    Supports speaker index cycling and random sampling across hundreds of volunteer voices.
    """
    w = word.lower().strip()
    if w not in CORE_WORDS:
        raise HTTPException(status_code=404, detail=f"Word '{word}' is not in the core vocabulary: {CORE_WORDS}")

    files = get_word_audio_files(w)
    if not files:
        word_dir = os.path.join(DATASET_DIR, w)
        raise HTTPException(status_code=404, detail=f"No audio files found in {word_dir}")

    if random_sample:
        import random
        chosen_file = random.choice(files)
        actual_idx = files.index(chosen_file)
    else:
        actual_idx = speaker_idx % len(files)
        chosen_file = files[actual_idx]

    file_path = os.path.join(DATASET_DIR, w, chosen_file)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Audio clip file not found: {file_path}")

    # Extract speaker hash from filename (e.g. '004ae714_nohash_0.wav' -> '004ae714')
    speaker_hash = chosen_file.split("_")[0] if "_" in chosen_file else "volunteer"

    headers = {
        "X-Word": w,
        "X-Speaker-Index": str(actual_idx),
        "X-Speaker-Hash": speaker_hash,
        "X-Filename": chosen_file,
        "X-Total-Clips": str(len(files)),
        "Cache-Control": "public, max-age=3600",
    }

    return FileResponse(
        file_path,
        media_type="audio/wav",
        headers=headers,
        filename=f"{w}_{speaker_hash}.wav"
    )


@app.get("/api/audio/pair_compare")
def compare_audio_pair(word1: str, word2: str):
    """
    Returns audio URLs and phonetic confusability analysis for two words,
    enabling sequential back-to-back playback and visual comparison.
    """
    w1, w2 = word1.lower().strip(), word2.lower().strip()
    if w1 not in CORE_WORDS or w2 not in CORE_WORDS:
        raise HTTPException(status_code=400, detail="Both words must be in core vocabulary.")

    ensure_pipeline_initialized()
    confusable_info = None
    if STATE["active_analysis"] and "confusable_pairs" in STATE["active_analysis"]:
        for p in STATE["active_analysis"]["confusable_pairs"]:
            if {p["word1"], p["word2"]} == {w1, w2}:
                confusable_info = p
                break

    meta1 = WORD_METADATA.get(w1, {})
    meta2 = WORD_METADATA.get(w2, {})

    return {
        "word1": {
            "word": w1,
            "display": w1.upper(),
            "ipa": meta1.get("ipa", f"/{w1}/"),
            "category": meta1.get("category", ""),
            "audio_url": f"/api/audio/sample/{w1}?speaker_idx=0",
        },
        "word2": {
            "word": w2,
            "display": w2.upper(),
            "ipa": meta2.get("ipa", f"/{w2}/"),
            "category": meta2.get("category", ""),
            "audio_url": f"/api/audio/sample/{w2}?speaker_idx=0",
        },
        "analysis": confusable_info,
    }


# Serve Frontend
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(
            os.path.join(FRONTEND_DIR, "index.html"),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )

