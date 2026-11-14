"""
Interactive 'Try It Yourself' Module.
Processes a user-uploaded or browser-recorded audio clip,
standardises features, projects into PCA space, assigns cluster,
and computes acoustic similarity to known command words.
"""
import os
import json
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
import joblib

from .feature_extractor import extract_features_from_audio_bytes
from .preprocessing import transform_single_clip

logger = logging.getLogger(__name__)

SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "scaler.joblib"))
FEATURE_META_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "feature_metadata.json"))


def process_user_audio(
    audio_bytes: bytes,
    active_model_centers: np.ndarray,
    distance_metric: str,
    pca_model,
    df_clean,
    X_scaled,
) -> Dict[str, Any]:
    """
    Extracts features from recorded/uploaded audio, transforms via fitted scaler,
    assigns cluster, projects to PCA, and ranks similarity to 10 core command words.
    """
    if not os.path.exists(SCALER_PATH) or not os.path.exists(FEATURE_META_PATH):
        raise RuntimeError("Model pipeline has not been run yet. Please initialize the dataset pipeline first.")

    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_META_PATH, "r") as f:
        meta = json.load(f)
    retained_features = meta["retained_features"]

    # 1. Extract raw features
    raw_feats, status = extract_features_from_audio_bytes(audio_bytes)

    # 2. Transform with StandardScaler
    x_scaled = transform_single_clip(raw_feats, scaler, retained_features)

    # 3. Predict Cluster Assignment
    if distance_metric.lower().startswith("manh"):
        from scipy.spatial.distance import cdist
        dists = cdist(x_scaled, active_model_centers, metric="cityblock")[0]
    else:
        dists = np.linalg.norm(active_model_centers - x_scaled, axis=1)

    predicted_cluster = int(np.argmin(dists))
    distance_to_center = round(float(dists[predicted_cluster]), 3)

    # 4. Project to PCA
    pca_coord = pca_model.transform(x_scaled)[0]
    pc1, pc2 = round(float(pca_coord[0]), 3), round(float(pca_coord[1]), 3)

    # 5. Calculate Acoustic Similarity to each of the 10 command words
    from scipy.spatial.distance import cdist
    is_manh = distance_metric.lower().startswith("manh")
    metric_name = "cityblock" if is_manh else "euclidean"

    words = sorted(list(df_clean["word"].unique()))
    word_distances = []

    for w in words:
        w_idx = df_clean.index[df_clean["word"] == w].tolist()
        w_samples = X_scaled[w_idx]

        # Distance to all training clips of word w
        sample_dists = cdist(x_scaled, w_samples, metric=metric_name)[0]
        # Average distance to 3 nearest exemplar clips of word w
        top3_dist = float(np.mean(np.sort(sample_dists)[:3]))

        word_distances.append((w, top3_dist))

    # Sort words by nearest distance ascending
    word_distances.sort(key=lambda s: s[1])
    min_dist = word_distances[0][1]

    # Convert distance to calibrated similarity percentages (15% to 96%)
    scale_factor = 14.0 if is_manh else 3.5
    word_similarities = []
    for w, dist in word_distances:
        dist_diff = dist - min_dist
        sim_pct = round(float(max(15.0, 96.0 * np.exp(-dist_diff / scale_factor))), 1)
        word_similarities.append({
            "word": w,
            "acoustic_distance": round(dist, 3),
            "similarity_percentage": sim_pct,
        })

    top_match = word_similarities[0]

    return {
        "status": status,
        "predicted_cluster": predicted_cluster,
        "distance_to_cluster_center": distance_to_center,
        "pca_coords": {"pc1": pc1, "pc2": pc2},
        "top_match_word": top_match["word"],
        "top_match_similarity": top_match["similarity_percentage"],
        "word_similarities": word_similarities,
        "raw_features_summary": {
            "rms_energy": round(raw_feats.get("rms_mean", 0.0), 4),
            "spectral_centroid": round(raw_feats.get("spectral_centroid_mean", 0.0), 1),
            "zero_crossing_rate": round(raw_feats.get("zero_crossing_rate_mean", 0.0), 4),
            "active_duration": round(raw_feats.get("active_duration", 0.0), 3),
        },
        "exploratory_disclaimer": (
            "NOTICE: This demo is an exploratory visualization based on unsupervised clustering from "
            "volunteer training clips. It is designed to illustrate acoustic similarity patterns and "
            "must not be relied upon as a production speech recognizer."
        ),
    }

