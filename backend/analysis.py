"""
Acoustic Interpretation, PCA Projection, Profiling, Confusable Pairs, and ANOVA.
Transforms standardized features into actionable accessibility insights.
"""
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif

logger = logging.getLogger(__name__)


def compute_pca_analysis(
    X_scaled: np.ndarray,
    retained_features: List[str],
    n_components: int = 3,
) -> Tuple[PCA, Dict[str, Any]]:
    """
    Computes PCA projection (2D & 3D), explained variance, scree plot,
    and feature loadings for PC1 and PC2.
    """
    pca = PCA(n_components=min(n_components, X_scaled.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    exp_var = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    cum_var = [round(float(v), 4) for v in np.cumsum(pca.explained_variance_ratio_)]

    # Feature loadings: components_ [n_components, n_features]
    loadings = []
    for f_idx, feat_name in enumerate(retained_features):
        loadings.append({
            "feature": feat_name,
            "pc1_loading": round(float(pca.components_[0, f_idx]), 4),
            "pc2_loading": round(float(pca.components_[1, f_idx]), 4) if len(pca.components_) > 1 else 0.0,
        })

    # Sort top contributors for PC1 and PC2
    top_pc1 = sorted(loadings, key=lambda x: abs(x["pc1_loading"]), reverse=True)[:5]
    top_pc2 = sorted(loadings, key=lambda x: abs(x["pc2_loading"]), reverse=True)[:5]

    pca_data = {
        "explained_variance_ratio": exp_var,
        "cumulative_variance_ratio": cum_var,
        "pc_coords_2d": [[round(float(c[0]), 3), round(float(c[1]), 3)] for c in X_pca[:, :2]],
        "pc_coords_3d": [[round(float(c[0]), 3), round(float(c[1]), 3), round(float(c[2]), 3)] for c in X_pca[:, :3]] if n_components >= 3 else [],
        "loadings": loadings,
        "top_pc1_features": top_pc1,
        "top_pc2_features": top_pc2,
    }
    return pca, pca_data


def generate_sound_description(defining_features: List[Dict[str, Any]], dominant_words: List[str]) -> str:
    """
    Synthesizes a plain-English acoustic description of how a cluster sounds
    based on its most distinctive acoustic feature deviations (z-scores).
    """
    traits = []
    for f in defining_features[:3]:
        name = f["feature"]
        z = f["z_score"]
        direction = "elevated" if z > 0 else "suppressed"

        if "zero_crossing_rate" in name:
            traits.append("high-frequency noisy frication (like 's', 'sh', 'f')" if z > 0 else "smooth, continuous tone with low friction")
        elif "spectral_centroid" in name:
            traits.append("bright, sharp treble spectral energy" if z > 0 else "deep, low-pitched spectral resonance")
        elif "spectral_bandwidth" in name:
            traits.append("diffuse, wide frequency spread" if z > 0 else "narrow, concentrated harmonic pitch")
        elif "spectral_rolloff" in name:
            traits.append("extended high-frequency overtones" if z > 0 else "rapid high-frequency dropoff")
        elif "rms_mean" in name:
            traits.append("strong overall acoustic volume/power" if z > 0 else "gentle, lower vocal intensity")
        elif "active_duration" in name:
            traits.append("elongated spoken duration" if z > 0 else "abrupt, short clipped burst")
        elif "mfcc" in name:
            if "1" in name or "2" in name:
                traits.append("distinct vocal tract formant resonance" if z > 0 else "relaxed vocal tract timbre")
            else:
                traits.append(f"{direction} higher-order spectral envelope detail")

    words_str = ", ".join([f"'{w}'" for w in dominant_words[:3]])
    if not traits:
        traits = ["balanced acoustic profile"]

    return f"Characterized by {'; '.join(traits)}. Predominantly captures speech clips of {words_str}."


def compute_cluster_profiles(
    df_clean: pd.DataFrame,
    labels: np.ndarray,
    retained_features: List[str],
) -> List[Dict[str, Any]]:
    """
    Calculates detailed cluster profiles:
      - Size and percentage
      - Dominant words (percentage composition)
      - Top defining features (centroid z-scores relative to global mean)
      - Plain-English acoustic sound description
    """
    df_work = df_clean.copy()
    df_work["cluster"] = labels
    n_total = len(df_work)
    k_clusters = len(np.unique(labels))

    # Global means and stds for z-score calculation
    global_means = df_work[retained_features].mean()
    global_stds = df_work[retained_features].std().replace(0, 1.0)

    profiles = []
    for c_id in range(k_clusters):
        c_sub = df_work[df_work["cluster"] == c_id]
        c_size = len(c_sub)
        c_pct = round(100.0 * c_size / n_total, 1)

        # Word composition
        word_counts = c_sub["word"].value_counts()
        word_composition = []
        for w, cnt in word_counts.items():
            word_composition.append({
                "word": w,
                "count": int(cnt),
                "percentage": round(100.0 * cnt / c_size, 1),
            })
        dominant_words = [wc["word"] for wc in word_composition if wc["percentage"] >= 15.0]
        if not dominant_words and len(word_composition) > 0:
            dominant_words = [word_composition[0]["word"]]

        # Defining features (z-scores of cluster centroid)
        c_centroid = c_sub[retained_features].mean()
        z_scores = (c_centroid - global_means) / global_stds
        # Sort by absolute z-score
        sorted_feats = z_scores.abs().sort_values(ascending=False)
        top_defining = []
        for feat in sorted_feats.index[:5]:
            top_defining.append({
                "feature": feat,
                "z_score": round(float(z_scores[feat]), 3),
                "cluster_mean": round(float(c_centroid[feat]), 3),
                "global_mean": round(float(global_means[feat]), 3),
            })

        sound_desc = generate_sound_description(top_defining, dominant_words)

        profiles.append({
            "cluster_id": int(c_id),
            "size": int(c_size),
            "percentage": c_pct,
            "word_composition": word_composition,
            "dominant_words": dominant_words,
            "top_defining_features": top_defining,
            "plain_english_description": sound_desc,
        })

    return profiles


def compute_word_cluster_heatmap(df_clean: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
    """
    Computes a cross-tabulation heatmap between spoken words and assigned clusters.
    Returns matrix of counts and row percentages.
    """
    df_work = df_clean.copy()
    df_work["cluster"] = labels

    words = sorted(list(df_work["word"].unique()))
    k_clusters = sorted(list(np.unique(labels)))

    ct_counts = pd.crosstab(df_work["word"], df_work["cluster"]).reindex(index=words, columns=k_clusters, fill_value=0)
    ct_pcts = ct_counts.div(ct_counts.sum(axis=1), axis=0) * 100.0

    return {
        "words": words,
        "clusters": [f"Cluster {c}" for c in k_clusters],
        "count_matrix": ct_counts.values.tolist(),
        "percentage_matrix": [[round(float(v), 1) for v in row] for row in ct_pcts.values],
    }


def compute_confusable_pairs(
    df_clean: pd.DataFrame,
    labels: np.ndarray,
    X_scaled: np.ndarray,
    retained_features: List[str],
) -> List[Dict[str, Any]]:
    """
    Calculates phonetic confusability between all pairs of spoken command words.
    Confusability is measured via:
      1. Cluster distribution cosine similarity (commands that fall into the same clusters)
      2. Feature centroid Euclidean distance in standardized acoustic space
    Highlights critical accessibility pairs (e.g., 'stop' vs 'go', 'no' vs 'go').
    """
    df_work = df_clean.copy()
    df_work["cluster"] = labels
    words = sorted(list(df_work["word"].unique()))
    k_clusters = len(np.unique(labels))

    # Cluster distribution vectors
    word_dists = {}
    word_centroids = {}
    for w in words:
        w_sub = df_work[df_work["word"] == w]
        counts = w_sub["cluster"].value_counts().reindex(range(k_clusters), fill_value=0).values
        word_dists[w] = counts / (np.sum(counts) + 1e-9)

        w_idx = df_work.index[df_work["word"] == w].tolist()
        word_centroids[w] = np.mean(X_scaled[w_idx], axis=0)

    pairs = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            w1, w2 = words[i], words[j]
            # Cosine similarity of cluster distributions
            v1, v2 = word_dists[w1], word_dists[w2]
            cos_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))

            # Acoustic centroid distance
            dist = float(np.linalg.norm(word_centroids[w1] - word_centroids[w2]))

            # Confusability score: higher cosine sim & lower centroid dist => higher risk
            confusability_index = 0.6 * cos_sim + 0.4 * max(0.0, 1.0 - (dist / 8.0))

            # Accessibility risk level
            is_critical = False
            critical_warning = ""
            if {w1, w2} == {"stop", "go"}:
                is_critical = True
                critical_warning = "CRITICAL SAFETY RISK: 'stop' and 'go' are diametrically opposite control commands. Any phonetic confusion can trigger unintended device action."
            elif {w1, w2} == {"on", "off"}:
                is_critical = True
                critical_warning = "HIGH ACCESSIBILITY RISK: Opposite toggle commands ('on' vs 'off') sharing phonemes can lead to inverted user intent."
            elif {w1, w2} == {"no", "go"}:
                is_critical = True
                critical_warning = "MODERATE PHONETIC OVERLAP: Both words feature the back-rounded vowel /oʊ/, leading to frequent rhyme confusion in noisy environments."

            pairs.append({
                "word1": w1,
                "word2": w2,
                "cluster_overlap_sim": round(cos_sim, 3),
                "acoustic_centroid_distance": round(dist, 3),
                "confusability_score": round(confusability_index, 3),
                "is_critical_risk": is_critical,
                "critical_warning": critical_warning,
            })

    pairs.sort(key=lambda p: p["confusability_score"], reverse=True)
    return pairs


def compute_key_feature_contributions(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    retained_features: List[str],
) -> List[Dict[str, Any]]:
    """
    Computes ANOVA F-scores to measure which acoustic features contribute most
    to separating the clusters.
    """
    f_vals, p_vals = f_classif(X_scaled, labels)

    feature_ranking = []
    for idx, feat_name in enumerate(retained_features):
        feature_ranking.append({
            "feature": feat_name,
            "f_score": round(float(f_vals[idx]), 2) if not np.isnan(f_vals[idx]) else 0.0,
            "p_value": float(f"{p_vals[idx]:.2e}") if not np.isnan(p_vals[idx]) else 1.0,
        })

    feature_ranking.sort(key=lambda x: x["f_score"], reverse=True)
    return feature_ranking

