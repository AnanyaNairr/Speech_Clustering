"""
Preprocessing and Variable Relevance Review for Acoustic Features.
Handles missing/invalid values, drops constant features, removes collinear features (r > 0.95),
documents label isolation, and applies StandardScaler.
"""
import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

logger = logging.getLogger(__name__)

SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "scaler.joblib"))
LABEL_ENCODER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "label_encoder.joblib"))
PREPROCESSING_LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "preprocessing_log.json"))


def preprocess_features(
    df_raw: pd.DataFrame,
    variance_threshold: float = 1e-4,
    correlation_threshold: float = 0.95,
) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler, LabelEncoder, Dict[str, Any]]:
    """
    Executes the complete preprocessing pipeline:
      1. Missing & invalid value detection and handling (median imputation / log counts)
      2. Categorical label encoding (documented strictly for post-clustering interpretation)
      3. Variance thresholding (drop constant / near-constant features)
      4. Multicollinearity filtering (drop one feature from pairs with |r| > 0.95)
      5. Standardisation with StandardScaler
      6. Comprehensive decision logging

    Returns:
      - df_clean: DataFrame containing original metadata, word, word_encoded, and remaining features
      - X_scaled: 2D numpy array of standardised features for clustering
      - scaler: fitted StandardScaler
      - label_encoder: fitted LabelEncoder for word labels
      - log: dictionary recording every decision and metric
    """
    log = {
        "initial_sample_count": len(df_raw),
        "initial_feature_count": 0,
        "missing_inf_handling": {},
        "label_encoding_note": (
            "CRITICAL: The 'word' column was encoded to integers strictly for post-hoc interpretation, "
            "evaluation, and visual ground-truth comparison. IT IS EXCLUDED FROM CLUSTERING INPUTS (X_scaled) "
            "to prevent any ground-truth leakage into the unsupervised algorithms."
        ),
        "dropped_constant_features": [],
        "dropped_collinear_features": [],
        "retained_features": [],
        "feature_correlations": {},
        "scaling_method": "StandardScaler (zero mean, unit variance)",
    }

    df = df_raw.copy()

    # Identify metadata columns vs numerical feature columns
    meta_cols = [c for c in ["word", "file_name", "pad_trim_status"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    log["initial_feature_count"] = len(feature_cols)

    # 1. Handle Missing / Inf Values
    num_nans = int(df[feature_cols].isna().sum().sum())
    num_infs = int(np.isinf(df[feature_cols].values).sum())
    log["missing_inf_handling"]["nan_count"] = num_nans
    log["missing_inf_handling"]["inf_count"] = num_infs

    if num_nans > 0 or num_infs > 0:
        logger.info(f"Replacing {num_nans} NaNs and {num_infs} Infs with column medians.")
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        col_medians = df[feature_cols].median()
        df[feature_cols] = df[feature_cols].fillna(col_medians)
        log["missing_inf_handling"]["action"] = "Imputed with column median"
    else:
        log["missing_inf_handling"]["action"] = "No missing or infinite values found. Clean dataset."

    # 2. Label Encoding for Word (Interpretation Only)
    label_encoder = LabelEncoder()
    if "word" in df.columns:
        df["word_encoded"] = label_encoder.fit_transform(df["word"])
        log["word_classes"] = list(label_encoder.classes_)
    else:
        df["word_encoded"] = 0
        log["word_classes"] = []

    # 3. Variance Filter: Drop constant or near-constant features
    variances = df[feature_cols].var()
    constant_features = variances[variances < variance_threshold].index.tolist()
    for feat in constant_features:
        log["dropped_constant_features"].append({
            "feature": feat,
            "variance": float(variances[feat]),
            "reason": f"Variance ({variances[feat]:.6e}) is below threshold ({variance_threshold}).",
        })
    active_features = [f for f in feature_cols if f not in constant_features]

    # 4. Correlation Filter: Drop one of each pair with |r| > 0.95
    corr_matrix = df[active_features].corr().abs()
    dropped_corr = set()
    collinear_log = []

    # Upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    for col in upper.columns:
        high_corr_rows = upper.index[upper[col] > correlation_threshold].tolist()
        for row in high_corr_rows:
            if row not in dropped_corr and col not in dropped_corr:
                # Drop col
                dropped_corr.add(col)
                r_val = float(upper.loc[row, col])
                collinear_log.append({
                    "dropped_feature": col,
                    "correlated_with": row,
                    "correlation": round(r_val, 4),
                    "reason": f"Correlation |r| = {r_val:.4f} exceeds {correlation_threshold}; dropped to eliminate collinearity.",
                })

    log["dropped_collinear_features"] = collinear_log
    retained_features = [f for f in active_features if f not in dropped_corr]
    log["retained_features"] = retained_features
    log["final_feature_count"] = len(retained_features)

    # Save full correlation matrix (before dropping) as dict for heatmap display
    raw_corr = df[feature_cols].corr()
    log["correlation_matrix"] = {
        "columns": feature_cols,
        "values": [[round(float(v), 3) for v in row] for row in raw_corr.values],
    }

    # 5. Standardisation with StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[retained_features].values)

    # Construct clean dataframe
    keep_cols = meta_cols + ["word_encoded"] + retained_features
    df_clean = df[keep_cols].copy()

    # Save scaler and label encoder
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    # Save retained feature names into scaler metadata
    feature_meta_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "feature_metadata.json"))
    with open(feature_meta_path, "w") as f:
        json.dump({
            "retained_features": retained_features,
            "dropped_features": [d["dropped_feature"] for d in collinear_log] + constant_features,
            "variance_threshold": variance_threshold,
            "correlation_threshold": correlation_threshold,
            "word_classes": log["word_classes"],
        }, f, indent=2)

    # Save preprocessing log
    with open(PREPROCESSING_LOG_PATH, "w") as f:
        # Don't save gigantic raw matrix to small log if too large
        json.dump(log, f, indent=2)

    # Also save clean CSV
    clean_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "features_clean.csv"))
    df_clean.to_csv(clean_csv_path, index=False)

    logger.info(
        f"Preprocessing complete: Started with {log['initial_feature_count']} features, "
        f"dropped {len(constant_features)} constant, {len(dropped_corr)} collinear. "
        f"Retained {len(retained_features)} standardized features."
    )

    return df_clean, X_scaled, scaler, label_encoder, log


def transform_single_clip(
    raw_features: Dict[str, float],
    scaler: StandardScaler,
    retained_features: List[str],
) -> np.ndarray:
    """
    Transforms a single clip's dictionary of raw features into the scaled array
    matching the exact retained feature order.
    """
    values = []
    for feat in retained_features:
        values.append(raw_features.get(feat, 0.0))
    x_arr = np.array(values, dtype=np.float32).reshape(1, -1)
    return scaler.transform(x_arr)

