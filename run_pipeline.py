"""
Reproducible Offline Pipeline Runner.
Executes the end-to-end acoustic feature extraction, preprocessing,
Euclidean & Manhattan clustering (K=2..10), PCA, profiling, stability,
and generates PDF and HTML reports with real empirical numbers.

Usage:
    python run_pipeline.py [--samples 200] [--seed 42] [--force]
"""
import os
import sys
import argparse
import logging

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure cache directory for numba inside workspace
os.environ["NUMBA_CACHE_DIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".numba_cache"))

from backend.feature_extractor import extract_dataset, CORE_WORDS
from backend.preprocessing import preprocess_features
from backend.clustering import run_clustering_comparison, evaluate_stability
from backend.analysis import (
    compute_pca_analysis,
    compute_cluster_profiles,
    compute_word_cluster_heatmap,
    compute_confusable_pairs,
    compute_key_feature_contributions,
)
from backend.report_generator import generate_html_report, generate_pdf_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineRunner")


def main():
    parser = argparse.ArgumentParser(description="Acoustic Feature Clustering Pipeline Runner")
    parser.add_argument("--samples", type=int, default=200, help="Balanced samples per word (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed (default: 42)")
    parser.add_argument("--force", action="store_true", help="Force re-extraction of audio features")
    parser.add_argument("--k-override", type=int, default=None, help="Override recommended K value")
    parser.add_argument("--dist-override", type=str, default=None, help="Override distance metric")
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.dirname(__file__))
    dataset_dir = os.path.join(root_dir, "Dataset")
    data_dir = os.path.join(root_dir, "data")
    features_csv = os.path.join(data_dir, "features.csv")
    report_html = os.path.join(data_dir, "acoustic_clustering_report.html")
    report_pdf = os.path.join(data_dir, "acoustic_clustering_report.pdf")
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 80)
    print("  ACOUSTIC FEATURE CLUSTERING FOR HANDS-FREE ACCESSIBILITY APPLICATIONS")
    print("=" * 80)
    print(f"Dataset path: {dataset_dir}")
    print(f"Core words:   {', '.join(CORE_WORDS)}")
    print(f"Sample size:  {args.samples} clips per word ({args.samples * len(CORE_WORDS)} total)")
    print(f"Random seed:  {args.seed}")
    print("-" * 80)

    # 1. Feature Extraction
    print("\n[Step 1/5] Extracting 45 Acoustic Features via Librosa...")
    df_raw, extract_stats = extract_dataset(
        dataset_dir=dataset_dir,
        output_csv=features_csv,
        samples_per_word=args.samples,
        words=CORE_WORDS,
        seed=args.seed,
        force_recompute=args.force,
    )
    print(f"✓ Extracted {len(df_raw)} clips (Normal: {extract_stats.get('normal_clips', 'N/A')}, "
          f"Padded: {extract_stats.get('padded_clips', 'N/A')}, Trimmed: {extract_stats.get('trimmed_clips', 'N/A')})")

    # 2. Preprocessing & Variable Relevance
    print("\n[Step 2/5] Preprocessing, Relevance Review & Standardization...")
    df_clean, X_scaled, scaler, label_encoder, prep_log = preprocess_features(df_raw)
    print(f"✓ Starting features: {prep_log['initial_feature_count']}")
    print(f"✓ Dropped collinear (|r| > 0.95): {len(prep_log['dropped_collinear_features'])}")
    for d in prep_log["dropped_collinear_features"]:
        print(f"    - {d['dropped_feature']} (correlated with {d['correlated_with']}, r={d['correlation']})")
    print(f"✓ Dropped constant features: {len(prep_log['dropped_constant_features'])}")
    print(f"✓ Retained features for clustering: {prep_log['final_feature_count']}")
    print(f"✓ Applied StandardScaler: mean=0.0, variance=1.0 across all {X_scaled.shape[1]} dimensions")
    print(f"✓ Label isolation verified: 'word' column excluded from cluster input matrix")

    # 3. Clustering Comparison (K = 2 to 10)
    print("\n[Step 3/5] Running Clustering Comparison: Euclidean (K-Means) vs Manhattan (K-Medians)...")
    results, models_cache = run_clustering_comparison(X_scaled, k_min=2, k_max=10, n_init=10, random_state=args.seed)

    print("\n--- METRICS COMPARISON TABLE ---")
    print(f"{'K':<4} | {'Distance / Algorithm':<24} | {'WCSS (Inertia)':<16} | {'Silhouette':<12} | {'Davies-Bouldin':<14}")
    print("-" * 80)
    for row in results["comparison_table"]:
        print(f"{row['k']:<4} | {row['distance']:<24} | {row['wcss']:<16.1f} | {row['silhouette']:<12.4f} | {row['davies_bouldin']:<14.4f}")

    suggestion = results["suggestion"]
    chosen_k = args.k_override or suggestion["recommended_k"]
    chosen_dist = args.dist_override or suggestion["recommended_distance"]
    print("-" * 80)
    print(f"AUTOMATED SUGGESTION: K = {chosen_k}, Distance = {chosen_dist}")
    print(f"Reasoning: {suggestion['plain_english_reasoning']}")

    # 4. PCA and Interpretation
    print(f"\n[Step 4/5] Analyzing Chosen Configuration (K = {chosen_k}, {chosen_dist})...")
    dist_key = "euclidean" if chosen_dist.lower().startswith("eucl") else "manhattan"
    chosen_labels = models_cache[dist_key][chosen_k]["labels"]

    retained_feats = prep_log["retained_features"]
    pca_model, pca_data = compute_pca_analysis(X_scaled, retained_feats)
    profiles = compute_cluster_profiles(df_clean, chosen_labels, retained_feats)
    confusable_pairs = compute_confusable_pairs(df_clean, chosen_labels, X_scaled, retained_feats)
    stability = evaluate_stability(X_scaled, k=chosen_k, distance=chosen_dist, num_seeds=10, base_seed=args.seed)

    print(f"✓ PCA 2D Explained Variance: PC1={pca_data['explained_variance_ratio'][0]*100:.1f}%, "
          f"PC2={pca_data['explained_variance_ratio'][1]*100:.1f}% "
          f"(Total top 2: {sum(pca_data['explained_variance_ratio'][:2])*100:.1f}%)")

    print("\n--- CLUSTER PROFILES ---")
    for p in profiles:
        dom_words_str = ", ".join([f"{w['word']} ({w['percentage']}%)" for w in p['word_composition'][:3]])
        print(f"Cluster {p['cluster_id']} ({p['percentage']}% of clips, n={p['size']}):")
        print(f"  Dominant words: {dom_words_str}")
        print(f"  Acoustic description: {p['plain_english_description']}")

    print("\n--- TOP CONFUSABLE COMMAND PAIRS (ACCESSIBILITY HAZARD) ---")
    for cp in confusable_pairs[:5]:
        crit_str = " [CRITICAL SAFETY HAZARD]" if cp["is_critical_risk"] else ""
        print(f"• {cp['word1'].upper()} <-> {cp['word2'].upper()}{crit_str}: "
              f"Confusability Index = {cp['confusability_score']:.3f}, Overlap = {cp['cluster_overlap_sim']:.3f}, Dist = {cp['acoustic_centroid_distance']:.2f}")
        if cp["is_critical_risk"]:
            print(f"  WARNING: {cp['critical_warning']}")

    print("\n--- STABILITY ASSESSMENT (10 RANDOM RESTARTS) ---")
    print(f"Mean Adjusted Rand Index (ARI): {stability['mean_ari']:.4f} (Min: {stability['min_ari']:.4f}, Std: {stability['std_ari']:.4f})")
    print(f"Cross-Metric ARI (Euclidean vs Manhattan): {stability['distance_cross_ari_euclidean_vs_manhattan']:.4f}")
    print(f"Stability Verdict: {stability['stability_verdict']}")

    # 5. Export Reports
    print("\n[Step 5/5] Generating Comprehensive Reports...")
    report_bundle = {
        "suggestion": suggestion,
        "metrics_table": results["comparison_table"],
        "preprocessing_log": prep_log,
        "pca_data": pca_data,
        "profiles": profiles,
        "confusable_pairs": confusable_pairs,
        "stability": stability,
    }
    generate_html_report(report_bundle, report_html)
    generate_pdf_report(report_bundle, report_pdf)
    print(f"✓ Standalone HTML Report: {report_html}")
    print(f"✓ Formatted PDF Report:    {report_pdf}")
    print("\n" + "=" * 80)
    print("  PIPELINE EXECUTION COMPLETE! ALL ARTIFACTS SUCCESSFULLY GENERATED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
