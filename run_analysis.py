"""
Standalone Data Science Analysis Runner for Acoustic Feature Clustering.
Replaces Jupyter Notebook with a clean, 100% reproducible, pure-Python script.
Runs directly in VS Code, PyCharm, or standard Python terminal.

Usage:
    python run_analysis.py
    python run_analysis.py --samples 200 --seed 42 --k-override 2 --dist-override Euclidean
"""

import os
import sys
import argparse
import logging
import time

# Headless matplotlib backend for non-GUI execution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.feature_selection import f_classif

# Ensure project root is in sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

# Configure cache directory for numba inside workspace
os.environ["NUMBA_CACHE_DIR"] = os.path.abspath(os.path.join(ROOT_DIR, ".numba_cache"))

from backend.feature_extractor import extract_dataset, CORE_WORDS
from backend.preprocessing import preprocess_features
from backend.clustering import (
    run_clustering_comparison,
    evaluate_stability,
    KMeans,
    KMediansManhattan,
)
from backend.analysis import (
    compute_pca_analysis,
    compute_cluster_profiles,
    compute_word_cluster_heatmap,
    compute_confusable_pairs,
    compute_key_feature_contributions,
)
from backend.report_generator import generate_html_report, generate_pdf_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AcousticAnalysis")

# Colorblind-safe palette (Okabe-Ito)
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#F0E442", "#56B4E9", "#CC79A7", "#E69F00", "#000000"]
sns.set_theme(style="whitegrid", font="sans-serif")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Acoustic Feature Clustering for Hands-Free Accessibility Control (Pure Python Analysis)"
    )
    parser.add_argument("--samples", type=int, default=200, help="Balanced audio clips per word (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed (default: 42)")
    parser.add_argument("--k-override", type=int, default=None, help="Override recommended K value (2 to 10)")
    parser.add_argument("--dist-override", type=str, default=None, help="Override distance metric (Euclidean or Manhattan)")
    parser.add_argument("--force", action="store_true", help="Force re-extraction of audio features from raw WAV clips")
    return parser.parse_args()


def main():
    args = parse_arguments()
    start_time = time.time()

    dataset_dir = os.path.join(ROOT_DIR, "Dataset")
    data_dir = os.path.join(ROOT_DIR, "data")
    figures_dir = os.path.join(ROOT_DIR, "outputs", "figures")
    reports_dir = os.path.join(ROOT_DIR, "outputs", "reports")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    features_csv = os.path.join(data_dir, "features.csv")
    clean_csv = os.path.join(data_dir, "features_clean.csv")
    metrics_csv = os.path.join(data_dir, "clustering_metrics.csv")

    print("\n" + "=" * 80)
    print("  ACOUSTIC FEATURE CLUSTERING FOR HANDS-FREE ACCESSIBILITY APPLICATIONS")
    print("  Pure-Python Reproducible Data Science Analysis Pipeline")
    print("=" * 80)
    print(f"Dataset path:      {dataset_dir}")
    print(f"Core vocabulary:   {', '.join(CORE_WORDS)}")
    print(f"Sampling per word: {args.samples} clips ({args.samples * len(CORE_WORDS)} total)")
    print(f"Random seed:       {args.seed}")
    print(f"Output figures:    {figures_dir}")
    print("-" * 80 + "\n")

    # -------------------------------------------------------------------------
    # 1. Feature Extraction
    # -------------------------------------------------------------------------
    print(">>> [Stage 1/8] Extracting 45 Acoustic Features via Librosa...")
    df_raw, extract_stats = extract_dataset(
        dataset_dir=dataset_dir,
        output_csv=features_csv,
        samples_per_word=args.samples,
        words=CORE_WORDS,
        seed=args.seed,
        force_recompute=args.force,
    )
    print(f"✓ Feature extraction complete: {len(df_raw)} total audio clips loaded.")
    print(f"  Alignment stats: Normal={extract_stats.get('normal_clips', 'N/A')}, "
          f"Padded={extract_stats.get('padded_clips', 'N/A')}, Trimmed={extract_stats.get('trimmed_clips', 'N/A')}")

    # -------------------------------------------------------------------------
    # 2. Preprocessing & Multicollinearity Filtering
    # -------------------------------------------------------------------------
    print("\n>>> [Stage 2/8] Preprocessing, Variable Relevance & Standardization...")
    df_clean, X_scaled, scaler, label_encoder, prep_log = preprocess_features(df_raw)
    retained_feats = prep_log["retained_features"]
    df_clean.to_csv(clean_csv, index=False)

    print(f"✓ Starting features:    {prep_log['initial_feature_count']}")
    print(f"✓ Dropped collinear:    {len(prep_log['dropped_collinear_features'])} pairs with |r| > 0.95")
    for d in prep_log["dropped_collinear_features"]:
        print(f"    - Dropped '{d['dropped_feature']}' (correlated with '{d['correlated_with']}', r={d['correlation']})")
    print(f"✓ Dropped constant:     {len(prep_log['dropped_constant_features'])}")
    print(f"✓ Retained features:    {prep_log['final_feature_count']}")
    print(f"✓ StandardScaler:       Transformed to zero-mean, unit-variance matrix (shape: {X_scaled.shape})")
    print(f"✓ Strict label isolation: Spoken word column isolated exclusively for interpretation.")

    # Save Figure 1: Correlation Heatmap
    corr_subset = ['mfcc_mean_1', 'mfcc_mean_2', 'mfcc_mean_3', 'spectral_centroid_mean',
                   'spectral_bandwidth_mean', 'spectral_rolloff_mean', 'zero_crossing_rate_mean', 'rms_mean']
    available_corr = [c for c in corr_subset if c in df_raw.columns]
    if len(available_corr) >= 4:
        plt.figure(figsize=(9, 7))
        corr_matrix = df_raw[available_corr].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5, cbar_kws={"label": "Pearson Correlation (r)"})
        plt.title("Acoustic Feature Correlation Heatmap (|r| > 0.95 Filter Check)", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        fig1_path = os.path.join(figures_dir, "01_correlation_heatmap.png")
        plt.savefig(fig1_path, dpi=200)
        plt.close()
        print(f"✓ Saved Figure 1: {fig1_path}")

    # -------------------------------------------------------------------------
    # 3. Clustering Comparison across K=2..10 (Euclidean vs Manhattan)
    # -------------------------------------------------------------------------
    print("\n>>> [Stage 3/8] Running Clustering: Euclidean (K-Means) vs Manhattan (K-Medians)...")
    results, models_cache = run_clustering_comparison(X_scaled, k_min=2, k_max=10, n_init=10, random_state=args.seed)

    df_metrics = pd.DataFrame(results["comparison_table"])
    df_metrics.to_csv(metrics_csv, index=False)

    print("\n" + "=" * 80)
    print(f"{'K':<4} | {'Distance / Metric':<24} | {'WCSS (Inertia)':<16} | {'Silhouette':<12} | {'Davies-Bouldin':<14}")
    print("-" * 80)
    for row in results["comparison_table"]:
        print(f"{row['k']:<4} | {row['distance']:<24} | {row['wcss']:<16.1f} | {row['silhouette']:<12.4f} | {row['davies_bouldin']:<14.4f}")
    print("=" * 80)

    # Save Figure 2: Metrics Comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    euc_m = df_metrics[df_metrics["distance"].str.contains("Euclidean")]
    man_m = df_metrics[df_metrics["distance"].str.contains("Manhattan")]

    # WCSS / Elbow
    axes[0].plot(euc_m["k"], euc_m["wcss"], "o-", label="Euclidean (L2)", color=OKABE_ITO[0], linewidth=2)
    axes[0].plot(man_m["k"], man_m["wcss"], "s--", label="Manhattan (L1)", color=OKABE_ITO[1], linewidth=2)
    axes[0].set_title("WCSS / Inertia (Elbow Method)", fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Within-Cluster Sum of Squares")
    axes[0].legend()

    # Silhouette Score
    axes[1].plot(euc_m["k"], euc_m["silhouette"], "o-", label="Euclidean (L2)", color=OKABE_ITO[0], linewidth=2)
    axes[1].plot(man_m["k"], man_m["silhouette"], "s--", label="Manhattan (L1)", color=OKABE_ITO[1], linewidth=2)
    axes[1].set_title("Silhouette Score (Higher is Better)", fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Average Silhouette Coefficient")
    axes[1].legend()

    # Davies-Bouldin Index
    axes[2].plot(euc_m["k"], euc_m["davies_bouldin"], "o-", label="Euclidean (L2)", color=OKABE_ITO[0], linewidth=2)
    axes[2].plot(man_m["k"], man_m["davies_bouldin"], "s--", label="Manhattan (L1)", color=OKABE_ITO[1], linewidth=2)
    axes[2].set_title("Davies-Bouldin Index (Lower is Better)", fontweight="bold")
    axes[2].set_xlabel("Number of Clusters (K)")
    axes[2].set_ylabel("Davies-Bouldin Index")
    axes[2].legend()

    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "02_clustering_metrics_comparison.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"✓ Saved Figure 2: {fig2_path}")

    # Automated Defensible Configuration Selection
    sug = results["suggestion"]
    chosen_k = args.k_override if args.k_override else sug["recommended_k"]
    chosen_dist = args.dist_override if args.dist_override else sug["recommended_distance"]

    print(f"\n★ AUTOMATED DEFENSIVE RECOMMENDATION: K = {sug['recommended_k']} ({sug['recommended_distance']} Distance)")
    print(f"  Composite Score: {sug['composite_score']:.3f} | Silhouette: {sug['silhouette']:.4f} | Davies-Bouldin: {sug['davies_bouldin']:.4f}")
    print(f"  Plain-English Rationale: {sug['plain_english_reasoning']}")
    if args.k_override or args.dist_override:
        print(f"  [User Override Applied]: Active Configuration set to K = {chosen_k}, Distance = {chosen_dist}")

    # -------------------------------------------------------------------------
    # 4. Dimensionality Reduction (PCA 2D & 3D, Scree Plot, Loadings)
    # -------------------------------------------------------------------------
    print(f"\n>>> [Stage 4/8] Computing PCA Projection & Feature Loadings...")
    dist_key = "euclidean" if chosen_dist.lower().startswith("eucl") else "manhattan"
    chosen_labels = models_cache[dist_key][chosen_k]["labels"]

    pca_model, pca_data = compute_pca_analysis(X_scaled, retained_feats)
    evr = pca_data["explained_variance_ratio"]

    print(f"✓ PCA Explained Variance: PC1 = {evr[0]*100:.1f}%, PC2 = {evr[1]*100:.1f}%, PC3 = {evr[2]*100:.1f}%")
    print(f"  Cumulative Top 2: {sum(evr[:2])*100:.1f}% | Top 3: {sum(evr[:3])*100:.1f}%")

    # Save Figure 3: Scree Plot
    plt.figure(figsize=(8, 4.5))
    components_range = range(1, len(evr) + 1)
    plt.bar(components_range, [v * 100 for v in evr], color=OKABE_ITO[0], alpha=0.8, label="Individual Variance (%)")
    plt.plot(components_range, [v * 100 for v in pca_data["cumulative_variance_ratio"]], "r-o", linewidth=2, label="Cumulative Variance (%)")
    plt.title("PCA Scree Plot (Explained Variance per Principal Component)", fontweight="bold")
    plt.xlabel("Principal Component Index")
    plt.ylabel("Variance Explained (%)")
    plt.xticks(components_range)
    plt.legend()
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "03_pca_scree_plot.png")
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"✓ Saved Figure 3: {fig3_path}")

    # Save Figure 4: PCA 2D Scatter (Clusters vs Spoken Words)
    coords_2d = np.array(pca_data["pc_coords_2d"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # By Cluster
    scatter1 = axes[0].scatter(coords_2d[:, 0], coords_2d[:, 1], c=chosen_labels, cmap="tab10", alpha=0.65, s=30, edgecolors="none")
    axes[0].set_title(f"PCA 2D: Colored by Acoustic Cluster (K = {chosen_k})", fontweight="bold")
    axes[0].set_xlabel(f"PC1 ({evr[0]*100:.1f}% Variance)")
    axes[0].set_ylabel(f"PC2 ({evr[1]*100:.1f}% Variance)")
    plt.colorbar(scatter1, ax=axes[0], label="Cluster ID")

    # By Spoken Word
    words = df_clean["word"].values
    unique_words = sorted(list(set(words)))
    for idx, w in enumerate(unique_words):
        mask = (words == w)
        color = OKABE_ITO[idx % len(OKABE_ITO)]
        axes[1].scatter(coords_2d[mask, 0], coords_2d[mask, 1], label=w.upper(), alpha=0.6, s=28, color=color, edgecolors="none")
    axes[1].set_title("PCA 2D: Colored by Spoken Command Word", fontweight="bold")
    axes[1].set_xlabel(f"PC1 ({evr[0]*100:.1f}% Variance)")
    axes[1].set_ylabel(f"PC2 ({evr[1]*100:.1f}% Variance)")
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Command")

    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "04_pca_clusters_and_words_2d.png")
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f"✓ Saved Figure 4: {fig4_path}")

    # Save Figure 5: Feature Loadings
    loadings = pca_data["loadings"]
    df_loadings = pd.DataFrame(loadings).sort_values(by="pc1_loading", ascending=False)
    plt.figure(figsize=(10, 6))
    top_loadings = pd.concat([df_loadings.head(5), df_loadings.tail(5)])
    colors = ["#2563eb" if v > 0 else "#dc2626" for v in top_loadings["pc1_loading"]]
    plt.barh(top_loadings["feature"], top_loadings["pc1_loading"], color=colors)
    plt.axvline(0, color="black", linestyle="--", alpha=0.7)
    plt.title("Top Feature Loadings on Principal Component 1 (PC1)", fontweight="bold")
    plt.xlabel("Loading Weight (Eigenvector Coordinate)")
    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, "05_pca_feature_loadings.png")
    plt.savefig(fig5_path, dpi=200)
    plt.close()
    print(f"✓ Saved Figure 5: {fig5_path}")

    # -------------------------------------------------------------------------
    # 5. Cluster Interpretation & Profiles
    # -------------------------------------------------------------------------
    print(f"\n>>> [Stage 5/8] Generating Cluster Profiles & Acoustic Impressions...")
    profiles = compute_cluster_profiles(df_clean, chosen_labels, retained_feats)
    for p in profiles:
        dom_words = ", ".join([f"{w['word'].upper()} ({w['percentage']}%)" for w in p["word_composition"][:3]])
        top_feats = ", ".join([f"{f['feature']} (z={f['z_score']})" for f in p["top_defining_features"][:3]])
        print(f"\n  ● Cluster {p['cluster_id']} ({p['percentage']}% of clips, n={p['size']}):")
        print(f"    - Dominant commands: {dom_words}")
        print(f"    - Acoustic profile:  {p['plain_english_description']}")
        print(f"    - Defining features: {top_feats}")

    # -------------------------------------------------------------------------
    # 6. Word-by-Cluster Cross-Tabulation & Heatmap
    # -------------------------------------------------------------------------
    print(f"\n>>> [Stage 6/8] Word-by-Cluster Heatmap & Confusable Pairs...")
    heatmap_data = compute_word_cluster_heatmap(df_clean, chosen_labels)
    df_ct = pd.DataFrame(heatmap_data["percentage_matrix"], index=heatmap_data["words"], columns=heatmap_data["clusters"])

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_ct, annot=True, fmt=".1f", cmap="Blues", cbar_kws={"label": "% of Word Clips in Cluster"}, linewidths=0.5)
    plt.title(f"Command Word Distribution Across Clusters (K = {chosen_k})", fontweight="bold", pad=12)
    plt.xlabel("Assigned Cluster")
    plt.ylabel("Spoken Voice Command")
    plt.tight_layout()
    fig6_path = os.path.join(figures_dir, "06_word_cluster_heatmap.png")
    plt.savefig(fig6_path, dpi=200)
    plt.close()
    print(f"✓ Saved Figure 6: {fig6_path}")

    # Confusable Pairs Hazard Ranking
    confusable_pairs = compute_confusable_pairs(df_clean, chosen_labels, X_scaled, retained_feats)
    print("\n--- TOP CONFUSABLE COMMAND PAIRS (ACCESSIBILITY SAFETY HAZARDS) ---")
    for cp in confusable_pairs[:6]:
        warn_str = " [CRITICAL SAFETY HAZARD]" if cp["is_critical_risk"] else ""
        print(f"  • {cp['word1'].upper()} <-> {cp['word2'].upper()}{warn_str}: "
              f"Confusability={cp['confusability_score']:.3f} | Overlap={cp['cluster_overlap_sim']*100:.1f}% | Distance={cp['acoustic_centroid_distance']:.2f}")
        if cp["critical_warning"]:
            print(f"    WARNING: {cp['critical_warning']}")

    # Save Figure 7: Confusable Pairs Matrix
    df_work = df_clean.copy()
    df_work["cluster"] = chosen_labels
    words_list = sorted(list(df_work["word"].unique()))
    pair_matrix = np.zeros((len(words_list), len(words_list)))
    for cp in confusable_pairs:
        i = words_list.index(cp["word1"])
        j = words_list.index(cp["word2"])
        pair_matrix[i, j] = cp["confusability_score"]
        pair_matrix[j, i] = cp["confusability_score"]

    plt.figure(figsize=(8, 7))
    sns.heatmap(pair_matrix, xticklabels=[w.upper() for w in words_list], yticklabels=[w.upper() for w in words_list],
                annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "Confusability Score"})
    plt.title("Pairwise Command Confusability Risk Matrix", fontweight="bold", pad=12)
    plt.tight_layout()
    fig7_path = os.path.join(figures_dir, "07_confusable_pairs_matrix.png")
    plt.savefig(fig7_path, dpi=200)
    plt.close()
    print(f"✓ Saved Figure 7: {fig7_path}")

    # -------------------------------------------------------------------------
    # 7. Stability Assessment across 10 Random Seeds
    # -------------------------------------------------------------------------
    print(f"\n>>> [Stage 7/8] Evaluating Model Stability (10 Random Restarts)...")
    stability = evaluate_stability(X_scaled, k=chosen_k, distance=chosen_dist, num_seeds=10, base_seed=args.seed)
    print(f"✓ Mean Adjusted Rand Index (ARI): {stability['mean_ari']:.4f} (Min: {stability['min_ari']:.4f}, Std: {stability['std_ari']:.4f})")
    print(f"✓ Cross-Metric ARI (Euclidean vs Manhattan): {stability['distance_cross_ari_euclidean_vs_manhattan']:.4f}")
    print(f"✓ Stability Verdict: {stability['stability_verdict']}")

    # Key Feature ANOVA
    f_contribs = compute_key_feature_contributions(X_scaled, chosen_labels, retained_feats)
    df_f = pd.DataFrame(f_contribs[:10])

    plt.figure(figsize=(10, 5))
    plt.barh(df_f["feature"], df_f["f_score"], color=OKABE_ITO[2])
    plt.title(f"Top 10 Acoustic Features Separating Clusters (ANOVA F-Statistic)", fontweight="bold")
    plt.xlabel("ANOVA F-Score (Cluster Separation Power)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig8_path = os.path.join(figures_dir, "08_key_feature_contributions_anova.png")
    plt.savefig(fig8_path, dpi=200)
    plt.close()
    print(f"✓ Saved Figure 8: {fig8_path}")

    # -------------------------------------------------------------------------
    # 8. Report Generation & Summary
    # -------------------------------------------------------------------------
    print(f"\n>>> [Stage 8/8] Generating Standalone HTML & Publication PDF Reports...")
    report_bundle = {
        "suggestion": sug,
        "metrics_table": results["comparison_table"],
        "preprocessing_log": prep_log,
        "pca_data": pca_data,
        "profiles": profiles,
        "confusable_pairs": confusable_pairs,
        "stability": stability,
    }

    report_html_path = os.path.join(reports_dir, "acoustic_clustering_report.html")
    report_pdf_path = os.path.join(reports_dir, "acoustic_clustering_report.pdf")
    generate_html_report(report_bundle, report_html_path)
    generate_pdf_report(report_bundle, report_pdf_path)

    # Also mirror to data/ for backward compatibility
    import shutil
    shutil.copy(report_html_path, os.path.join(data_dir, "acoustic_clustering_report.html"))
    shutil.copy(report_pdf_path, os.path.join(data_dir, "acoustic_clustering_report.pdf"))

    elapsed = time.time() - start_time
    print(f"✓ Generated HTML Report: {report_html_path}")
    print(f"✓ Generated PDF Report:  {report_pdf_path}")
    print("\n" + "=" * 80)
    print("  ANALYSIS EXECUTION COMPLETE! (Pure Python Environment)")
    print(f"  Total elapsed time: {elapsed:.2f} seconds")
    print(f"  All 8 high-resolution figures saved in: {figures_dir}")
    print(f"  Start interactive web app anytime with: python run_server.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
