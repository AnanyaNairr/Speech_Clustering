"""
Comprehensive Report Generator.
Exports analysis results to both clean standalone HTML and formatted PDF using ReportLab.
"""
import os
import json
import logging
from typing import Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

logger = logging.getLogger(__name__)


def generate_html_report(data: Dict[str, Any], output_path: str) -> str:
    """
    Generates a rich, print-ready HTML report with modern CSS styling.
    """
    suggestion = data.get("suggestion", {})
    profiles = data.get("profiles", [])
    stability = data.get("stability", {})
    confusable_pairs = data.get("confusable_pairs", [])
    metrics_table = data.get("metrics_table", [])
    prep_log = data.get("preprocessing_log", {})
    pca_data = data.get("pca_data", {})

    metrics_rows_html = "".join([
        f"<tr><td>{m['k']}</td><td>{m['distance']}</td><td>{m['wcss']:.1f}</td><td>{m['silhouette']:.4f}</td><td>{m['davies_bouldin']:.4f}</td></tr>"
        for m in metrics_table
    ])

    profiles_html = "".join([
        f"""
        <div class="card" style="margin-bottom: 1.25rem;">
            <h3>Cluster {p['cluster_id']} ({p['percentage']}% of clips, {p['size']} samples)</h3>
            <p><strong>Acoustic Profile:</strong> {p['plain_english_description']}</p>
            <p><strong>Dominant Commands:</strong> {', '.join([f"{w['word']} ({w['percentage']}%)" for w in p['word_composition'][:4]])}</p>
            <p><strong>Distinctive Acoustic Traits:</strong> {', '.join([f"{f['feature']} (z={f['z_score']})" for f in p['top_defining_features'][:3]])}</p>
        </div>
        """
        for p in profiles
    ])

    confusable_rows_html = "".join([
        f"""
        <tr style="{'background-color: #fee2e2;' if p['is_critical_risk'] else ''}">
            <td><strong>{p['word1']}</strong> & <strong>{p['word2']}</strong></td>
            <td>{p['confusability_score']:.3f}</td>
            <td>{p['cluster_overlap_sim']:.3f}</td>
            <td>{p['acoustic_centroid_distance']:.2f}</td>
            <td><span class="badge {'badge-danger' if p['is_critical_risk'] else 'badge-warning'}">{'CRITICAL RISK' if p['is_critical_risk'] else 'Moderate'}</span> {p['critical_warning']}</td>
        </tr>
        """
        for p in confusable_pairs[:8]
    ])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Acoustic Feature Clustering Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #1e293b; max-width: 960px; margin: 0 auto; padding: 2rem; }}
    h1 {{ color: #0f172a; border-bottom: 2px solid #2563eb; padding-bottom: 0.5rem; }}
    h2 {{ color: #1e3a8a; margin-top: 2rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.3rem; }}
    h3 {{ color: #1e293b; margin-top: 1rem; }}
    table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 0.6rem 0.8rem; text-align: left; }}
    th {{ background-color: #f1f5f9; font-weight: 600; }}
    .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; }}
    .alert {{ background: #eff6ff; border-left: 4px solid #2563eb; padding: 1rem; margin: 1rem 0; border-radius: 4px; }}
    .alert-warning {{ background: #fefce8; border-left-color: #eab308; }}
    .badge {{ display: inline-block; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }}
    .badge-danger {{ background: #dc2626; color: white; }}
    .badge-warning {{ background: #d97706; color: white; }}
    @media print {{ body {{ padding: 0; }} .card {{ break-inside: avoid; }} }}
</style>
</head>
<body>
    <h1>Acoustic Feature Clustering for Hands-Free Accessibility Control</h1>
    <p><em>Comprehensive Technical Report & Responsible Accessibility Evaluation</em></p>

    <div class="alert">
        <strong>Selected Model Configuration:</strong> K = {suggestion.get('recommended_k', 5)}, Distance: {suggestion.get('recommended_distance', 'Euclidean')}<br>
        <strong>Composite Quality Score:</strong> {suggestion.get('composite_score', 0.0):.3f} | <strong>Silhouette Score:</strong> {suggestion.get('silhouette', 0.0):.3f} | <strong>Davies-Bouldin Index:</strong> {suggestion.get('davies_bouldin', 0.0):.3f}<br>
        <p style="margin-top: 0.5rem;">{suggestion.get('plain_english_reasoning', '')}</p>
    </div>

    <h2>1. Data Preparation & Preprocessing Log</h2>
    <p>Clips were standardized to 1.0 second (16kHz). Extracted 45 acoustic features across 10 balanced command words (200 clips/word).</p>
    <ul>
        <li><strong>Label Leakage Prevention:</strong> {prep_log.get('label_encoding_note', '')}</li>
        <li><strong>Missing / Invalid Handling:</strong> {prep_log.get('missing_inf_handling', {}).get('action', 'Clean dataset')}</li>
        <li><strong>Constant Features Dropped:</strong> {len(prep_log.get('dropped_constant_features', []))}</li>
        <li><strong>Collinear Features Dropped (|r| > 0.95):</strong> {len(prep_log.get('dropped_collinear_features', []))} features removed to prevent distortion.</li>
        <li><strong>Final Feature Count:</strong> {prep_log.get('final_feature_count', 0)} normalized acoustic features scaled via StandardScaler.</li>
    </ul>

    <h2>2. Clustering Evaluation (K = 2 to 10)</h2>
    <table>
        <thead>
            <tr><th>K</th><th>Distance / Algorithm</th><th>WCSS (Inertia)</th><th>Silhouette Score</th><th>Davies-Bouldin Index</th></tr>
        </thead>
        <tbody>
            {metrics_rows_html}
        </tbody>
    </table>

    <h2>3. PCA Dimensionality Reduction</h2>
    <p>Standardized acoustic dimensions were projected via PCA. The top 2 principal components capture <strong>{sum(pca_data.get('explained_variance_ratio', [0, 0])[:2]) * 100:.1f}%</strong> of total acoustic variance.</p>
    <p><strong>Top PC1 Loadings:</strong> {', '.join([f"{f['feature']} ({f['pc1_loading']:.2f})" for f in pca_data.get('top_pc1_features', [])[:4]])}</p>
    <p><strong>Top PC2 Loadings:</strong> {', '.join([f"{f['feature']} ({f['pc2_loading']:.2f})" for f in pca_data.get('top_pc2_features', [])[:4]])}</p>

    <h2>4. Cluster Acoustic Profiles</h2>
    {profiles_html}

    <h2>5. Command Confusability & Safety Risks</h2>
    <p>Phonetically similar command words share acoustic clusters and present voice-control navigation hazards.</p>
    <table>
        <thead>
            <tr><th>Command Pair</th><th>Confusability Index</th><th>Cluster Overlap</th><th>Acoustic Distance</th><th>Accessibility Hazard Analysis</th></tr>
        </thead>
        <tbody>
            {confusable_rows_html}
        </tbody>
    </table>

    <h2>6. Stability Analysis (10 Random Restarts)</h2>
    <div class="card">
        <p><strong>Adjusted Rand Index (ARI):</strong> Mean = {stability.get('mean_ari', 0.0):.3f}, Min = {stability.get('min_ari', 0.0):.3f}, Std = {stability.get('std_ari', 0.0):.3f}</p>
        <p><strong>Cross-Metric Agreement (Euclidean vs Manhattan):</strong> ARI = {stability.get('distance_cross_ari_euclidean_vs_manhattan', 0.0):.3f}</p>
        <p><strong>Verdict:</strong> {stability.get('stability_verdict', '')}</p>
    </div>

    <h2>7. Critical Limitations & Responsible Use</h2>
    <div class="alert alert-warning">
        <p><strong>1. Dataset Demographics & Volunteer Bias:</strong> Google Speech Commands consists primarily of healthy adult volunteers speaking clear, isolated words in controlled conditions. Children, elderly users, non-native accents, and distinct regional dialects are under-represented.</p>
        <p><strong>2. Exclusion of Dysarthric & Impaired Speech:</strong> Individuals with motor disabilities often present dysarthric speech, vocal tremor, reduced breath support, or atypical articulation—the exact population needing hands-free accessibility. These clusters DO NOT reflect dysarthric acoustic patterns.</p>
        <p><strong>3. Exploratory Disclaimer:</strong> These unsupervised clusters represent mathematical acoustic proximities from a single sample. They must never be treated as immutable phonetic truths. Any production accessibility control interface must conduct empirical user testing and offer user-calibrated command customization.</p>
    </div>
</body>
</html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Generated HTML report at {output_path}")
    return output_path


def generate_pdf_report(data: Dict[str, Any], output_path: str) -> str:
    """
    Generates a professional PDF report using ReportLab.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle("RepTitle", parent=styles["Heading1"], fontSize=18, leading=22, textColor=colors.HexColor("#0f172a"))
    h2_style = ParagraphStyle("RepH2", parent=styles["Heading2"], fontSize=13, leading=16, textColor=colors.HexColor("#1e3a8a"), spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle("RepBody", parent=styles["Normal"], fontSize=9, leading=12, textColor=colors.HexColor("#334155"))
    body_bold = ParagraphStyle("RepBold", parent=body_style, fontName="Helvetica-Bold")
    table_cell = ParagraphStyle("TableCell", parent=styles["Normal"], fontSize=8, leading=10)
    table_header = ParagraphStyle("TableHead", parent=table_cell, fontName="Helvetica-Bold", textColor=colors.HexColor("#0f172a"))

    story = []

    # Title & Header
    story.append(Paragraph("Acoustic Feature Clustering for Hands-Free Accessibility Control", title_style))
    story.append(Paragraph("Automated Analysis Report & Responsible Use Evaluation", ParagraphStyle("Sub", parent=body_style, fontSize=10, textColor=colors.HexColor("#64748b"))))
    story.append(Spacer(1, 10))

    # Configuration Box
    sug = data.get("suggestion", {})
    conf_text = (
        f"<b>Recommended Configuration:</b> K = {sug.get('recommended_k', 5)} ({sug.get('recommended_distance', 'Euclidean')}) | "
        f"<b>Composite Score:</b> {sug.get('composite_score', 0.0):.3f} | "
        f"<b>Silhouette Score:</b> {sug.get('silhouette', 0.0):.3f} | "
        f"<b>Davies-Bouldin Index:</b> {sug.get('davies_bouldin', 0.0):.3f}<br/>"
        f"<i>{sug.get('plain_english_reasoning', '')}</i>"
    )
    conf_table = Table([[Paragraph(conf_text, body_style)]], colWidths=[530])
    conf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#2563eb")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(conf_table)
    story.append(Spacer(1, 10))

    # 1. Preprocessing Summary
    story.append(Paragraph("1. Data Preprocessing & Variable Relevance", h2_style))
    prep = data.get("preprocessing_log", {})
    prep_text = (
        f"Sampled 10 balanced command words (200 clips each, 2,000 total). Audio padded or trimmed to 1.0s @ 16kHz.<br/>"
        f"• <b>Label Isolation:</b> Word labels are label-encoded for interpretation ONLY and completely excluded from clustering inputs.<br/>"
        f"• <b>Collinearity Filter:</b> Dropped {len(prep.get('dropped_collinear_features', []))} features with correlation |r| > 0.95.<br/>"
        f"• <b>Constant Features:</b> Dropped {len(prep.get('dropped_constant_features', []))} near-constant features.<br/>"
        f"• <b>Standardization:</b> Applied StandardScaler across all {prep.get('final_feature_count', 0)} retained features."
    )
    story.append(Paragraph(prep_text, body_style))
    story.append(Spacer(1, 8))

    # 2. Metrics Table
    story.append(Paragraph("2. Clustering Evaluation Comparison Table (K = 2 to 10)", h2_style))
    metrics_data = [[
        Paragraph("K", table_header),
        Paragraph("Distance / Algorithm", table_header),
        Paragraph("WCSS (Inertia)", table_header),
        Paragraph("Silhouette", table_header),
        Paragraph("Davies-Bouldin", table_header),
    ]]
    for m in data.get("metrics_table", [])[:10]:
        metrics_data.append([
            Paragraph(str(m["k"]), table_cell),
            Paragraph(m["distance"], table_cell),
            Paragraph(f"{m['wcss']:.1f}", table_cell),
            Paragraph(f"{m['silhouette']:.4f}", table_cell),
            Paragraph(f"{m['davies_bouldin']:.4f}", table_cell),
        ])
    t_metrics = Table(metrics_data, colWidths=[30, 160, 100, 120, 120])
    t_metrics.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 10))

    # 3. Confusable Pairs
    story.append(Paragraph("3. Acoustic Confusability & Safety Risks", h2_style))
    cp_data = [[
        Paragraph("Commands", table_header),
        Paragraph("Confusability", table_header),
        Paragraph("Overlap", table_header),
        Paragraph("Acoustic Dist", table_header),
        Paragraph("Hazard Analysis", table_header),
    ]]
    for p in data.get("confusable_pairs", [])[:6]:
        cp_data.append([
            Paragraph(f"<b>{p['word1']}</b> / <b>{p['word2']}</b>", table_cell),
            Paragraph(f"{p['confusability_score']:.3f}", table_cell),
            Paragraph(f"{p['cluster_overlap_sim']:.3f}", table_cell),
            Paragraph(f"{p['acoustic_centroid_distance']:.2f}", table_cell),
            Paragraph(f"{'CRITICAL: ' if p['is_critical_risk'] else ''}{p['critical_warning'] or 'Phonetic similarity'}", table_cell),
        ])
    t_cp = Table(cp_data, colWidths=[90, 70, 60, 70, 240])
    t_cp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_cp)
    story.append(Spacer(1, 10))

    # 4. Stability
    story.append(Paragraph("4. Clustering Stability Analysis", h2_style))
    stab = data.get("stability", {})
    stab_text = (
        f"<b>10-Seed ARI:</b> Mean = {stab.get('mean_ari', 0.0):.3f}, Min = {stab.get('min_ari', 0.0):.3f}, Std = {stab.get('std_ari', 0.0):.3f}.<br/>"
        f"<b>Cross-Metric ARI:</b> {stab.get('distance_cross_ari_euclidean_vs_manhattan', 0.0):.3f} (Euclidean vs Manhattan).<br/>"
        f"<b>Verdict:</b> {stab.get('stability_verdict', '')}"
    )
    story.append(Paragraph(stab_text, body_style))
    story.append(Spacer(1, 10))

    # 5. Limitations & Responsible Use
    story.append(Paragraph("5. Limitations & Responsible Accessibility Use", h2_style))
    lim_text = (
        "<b>Volunteer Demographics:</b> Google Speech Commands contains predominantly clear recordings from volunteers. Accents, ages, microphones, and vocal stamina vary widely in real-world settings.<br/>"
        "<b>Under-representation of Target Accessibility Populations:</b> Individuals with dysarthria, cerebral palsy, or speech apraxia are not adequately represented. These acoustic clusters DO NOT model impaired speech acoustics.<br/>"
        "<b>Exploratory Nature:</b> Unsupervised clustering uncovers mathematical proximities in feature space, not clinical speech categories. Real-world systems must incorporate user-specific acoustic calibration and safety confirmation prompts."
    )
    lim_table = Table([[Paragraph(lim_text, body_style)]], colWidths=[530])
    lim_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fefce8")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#eab308")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(lim_table)

    doc.build(story)
    logger.info(f"Generated PDF report at {output_path}")
    return output_path

