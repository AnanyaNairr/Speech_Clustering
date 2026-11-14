/**
 * Plotly-based Interactive Visualization Library.
 * Features color-blind-safe palettes (Okabe-Ito), dark/light theme awareness,
 * accessible tooltips, and responsive layout scaling.
 */

// Color-blind safe palette (Okabe-Ito / ColorBrewer Accessible)
const COLORBLIND_PALETTE = [
  '#0072B2', // Blue
  '#D55E00', // Vermilion / Orange
  '#009E73', // Bluish Green
  '#CC79A7', // Reddish Purple
  '#E69F00', // Orange
  '#56B4E9', // Sky Blue
  '#F0E442', // Yellow
  '#999999', // Gray
  '#332288', // Indigo
  '#882255', // Wine
];

function getThemeLayout() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  return {
    paper_bgcolor: isDark ? '#121929' : '#ffffff',
    plot_bgcolor: isDark ? '#1a2338' : '#f8fafc',
    font: {
      family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      color: isDark ? '#f8fafc' : '#0f172a',
      size: 12,
    },
    gridcolor: isDark ? '#27354f' : '#e2e8f0',
    zerolinecolor: isDark ? '#334155' : '#cbd5e1',
  };
}

/**
 * 1. Metrics Comparison Charts (Elbow WCSS, Silhouette Score, Davies-Bouldin)
 */
function renderMetricsCharts(metricsData) {
  const theme = getThemeLayout();
  const kRange = metricsData.k_range;
  const euc = metricsData.euclidean;
  const manh = metricsData.manhattan;

  // Chart 1: WCSS (Elbow plot)
  const traceWcssEuc = {
    x: kRange,
    y: euc.map(d => d.wcss),
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Euclidean (K-Means)',
    line: { color: '#0072B2', width: 3 },
    marker: { size: 8, symbol: 'circle' },
    hovertemplate: 'K=%{x}<br>WCSS: %{y:.1f}<extra></extra>',
  };

  const traceWcssManh = {
    x: kRange,
    y: manh.map(d => d.wcss),
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Manhattan (K-Medians)',
    line: { color: '#D55E00', width: 3, dash: 'dot' },
    marker: { size: 8, symbol: 'diamond' },
    hovertemplate: 'K=%{x}<br>L1 Inertia: %{y:.1f}<extra></extra>',
  };

  const layoutWcss = {
    ...theme,
    title: { text: '<b>Elbow Plot: Inertia / WCSS vs K</b>', font: { size: 14 } },
    xaxis: { title: 'Number of Clusters (K)', dtick: 1, gridcolor: theme.gridcolor },
    yaxis: { title: 'Within-Cluster Sum of Distances', gridcolor: theme.gridcolor },
    margin: { t: 40, b: 40, l: 60, r: 20 },
    legend: { orientation: 'h', y: -0.2 },
  };

  Plotly.newPlot('chart-elbow', [traceWcssEuc, traceWcssManh], layoutWcss, { responsive: true, displayModeBar: false });

  // Chart 2: Silhouette Score
  const traceSilEuc = {
    x: kRange,
    y: euc.map(d => d.silhouette),
    type: 'bar',
    name: 'Euclidean',
    marker: { color: '#0072B2' },
    hovertemplate: 'K=%{x}<br>Silhouette: %{y:.4f}<extra></extra>',
  };

  const traceSilManh = {
    x: kRange,
    y: manh.map(d => d.silhouette),
    type: 'bar',
    name: 'Manhattan',
    marker: { color: '#D55E00' },
    hovertemplate: 'K=%{x}<br>Silhouette: %{y:.4f}<extra></extra>',
  };

  const layoutSil = {
    ...theme,
    title: { text: '<b>Silhouette Score (Higher is Better)</b>', font: { size: 14 } },
    xaxis: { title: 'Number of Clusters (K)', dtick: 1, gridcolor: theme.gridcolor },
    yaxis: { title: 'Silhouette Score', gridcolor: theme.gridcolor },
    barmode: 'group',
    margin: { t: 40, b: 40, l: 60, r: 20 },
    legend: { orientation: 'h', y: -0.2 },
  };

  Plotly.newPlot('chart-silhouette', [traceSilEuc, traceSilManh], layoutSil, { responsive: true, displayModeBar: false });

  // Chart 3: Davies-Bouldin Index
  const traceDbEuc = {
    x: kRange,
    y: euc.map(d => d.davies_bouldin),
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Euclidean',
    line: { color: '#0072B2', width: 2.5 },
    marker: { size: 8 },
    hovertemplate: 'K=%{x}<br>DB Index: %{y:.4f}<extra></extra>',
  };

  const traceDbManh = {
    x: kRange,
    y: manh.map(d => d.davies_bouldin),
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Manhattan',
    line: { color: '#D55E00', width: 2.5, dash: 'dash' },
    marker: { size: 8, symbol: 'square' },
    hovertemplate: 'K=%{x}<br>DB Index: %{y:.4f}<extra></extra>',
  };

  const layoutDb = {
    ...theme,
    title: { text: '<b>Davies-Bouldin Index (Lower is Better)</b>', font: { size: 14 } },
    xaxis: { title: 'Number of Clusters (K)', dtick: 1, gridcolor: theme.gridcolor },
    yaxis: { title: 'Davies-Bouldin Score', gridcolor: theme.gridcolor },
    margin: { t: 40, b: 40, l: 60, r: 20 },
    legend: { orientation: 'h', y: -0.2 },
  };

  Plotly.newPlot('chart-davies-bouldin', [traceDbEuc, traceDbManh], layoutDb, { responsive: true, displayModeBar: false });
}

/**
 * 2. Correlation Heatmap
 */
function renderCorrelationHeatmap(corrData) {
  const theme = getThemeLayout();
  // Filter top correlated or subset for clean readability
  const cols = corrData.columns;
  const values = corrData.values;

  const trace = {
    z: values,
    x: cols,
    y: cols,
    type: 'heatmap',
    colorscale: 'Viridis',
    zmin: -1.0,
    zmax: 1.0,
    colorbar: { title: 'Pearson r', thickness: 15 },
    hovertemplate: '%{x} & %{y}<br>r = %{z:.3f}<extra></extra>',
  };

  const layout = {
    ...theme,
    title: { text: '<b>Acoustic Feature Pairwise Correlation Matrix</b>', font: { size: 14 } },
    xaxis: { tickangle: -45, automargin: true },
    yaxis: { automargin: true },
    margin: { t: 40, b: 120, l: 150, r: 30 },
  };

  Plotly.newPlot('chart-correlation-heatmap', [trace], layout, { responsive: true });
}

/**
 * 3. PCA 2D Scatter Plot with Cluster / Word Coloring Toggle & User Clip Overlay
 */
function renderPcaScatter(pcaData, labels, words, colorMode = 'cluster', userPoint = null) {
  const theme = getThemeLayout();
  const coords = pcaData.pc_coords_2d;
  const expVar = pcaData.explained_variance_ratio;

  const traces = [];

  if (colorMode === 'cluster') {
    // Group by cluster
    const clusterIds = Array.from(new Set(labels)).sort((a, b) => a - b);
    clusterIds.forEach((cId) => {
      const idxs = [];
      for (let i = 0; i < labels.length; i++) {
        if (labels[i] === cId) idxs.push(i);
      }

      traces.push({
        x: idxs.map(i => coords[i][0]),
        y: idxs.map(i => coords[i][1]),
        mode: 'markers',
        type: 'scatter',
        name: `Cluster ${cId}`,
        customdata: idxs.map(i => ({ idx: i, word: words[i], cluster: cId })),
        hovertemplate: '<b>Word: "%{customdata.word}"</b><br>Cluster: %{customdata.cluster}<br>PC1: %{x:.2f}, PC2: %{y:.2f}<br><span style="color:#38bdf8;">🔊 Click point to hear audio clip</span><extra></extra>',
        marker: {
          color: COLORBLIND_PALETTE[cId % COLORBLIND_PALETTE.length],
          size: 7,
          opacity: 0.75,
        },
      });
    });
  } else {
    // Group by actual spoken word
    const uniqueWords = Array.from(new Set(words)).sort();
    uniqueWords.forEach((word, wIdx) => {
      const idxs = [];
      for (let i = 0; i < words.length; i++) {
        if (words[i] === word) idxs.push(i);
      }

      traces.push({
        x: idxs.map(i => coords[i][0]),
        y: idxs.map(i => coords[i][1]),
        mode: 'markers',
        type: 'scatter',
        name: `Word: "${word}"`,
        customdata: idxs.map(i => ({ idx: i, word: word, cluster: labels[i] })),
        hovertemplate: '<b>Word: "%{customdata.word}"</b><br>Cluster: %{customdata.cluster}<br>PC1: %{x:.2f}, PC2: %{y:.2f}<br><span style="color:#38bdf8;">🔊 Click point to hear audio clip</span><extra></extra>',
        marker: {
          color: COLORBLIND_PALETTE[wIdx % COLORBLIND_PALETTE.length],
          size: 7,
          opacity: 0.75,
        },
      });
    });
  }

  // If a live recorded user voice point exists, plot it as a pulsing red star
  if (userPoint) {
    traces.push({
      x: [userPoint.pc1],
      y: [userPoint.pc2],
      mode: 'markers+text',
      type: 'scatter',
      name: '★ Your Voice Clip',
      text: ['★ Your Voice Clip'],
      textposition: 'top center',
      textfont: { size: 13, color: '#dc2626', family: 'Arial Black' },
      marker: {
        color: '#dc2626',
        size: 16,
        symbol: 'star',
        line: { color: '#ffffff', width: 2 },
      },
      hoverinfo: 'text',
    });
  }

  const layout = {
    ...theme,
    title: {
      text: `<b>PCA 2D Acoustic Projection (${colorMode === 'cluster' ? 'Colored by Cluster' : 'Colored by Spoken Word'})</b>`,
      font: { size: 14 },
    },
    xaxis: {
      title: `PC1 (${(expVar[0] * 100).toFixed(1)}% Variance Explained)`,
      gridcolor: theme.gridcolor,
    },
    yaxis: {
      title: `PC2 (${(expVar[1] * 100).toFixed(1)}% Variance Explained)`,
      gridcolor: theme.gridcolor,
    },
    margin: { t: 40, b: 50, l: 60, r: 20 },
    legend: { orientation: 'h', y: -0.25 },
  };

  const chartEl = document.getElementById('chart-pca-2d');
  Plotly.newPlot(chartEl, traces, layout, { responsive: true }).then(() => {
    if (chartEl && chartEl.on) {
      chartEl.removeAllListeners && chartEl.removeAllListeners('plotly_click');
      chartEl.on('plotly_click', (data) => {
        if (!data || !data.points || data.points.length === 0) return;
        const pt = data.points[0];
        if (pt.customdata && typeof window.onPcaPointClicked === 'function') {
          window.onPcaPointClicked(pt.customdata, pt.x, pt.y);
        }
      });
    }
  });
}

/**
 * 4. PCA 3D Projection
 */
function renderPca3D(pcaData, labels, words, colorMode = 'cluster') {
  const theme = getThemeLayout();
  const coords = pcaData.pc_coords_3d;
  if (!coords || coords.length === 0) return;
  const expVar = pcaData.explained_variance_ratio;

  const traces = [];
  const clusterIds = Array.from(new Set(labels)).sort((a, b) => a - b);

  clusterIds.forEach((cId) => {
    const idxs = [];
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] === cId) idxs.push(i);
    }

    traces.push({
      x: idxs.map(i => coords[i][0]),
      y: idxs.map(i => coords[i][1]),
      z: idxs.map(i => coords[i][2]),
      mode: 'markers',
      type: 'scatter3d',
      name: `Cluster ${cId}`,
      text: idxs.map(i => `Word: "${words[i]}"<br>Cluster: ${cId}`),
      hoverinfo: 'text',
      marker: {
        color: COLORBLIND_PALETTE[cId % COLORBLIND_PALETTE.length],
        size: 4,
        opacity: 0.8,
      },
    });
  });

  const layout = {
    ...theme,
    title: { text: '<b>PCA 3D Interactive Acoustic Space</b>', font: { size: 14 } },
    scene: {
      xaxis: { title: 'PC1', gridcolor: theme.gridcolor },
      yaxis: { title: 'PC2', gridcolor: theme.gridcolor },
      zaxis: { title: 'PC3', gridcolor: theme.gridcolor },
    },
    margin: { t: 40, b: 30, l: 20, r: 20 },
  };

  Plotly.newPlot('chart-pca-3d', traces, layout, { responsive: true });
}

/**
 * 5. Feature Loadings (Top Features driving PC1 and PC2)
 */
function renderLoadingsChart(pcaData) {
  const theme = getThemeLayout();
  const topPc1 = pcaData.top_pc1_features || [];
  const topPc2 = pcaData.top_pc2_features || [];

  const trace1 = {
    x: topPc1.map(d => d.pc1_loading),
    y: topPc1.map(d => d.feature),
    type: 'bar',
    orientation: 'h',
    name: 'PC1 Loading',
    marker: { color: '#0072B2' },
    hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
  };

  const trace2 = {
    x: topPc2.map(d => d.pc2_loading),
    y: topPc2.map(d => d.feature),
    type: 'bar',
    orientation: 'h',
    name: 'PC2 Loading',
    marker: { color: '#D55E00' },
    hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
  };

  const layout = {
    ...theme,
    title: { text: '<b>Top Acoustic Feature Loadings (PC1 vs PC2)</b>', font: { size: 14 } },
    barmode: 'group',
    margin: { t: 40, b: 50, l: 180, r: 20 },
    xaxis: { title: 'PCA Loading Weight', gridcolor: theme.gridcolor },
    yaxis: { automargin: true },
    legend: { orientation: 'h', y: -0.25 },
  };

  Plotly.newPlot('chart-loadings', [trace1, trace2], layout, { responsive: true });
}

/**
 * 6. Word-by-Cluster Cross-Tabulation Heatmap
 */
function renderWordClusterHeatmap(heatmapData) {
  const theme = getThemeLayout();
  const trace = {
    z: heatmapData.percentage_matrix,
    x: heatmapData.clusters,
    y: heatmapData.words,
    type: 'heatmap',
    colorscale: 'Blues',
    colorbar: { title: '% of Word Clips' },
    hovertemplate: 'Word: %{y}<br>%{x}: %{z:.1f}%<extra></extra>',
  };

  const layout = {
    ...theme,
    title: { text: '<b>Spoken Word vs Cluster Distribution (% of Clips)</b>', font: { size: 14 } },
    xaxis: { title: 'Cluster Group', gridcolor: theme.gridcolor },
    yaxis: { title: 'Command Word', automargin: true },
    margin: { t: 40, b: 50, l: 80, r: 20 },
  };

  Plotly.newPlot('chart-word-cluster-heatmap', [trace], layout, { responsive: true });
}

