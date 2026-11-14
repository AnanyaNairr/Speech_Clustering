/**
 * Main Application Logic & 5-Step Guided Stepper.
 * Connects frontend views with backend REST endpoints.
 */

window.appState = {
  currentStep: 1,
  datasetStatus: null,
  prepData: null,
  clusteringResults: null,
  activeConfig: { k: 2, distance: 'Euclidean' },
  analysis: null,
  pcaData: null,
  pcaColorMode: 'cluster',
  theme: localStorage.getItem('theme') || 'light',
};

// Initialize application
document.addEventListener('DOMContentLoaded', async () => {
  initTheme();
  setupEventListeners();
  await loadInitialPipelineStatus();
});

function initTheme() {
  const theme = window.appState.theme;
  document.documentElement.setAttribute('data-theme', theme);
  const themeBtn = document.getElementById('btn-theme-toggle');
  if (themeBtn) {
    themeBtn.textContent = theme === 'dark' ? '☀️' : '🌙';
    themeBtn.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
  }
}

function toggleTheme() {
  const newTheme = window.appState.theme === 'dark' ? 'light' : 'dark';
  window.appState.theme = newTheme;
  localStorage.setItem('theme', newTheme);
  document.documentElement.setAttribute('data-theme', newTheme);
  const themeBtn = document.getElementById('btn-theme-toggle');
  if (themeBtn) {
    themeBtn.textContent = newTheme === 'dark' ? '☀️' : '🌙';
  }

  // Re-render charts with new theme colors
  refreshCurrentCharts();
}

function setupEventListeners() {
  // Theme Toggle
  const themeBtn = document.getElementById('btn-theme-toggle');
  if (themeBtn) themeBtn.addEventListener('click', toggleTheme);

  // Stepper navigation buttons
  document.querySelectorAll('.step-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      const step = parseInt(btn.getAttribute('data-step'));
      goToStep(step);
    });
  });

  // Next / Back buttons
  document.querySelectorAll('[data-nav="next"]').forEach((btn) => {
    btn.addEventListener('click', () => goToStep(window.appState.currentStep + 1));
  });

  document.querySelectorAll('[data-nav="prev"]').forEach((btn) => {
    btn.addEventListener('click', () => goToStep(window.appState.currentStep - 1));
  });

  // PCA Color toggle
  const pcaToggle = document.getElementById('pca-color-toggle');
  if (pcaToggle) {
    pcaToggle.addEventListener('change', (e) => {
      window.appState.pcaColorMode = e.target.value;
      if (window.appState.pcaData && window.appState.analysis) {
        renderPcaScatter(
          window.appState.pcaData,
          window.appState.analysis.labels,
          window.appState.analysis.words,
          window.appState.pcaColorMode
        );
      }
    });
  }

  // Model override selectors in Step 3
  const applyModelBtn = document.getElementById('btn-apply-model');
  if (applyModelBtn) {
    applyModelBtn.addEventListener('click', async () => {
      const kVal = parseInt(document.getElementById('select-k').value);
      const distVal = document.getElementById('select-distance').value;
      await changeActiveModel(kVal, distVal);
    });
  }
}

function goToStep(step) {
  if (step < 1 || step > 5) return;
  window.appState.currentStep = step;

  // Update Stepper UI
  document.querySelectorAll('.step-item').forEach((item) => {
    const s = parseInt(item.getAttribute('data-step'));
    item.classList.remove('active');
    if (s === step) {
      item.classList.add('active');
      item.querySelector('.step-btn').setAttribute('aria-current', 'step');
    } else if (s < step) {
      item.classList.add('completed');
      item.querySelector('.step-btn').removeAttribute('aria-current');
    } else {
      item.classList.remove('completed');
      item.querySelector('.step-btn').removeAttribute('aria-current');
    }
  });

  // Update Section Visibility
  document.querySelectorAll('.step-section').forEach((sec) => {
    sec.classList.remove('active');
  });
  const activeSec = document.getElementById(`step-${step}`);
  if (activeSec) activeSec.classList.add('active');

  // Window scroll to top of section
  window.scrollTo({ top: 180, behavior: 'smooth' });

  // Render step-specific charts if needed
  refreshCurrentCharts();
}

function refreshCurrentCharts() {
  const step = window.appState.currentStep;
  if (step === 2 && window.appState.prepData && window.appState.prepData.log.correlation_matrix) {
    renderCorrelationHeatmap(window.appState.prepData.log.correlation_matrix);
  } else if (step === 3 && window.appState.clusteringResults) {
    renderMetricsCharts(window.appState.clusteringResults);
  } else if (step === 4 && window.appState.analysis && window.appState.pcaData) {
    renderPcaScatter(
      window.appState.pcaData,
      window.appState.analysis.labels,
      window.appState.analysis.words,
      window.appState.pcaColorMode
    );
    renderPca3D(
      window.appState.pcaData,
      window.appState.analysis.labels,
      window.appState.analysis.words,
      window.appState.pcaColorMode
    );
    renderLoadingsChart(window.appState.pcaData);
    if (window.appState.analysis.heatmap) {
      renderWordClusterHeatmap(window.appState.analysis.heatmap);
    }
  }
}

async function loadInitialPipelineStatus() {
  try {
    const res = await fetch('/api/status');
    const status = await res.json();
    window.appState.datasetStatus = status;

    // Load rich audio metadata for words
    let wordsInfo = [];
    if (window.audioStudio) {
      wordsInfo = await window.audioStudio.loadWordsMetadata();
    }

    // Load Preprocessing
    const prepRes = await fetch('/api/preprocessing');
    if (prepRes.ok) {
      const prep = await prepRes.json();
      window.appState.prepData = prep;
      renderStep1(status, prep, wordsInfo);
      renderStep2(prep);
    }

    // Load Clustering Metrics
    const clustRes = await fetch('/api/clustering');
    if (clustRes.ok) {
      const clust = await clustRes.json();
      window.appState.clusteringResults = clust;
      renderStep3(clust);
    }

    // Load Analysis
    const analRes = await fetch('/api/analysis');
    if (analRes.ok) {
      const anal = await analRes.json();
      window.appState.analysis = anal;
      window.appState.pcaData = anal.pca_data;
      window.appState.activeConfig = { k: anal.k, distance: anal.distance };
      renderStep4(anal);
      renderStep5(anal);
    }

  } catch (err) {
    console.error('Failed to load initial pipeline status:', err);
  }
}

function renderStep1(status, prep, wordsInfo) {
  const container = document.getElementById('step-1-content');
  if (!container) return;

  const totalClips = prep.log.initial_sample_count || 2000;
  const wordClasses = prep.log.word_classes || status.core_words;
  const wordsList = (wordsInfo && wordsInfo.length > 0) ? wordsInfo : wordClasses.map(w => ({
    word: w,
    display: w.toUpperCase(),
    ipa: `/${w}/`,
    category: 'Spoken Command',
    acoustic_summary: '1-second 16kHz audio sample.',
    total_clips: 200,
  }));

  container.innerHTML = `
    <div class="grid-4" style="margin-bottom: 1.5rem;">
      <div class="stat-tile">
        <div class="stat-label">Core Commands</div>
        <div class="stat-value" style="color: var(--primary);">${wordClasses.length} Words</div>
        <div class="stat-sub">Balanced vocabulary</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">Total Audio Clips</div>
        <div class="stat-value">${totalClips}</div>
        <div class="stat-sub">200 clips sampled per command</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">Sampling Rate</div>
        <div class="stat-value">16,000 Hz</div>
        <div class="stat-sub">16-bit Mono PCM</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">Audio Quality</div>
        <div class="stat-value" style="color: var(--success);">Clean 1.00s</div>
        <div class="stat-sub">Energy-aligned & zero-padded</div>
      </div>
    </div>

    <!-- Acoustic Command Studio & Live Oscilloscope Banner -->
    <div class="card audio-studio-panel" id="audio-studio-banner" style="margin-bottom: 1.75rem; background: var(--bg-card); border: 1px solid var(--border); box-shadow: var(--shadow-lg);">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.85rem; flex-wrap: wrap; gap: 0.75rem;">
        <div>
          <h4 style="font-size: 1.2rem; font-weight: 700; display: flex; align-items: center; gap: 0.6rem;">
            <span class="live-dot" aria-hidden="true"></span>
            <span>Acoustic Command Studio & Live Oscilloscope</span>
          </h4>
          <p style="font-size: 0.88rem; color: var(--text-muted); margin-top: 0.15rem;">
            Click any command word below to stream real audio recordings from disk and watch real-time frequency resonances.
          </p>
        </div>
        <div style="display: flex; align-items: center; gap: 0.6rem;">
          <span style="font-size: 0.82rem; color: var(--text-secondary); font-weight: 600;">Slow Motion:</span>
          <button class="btn btn-sm btn-secondary speed-btn active" data-speed="1.0" onclick="window.audioStudio.setPlaybackSpeed(1.0)">1.0x</button>
          <button class="btn btn-sm btn-secondary speed-btn" data-speed="0.75" onclick="window.audioStudio.setPlaybackSpeed(0.75)" title="Listen at 0.75x speed to hear subtle phonetics">0.75x</button>
          <button class="btn btn-sm btn-secondary speed-btn" data-speed="0.5" onclick="window.audioStudio.setPlaybackSpeed(0.5)" title="Listen at 0.5x half speed">0.5x</button>
        </div>
      </div>

      <!-- Real-time Oscilloscope & Frequency Spectrum Canvas -->
      <div style="position: relative; width: 100%; height: 110px; border-radius: var(--radius); overflow: hidden; background: var(--bg-secondary); border: 1px solid var(--border-subtle); margin-bottom: 0.5rem;">
        <canvas id="studio-audio-canvas" style="width: 100%; height: 100%; display: block;"></canvas>
        <div id="studio-canvas-overlay" style="position: absolute; top: 10px; left: 16px; pointer-events: none; display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap;">
          <span id="studio-banner-word" style="font-size: 1.25rem; font-weight: 800; color: var(--text-primary); letter-spacing: 0.05em;">STOP</span>
          <span id="studio-banner-ipa" class="badge badge-primary" style="font-family: monospace; font-size: 0.85rem;">/stɒp/</span>
          <span id="studio-banner-speaker" class="badge badge-secondary" style="font-size: 0.8rem;">Speaker #1</span>
          <span id="studio-banner-category" class="badge badge-warning" style="font-size: 0.8rem;">Sibilant + Plosive</span>
        </div>
        <div id="studio-banner-traits" style="position: absolute; bottom: 8px; left: 16px; right: 16px; pointer-events: none; font-size: 0.82rem; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
          High-frequency hiss /s/ (>3500 Hz), silent stop gap, sharp plosive release burst.
        </div>
      </div>
    </div>

    <!-- 10 Interactive Audio Command Cards -->
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap; gap: 0.5rem;">
      <div>
        <h4 style="font-size: 1.15rem; font-weight: 700;">🔊 Interactive Command Soundboard (10 Core Words)</h4>
        <p style="font-size: 0.85rem; color: var(--text-muted);">
          Click <strong>"Hear Sound"</strong> on any card to listen. Use <strong>"Next Voice"</strong> to sample different volunteer speakers (varying accents, genders, and pitches).
        </p>
      </div>
      <button class="btn btn-sm btn-outline" onclick="
        const words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'];
        const randomWord = words[Math.floor(Math.random() * words.length)];
        window.audioStudio.playWord(randomWord);
      ">
        🎲 Play Random Voice
      </button>
    </div>

    <div class="audio-cards-grid">
      ${wordsList.map(item => `
        <div class="audio-command-card" data-word="${item.word}" id="card-${item.word}">
          <div>
            <div class="card-top-row">
              <div class="card-word-title">
                <span>${item.display}</span>
                <span class="card-ipa-tag">${item.ipa}</span>
              </div>
              <div class="mini-equalizer" title="Equalizer activity">
                <span class="bar"></span>
                <span class="bar"></span>
                <span class="bar"></span>
                <span class="bar"></span>
              </div>
            </div>

            <div style="margin-bottom: 0.6rem;">
              <span class="badge badge-secondary" style="font-size: 0.78rem;">${item.category}</span>
            </div>

            <p style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.45; margin-bottom: 0.85rem; min-height: 2.6rem;">
              ${item.acoustic_summary}
            </p>
          </div>

          <div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.65rem; font-size: 0.8rem; color: var(--text-muted);">
              <span id="speaker-badge-${item.word}">Speaker #1 of ${item.total_clips}</span>
              <span style="font-weight: 600; color: var(--primary);">16kHz WAV</span>
            </div>

            <div style="display: flex; gap: 0.5rem;">
              <button class="btn-card-play" onclick="window.audioStudio.playWord('${item.word}')" aria-label="Play sound for ${item.display}">
                <span class="play-icon" aria-hidden="true">▶</span>
                <span>Hear Sound</span>
              </button>
              <button class="btn-card-cycle" onclick="window.audioStudio.cycleSpeaker('${item.word}')" title="Hear a different volunteer voice for this word">
                <span>🎲 Next Voice</span>
              </button>
            </div>
          </div>
        </div>
      `).join('')}
    </div>

    <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.5rem; text-align: center;">
      Local Dataset Path: <code>${status.dataset_path}</code> (Excluding background noise folders)
    </div>
  `;

  // Attach canvas to visualizer engine
  const canvasEl = document.getElementById('studio-audio-canvas');
  if (canvasEl && window.audioStudio) {
    window.audioStudio.attachCanvas(canvasEl);
  }
}

function renderStep2(prep) {
  const log = prep.log;
  const container = document.getElementById('step-2-summary');
  if (!container) return;

  container.innerHTML = `
    <div class="grid-3" style="margin-bottom: 1.5rem;">
      <div class="stat-tile">
        <div class="stat-label">Raw Features Extracted</div>
        <div class="stat-value">${log.initial_feature_count}</div>
        <div class="stat-sub">MFCCs, Deltas, Centroid, Bandwidth, RMS</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">Collinear Features Dropped</div>
        <div class="stat-value" style="color: var(--warning);">${log.dropped_collinear_features.length}</div>
        <div class="stat-sub">Pearson correlation |r| > 0.95</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">Standardized Features Retained</div>
        <div class="stat-value" style="color: var(--success);">${log.final_feature_count}</div>
        <div class="stat-sub">StandardScaler (mean=0, variance=1)</div>
      </div>
    </div>

    <div class="explainer-box">
      <h4>🛡️ Ground Truth Isolation Guarantee</h4>
      <p>${log.label_encoding_note}</p>
    </div>
  `;

  // Render Correlation Matrix
  if (log.correlation_matrix) {
    renderCorrelationHeatmap(log.correlation_matrix);
  }
}

function renderStep3(clust) {
  const tableBody = document.getElementById('metrics-table-body');
  if (tableBody) {
    tableBody.innerHTML = clust.comparison_table.map((m) => `
      <tr class="${m.k === window.appState.activeConfig.k && m.distance.startsWith(window.appState.activeConfig.distance) ? 'selected' : ''}">
        <td><strong>K = ${m.k}</strong></td>
        <td>${m.distance}</td>
        <td>${m.wcss.toFixed(1)}</td>
        <td><strong>${m.silhouette.toFixed(4)}</strong></td>
        <td>${m.davies_bouldin.toFixed(4)}</td>
      </tr>
    `).join('');
  }

  // Recommendation Banner
  const sug = clust.suggestion;
  const sugBanner = document.getElementById('recommendation-banner');
  if (sugBanner && sug) {
    sugBanner.innerHTML = `
      <div class="alert" style="background: var(--primary-light); border-left: 4px solid var(--primary); padding: 1.25rem; border-radius: 6px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
          <h4 style="color: var(--primary); font-size: 1.15rem; font-weight: 700;">
            ✨ Recommended Configuration: K = ${sug.recommended_k} (${sug.recommended_distance} Distance)
          </h4>
          <span class="badge badge-primary">Composite Score: ${sug.composite_score.toFixed(3)}</span>
        </div>
        <p style="font-size: 0.95rem; margin-bottom: 0.75rem;">${sug.plain_english_reasoning}</p>
        <div style="font-size: 0.85rem; color: var(--text-secondary); display: flex; gap: 1.5rem;">
          <span><strong>Silhouette:</strong> ${sug.silhouette.toFixed(3)}</span>
          <span><strong>Davies-Bouldin:</strong> ${sug.davies_bouldin.toFixed(3)}</span>
          <span><strong>Inertia/WCSS:</strong> ${sug.wcss.toFixed(1)}</span>
        </div>
      </div>
    `;

    // Sync select dropdowns
    const selectK = document.getElementById('select-k');
    const selectDist = document.getElementById('select-distance');
    if (selectK) selectK.value = sug.recommended_k;
    if (selectDist) selectDist.value = sug.recommended_distance;
  }

  // Render Charts
  renderMetricsCharts(clust);
}

async function changeActiveModel(k, distance) {
  const formData = new FormData();
  formData.append('k', k);
  formData.append('distance', distance);

  try {
    const res = await fetch('/api/select_model', {
      method: 'POST',
      body: formData,
    });
    if (res.ok) {
      const updated = await res.json();
      window.appState.activeConfig = { k, distance };
      window.appState.analysis = updated;
      window.appState.pcaData = updated.pca_data;

      // Re-render
      renderStep3(window.appState.clusteringResults);
      renderStep4(updated);
      renderStep5(updated);
      goToStep(4);
    }
  } catch (err) {
    console.error('Failed to change active model:', err);
  }
}

function renderStep4(anal) {
  // 1. Cluster Profiles
  const profilesContainer = document.getElementById('cluster-profiles-container');
  if (profilesContainer && anal.profiles) {
    profilesContainer.innerHTML = anal.profiles.map(p => `
      <div class="card" style="margin-bottom: 1.25rem;">
        <div class="card-header">
          <div class="card-title">
            <span style="color: ${COLORBLIND_PALETTE[p.cluster_id % COLORBLIND_PALETTE.length]};">●</span>
            Cluster ${p.cluster_id} Profile
          </div>
          <span class="badge badge-primary">${p.percentage}% of Clips (${p.size} samples)</span>
        </div>
        <p style="font-size: 1rem; margin-bottom: 0.75rem;"><strong>Acoustic Impression:</strong> ${p.plain_english_description}</p>
        <div style="margin-bottom: 0.75rem;">
          <strong>Dominant Spoken Words:</strong>
          <div style="display: flex; gap: 0.5rem; margin-top: 0.35rem; flex-wrap: wrap;">
            ${p.word_composition.slice(0, 4).map(w => `
              <span class="badge badge-secondary" style="font-size: 0.85rem; display: inline-flex; align-items: center; gap: 0.35rem;">
                <strong style="text-transform: uppercase;">${w.word}</strong>: ${w.percentage}%
                <button class="sound-preview-btn" onclick="window.audioStudio.playWord('${w.word}')" title="Listen to '${w.word}'">🔊</button>
              </span>
            `).join('')}
          </div>
        </div>
        <details class="collapsible">
          <summary>View Defining Acoustic Features (Z-Scores)</summary>
          <div class="panel-content">
            <div class="table-wrapper">
              <table class="data-table">
                <thead><tr><th>Feature</th><th>Centroid Z-Score</th><th>Cluster Mean</th><th>Global Mean</th></tr></thead>
                <tbody>
                  ${p.top_defining_features.map(f => `
                    <tr>
                      <td><code>${f.feature}</code></td>
                      <td><strong>${f.z_score > 0 ? '+' : ''}${f.z_score}</strong></td>
                      <td>${f.cluster_mean}</td>
                      <td>${f.global_mean}</td>
                    </tr>
                  `).join('')}
                </tbody>
              </table>
            </div>
          </div>
        </details>
      </div>
    `).join('');
  }

  // 2. Confusable Pairs
  const cpContainer = document.getElementById('confusable-pairs-body');
  if (cpContainer && anal.confusable_pairs) {
    cpContainer.innerHTML = anal.confusable_pairs.slice(0, 8).map(cp => `
      <tr style="${cp.is_critical_risk ? 'background: var(--danger-light);' : ''}">
        <td><strong style="text-transform: uppercase;">${cp.word1}</strong> & <strong style="text-transform: uppercase;">${cp.word2}</strong></td>
        <td>
          <button class="pair-compare-badge-btn" onclick="window.audioStudio.playPairBackToBack('${cp.word1}', '${cp.word2}')" title="Play both words sequentially to hear phonetic similarity">
            🔊 Compare Pair
          </button>
        </td>
        <td><strong>${cp.confusability_score.toFixed(3)}</strong></td>
        <td>${(cp.cluster_overlap_sim * 100).toFixed(1)}%</td>
        <td>${cp.acoustic_centroid_distance.toFixed(2)}</td>
        <td>
          <span class="badge ${cp.is_critical_risk ? 'badge-danger' : 'badge-warning'}">
            ${cp.is_critical_risk ? 'CRITICAL SAFETY RISK' : 'Moderate'}
          </span>
          <div style="font-size: 0.8rem; margin-top: 0.25rem;">${cp.critical_warning || 'Phonetically close commands'}</div>
        </td>
      </tr>
    `).join('');
  }

  // 3. Stability Summary
  const stab = anal.stability;
  const stabContainer = document.getElementById('stability-summary');
  if (stabContainer && stab) {
    stabContainer.innerHTML = `
      <div class="stat-tile">
        <div class="stat-label">10-Seed ARI Mean</div>
        <div class="stat-value" style="color: var(--success);">${stab.mean_ari.toFixed(4)}</div>
        <div class="stat-sub">Min: ${stab.min_ari.toFixed(4)}, Std: ${stab.std_ari.toFixed(4)}</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">Cross-Metric Agreement</div>
        <div class="stat-value">${stab.distance_cross_ari_euclidean_vs_manhattan.toFixed(4)}</div>
        <div class="stat-sub">Euclidean vs Manhattan ARI</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">Stability Verdict</div>
        <div class="stat-value" style="font-size: 1.15rem; color: var(--primary);">${stab.stability_verdict.split(':')[0]}</div>
        <div class="stat-sub">${stab.stability_verdict.split(':')[1] || ''}</div>
      </div>
    `;
  }

  // Render Charts
  refreshCurrentCharts();
}

function renderStep5(anal) {
  // Download buttons are already bound to REST API links
}

