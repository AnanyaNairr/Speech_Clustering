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

  // PCA Point Inspector Close
  const closeInspBtn = document.getElementById('btn-close-pca-insp');
  if (closeInspBtn) {
    closeInspBtn.addEventListener('click', () => {
      const inspector = document.getElementById('pca-point-inspector');
      if (inspector) inspector.style.display = 'none';
      if (window.audioManager) window.audioManager.stopAll();
    });
  }

  // Stop Confusable Comparison
  const stopCompBtn = document.getElementById('btn-stop-comparison');
  if (stopCompBtn) {
    stopCompBtn.addEventListener('click', () => {
      if (window.audioManager) window.audioManager.stopAll();
    });
  }
}

function goToStep(step) {
  if (step < 1 || step > 5) return;
  if (window.audioManager) window.audioManager.stopAll();
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

    // Load Preprocessing
    const prepRes = await fetch('/api/preprocessing');
    if (prepRes.ok) {
      const prep = await prepRes.json();
      window.appState.prepData = prep;
      renderStep1(status, prep);
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

    // Load Audio Vocabulary Metadata
    window.audioManager = new VocabularyAudioManager();
    await window.audioManager.loadMetadata();
    if (window.appState.prepData && window.appState.datasetStatus) {
      renderStep1(window.appState.datasetStatus, window.appState.prepData);
    }

  } catch (err) {
    console.error('Failed to load initial pipeline status:', err);
  }
}

/**
 * --------------------------------------------------------------------------
 * VOCABULARY AUDIO MANAGER & LIVE VISUALIZERS
 * --------------------------------------------------------------------------
 */
class VocabularyAudioManager {
  constructor() {
    this.audio = null;
    this.activeWord = null;
    this.activeFilename = null;
    this.speakerIndices = {};
    this.metadata = {};
    this.isTourPlaying = false;
    this.tourIndex = 0;
    this.tourTimeout = null;
    this.animFrame = null;
  }

  async loadMetadata() {
    try {
      const res = await fetch('/api/audio/words_metadata');
      if (res.ok) {
        const data = await res.json();
        this.metadata = data.words || {};
      }
    } catch (err) {
      console.warn('Could not fetch words metadata:', err);
    }
  }

  stopAll() {
    this.isTourPlaying = false;
    if (this.tourTimeout) clearTimeout(this.tourTimeout);
    if (this.audio) {
      this.audio.pause();
      this.audio.currentTime = 0;
      this.audio = null;
    }
    if (this.animFrame) cancelAnimationFrame(this.animFrame);
    this.clearPlayingCards();
    this.updateHud(null);
    this.resetCompareButtons();

    const canvas = document.getElementById('hud-visualizer-canvas');
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    const pcaBtn = document.getElementById('btn-play-pca-sample');
    if (pcaBtn) pcaBtn.innerHTML = '▶ Play This Clip';
  }

  clearPlayingCards() {
    document.querySelectorAll('.vocab-card').forEach(c => {
      c.classList.remove('is-playing');
      const icon = c.querySelector('.play-icon');
      const label = c.querySelector('.play-label');
      if (icon) icon.textContent = '▶';
      if (label) label.textContent = 'Listen';
    });
    this.activeWord = null;
  }

  resetCompareButtons() {
    document.querySelectorAll('.btn-compare-pair').forEach(b => {
      b.classList.remove('is-playing');
      b.innerHTML = '🔊 Compare A vs B';
    });
    const banner = document.getElementById('confusable-audio-banner');
    if (banner) banner.style.display = 'none';
  }

  async playWord(word, customIndex = null) {
    if (this.activeWord === word && this.audio && !this.audio.paused && customIndex === null) {
      this.stopAll();
      return;
    }

    this.stopAll();

    const meta = this.metadata[word] || {};
    const sampleFiles = meta.sample_files || [];
    let idx = customIndex !== null ? customIndex : (this.speakerIndices[word] || 0);
    if (sampleFiles.length > 0) {
      idx = idx % sampleFiles.length;
    }
    this.speakerIndices[word] = idx;

    const audioUrl = `/api/audio/sample/${word}?index=${idx}&t=${Date.now()}`;
    const audio = new Audio(audioUrl);
    this.audio = audio;
    this.activeWord = word;

    const card = document.getElementById(`vocab-card-${word}`);
    if (card) {
      card.classList.add('is-playing');
      const icon = card.querySelector('.play-icon');
      const label = card.querySelector('.play-label');
      if (icon) icon.textContent = '⏸';
      if (label) label.textContent = 'Playing';
    }

    const currentFilename = sampleFiles[idx] || '';
    const speakerId = currentFilename.split('_')[0] || (idx + 1);
    this.updateHud({
      word: word.toUpperCase(),
      ipa: meta.ipa || '',
      role: meta.accessibility_role || 'Command',
      speaker: speakerId,
    });

    this.initCanvasAnimation();

    audio.onended = () => {
      if (card) {
        card.classList.remove('is-playing');
        const icon = card.querySelector('.play-icon');
        const label = card.querySelector('.play-label');
        if (icon) icon.textContent = '▶';
        if (label) label.textContent = 'Listen';
      }
      this.updateHud(null);
      this.activeWord = null;
    };

    audio.onerror = (e) => {
      console.error('Audio playback error:', e);
      this.clearPlayingCards();
      this.updateHud(null);
    };

    try {
      await audio.play();
    } catch (e) {
      console.warn('Playback prevented or aborted:', e);
    }
  }

  shuffleSpeaker(word) {
    const meta = this.metadata[word] || {};
    const sampleFiles = meta.sample_files || [];
    const count = sampleFiles.length > 0 ? sampleFiles.length : 10;
    const nextIdx = ((this.speakerIndices[word] || 0) + 1) % count;
    this.speakerIndices[word] = nextIdx;

    const speakerLabel = document.getElementById(`speaker-label-${word}`);
    const nextSpeaker = (sampleFiles[nextIdx] || '').split('_')[0] || (nextIdx + 1);
    if (speakerLabel) {
      speakerLabel.textContent = `#${nextSpeaker}`;
    }

    this.playWord(word, nextIdx);
  }

  async startTour() {
    this.stopAll();
    const words = window.appState.datasetStatus?.core_words || [
      'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'
    ];
    this.isTourPlaying = true;
    this.tourIndex = 0;

    const playNext = async () => {
      if (!this.isTourPlaying || this.tourIndex >= words.length) {
        this.stopAll();
        return;
      }
      const w = words[this.tourIndex];
      this.tourIndex++;
      await this.playWord(w);

      if (this.audio) {
        this.audio.onended = () => {
          this.clearPlayingCards();
          this.tourTimeout = setTimeout(() => {
            playNext();
          }, 350);
        };
      } else {
        playNext();
      }
    };

    await playNext();
  }

  async comparePair(word1, word2, btnElement = null) {
    this.stopAll();
    const banner = document.getElementById('confusable-audio-banner');
    const title = document.getElementById('confusable-banner-title');
    const desc = document.getElementById('confusable-banner-desc');

    if (banner) {
      banner.style.display = 'flex';
      if (title) title.innerHTML = `Comparing <strong style="color:var(--primary); font-size:1.1rem;">${word1.toUpperCase()}</strong> vs <strong style="color:var(--accent); font-size:1.1rem;">${word2.toUpperCase()}</strong>`;
      if (desc) desc.textContent = `Auditioning sequential clips to demonstrate acoustic similarity...`;
    }

    if (btnElement) {
      btnElement.classList.add('is-playing');
      btnElement.innerHTML = '🔊 Playing Pair...';
    }

    // Play Word 1
    const audio1 = new Audio(`/api/audio/sample/${word1}?t=${Date.now()}`);
    this.audio = audio1;
    this.updateHud({
      word: word1.toUpperCase(),
      ipa: (this.metadata[word1] || {}).ipa || '',
      role: 'Hazard Pair A',
      speaker: 'Sample A',
    });
    this.initCanvasAnimation();

    audio1.onended = () => {
      if (desc) desc.innerHTML = `<span style="color:var(--primary); font-weight:700;">● Played ${word1.toUpperCase()}</span> — now playing ${word2.toUpperCase()}...`;
      this.tourTimeout = setTimeout(() => {
        // Play Word 2
        const audio2 = new Audio(`/api/audio/sample/${word2}?t=${Date.now()}`);
        this.audio = audio2;
        this.updateHud({
          word: word2.toUpperCase(),
          ipa: (this.metadata[word2] || {}).ipa || '',
          role: 'Hazard Pair B',
          speaker: 'Sample B',
        });
        this.initCanvasAnimation();

        audio2.onended = () => {
          if (desc) desc.innerHTML = `✓ Finished comparison. Notice vowel resonances and consonant releases!`;
          if (btnElement) {
            btnElement.classList.remove('is-playing');
            btnElement.innerHTML = '🔊 Compare A vs B';
          }
          this.updateHud(null);
        };
        audio2.play();
      }, 400);
    };

    try {
      await audio1.play();
    } catch (e) {
      console.warn('Pair comparison playback error:', e);
    }
  }

  async playClip(word, filename) {
    if (this.audio && !this.audio.paused && this.activeFilename === filename) {
      this.stopAll();
      return;
    }

    this.stopAll();
    const url = `/api/audio/clip?word=${encodeURIComponent(word)}&file_name=${encodeURIComponent(filename)}`;
    const audio = new Audio(url);
    this.audio = audio;
    this.activeWord = word;
    this.activeFilename = filename;

    this.updateHud({
      word: word.toUpperCase(),
      ipa: (this.metadata[word] || {}).ipa || '',
      role: 'PCA Point Exemplar',
      speaker: filename.split('_')[0],
    });

    const playBtn = document.getElementById('btn-play-pca-sample');
    if (playBtn) playBtn.innerHTML = '⏸ Pause Clip';

    this.initCanvasAnimation();

    audio.onended = () => {
      if (playBtn) playBtn.innerHTML = '▶ Play This Clip';
      this.updateHud(null);
      this.activeFilename = null;
    };

    try {
      await audio.play();
    } catch (e) {
      console.warn('Clip playback error:', e);
    }
  }

  updateHud(info) {
    const dot = document.getElementById('hud-status-dot');
    const label = document.getElementById('hud-now-playing');
    if (!label) return;

    if (info) {
      if (dot) dot.classList.add('active');
      label.innerHTML = `Now Playing: <strong style="color: var(--primary);">${info.word}</strong> <span style="font-family: monospace; font-size: 0.9rem; color: var(--accent);">${info.ipa}</span> • Speaker <strong style="font-family: monospace;">#${info.speaker}</strong> <span class="badge badge-secondary" style="font-size:0.75rem;">${info.role}</span>`;
    } else {
      if (dot) dot.classList.remove('active');
      label.innerHTML = `Click any command card below to hear real voice clips from the dataset`;
    }
  }

  initCanvasAnimation() {
    const canvas = document.getElementById('hud-visualizer-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let phase = 0;

    const render = () => {
      if (!this.audio || this.audio.paused) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
      }
      this.animFrame = requestAnimationFrame(render);
      phase += 0.2;
      ctx.fillStyle = document.documentElement.getAttribute('data-theme') === 'dark' ? '#1a2338' : '#f1f5f9';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#38bdf8';
      ctx.shadowBlur = 6;
      ctx.shadowColor = '#0284c7';
      ctx.beginPath();
      const numPoints = 28;
      const step = canvas.width / numPoints;
      for (let i = 0; i <= numPoints; i++) {
        const x = i * step;
        const envelope = Math.sin((i / numPoints) * Math.PI);
        const y = (canvas.height / 2) + Math.sin(i * 0.5 + phase) * (canvas.height / 2.6) * envelope;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    };
    render();
  }
}

// Global Callback when user clicks a point on the PCA scatter plot
window.onPcaPointClicked = function(pointData, x, y) {
  const inspector = document.getElementById('pca-point-inspector');
  const wordBadge = document.getElementById('pca-insp-word');
  const title = document.getElementById('pca-insp-title');
  const details = document.getElementById('pca-insp-details');
  const playBtn = document.getElementById('btn-play-pca-sample');

  if (!inspector) return;
  inspector.style.display = 'flex';

  const word = pointData.word;
  const cluster = pointData.cluster;
  const idx = pointData.idx;
  const filename = window.appState.analysis?.file_names?.[idx] || `${word}_sample_${idx}.wav`;
  const speakerId = filename.split('_')[0];

  if (wordBadge) wordBadge.textContent = word.toUpperCase();
  if (title) title.innerHTML = `Speech Command Sample: "<strong>${word.toUpperCase()}</strong>"`;
  if (details) {
    details.innerHTML = `Cluster <strong>${cluster}</strong> • File: <code>${filename}</code> • Speaker: <code>#${speakerId}</code> • Coordinates: PC1=<strong>${x.toFixed(2)}</strong>, PC2=<strong>${y.toFixed(2)}</strong>`;
  }

  if (playBtn) {
    playBtn.innerHTML = '▶ Play This Clip';
    playBtn.onclick = () => {
      window.audioManager.playClip(word, filename);
    };
  }

  inspector.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
};

function renderStep1(status, prep) {
  const container = document.getElementById('step-1-content');
  if (!container) return;

  const totalClips = prep.log.initial_sample_count || 2000;
  const wordClasses = prep.log.word_classes || status.core_words;
  const meta = window.audioManager?.metadata || {};

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
        <div class="stat-sub">200 clips per command</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">Clip Duration</div>
        <div class="stat-value">1.00s</div>
        <div class="stat-sub">16,000 samples @ 16kHz mono</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">Audio Integrations</div>
        <div class="stat-value" style="color: var(--success);">Clean</div>
        <div class="stat-sub">Padded/Trimmed seamlessly</div>
      </div>
    </div>

    <!-- Live Audio Master HUD -->
    <div class="audio-hud-bar" role="region" aria-label="Audio Audition HUD">
      <div class="hud-info">
        <div id="hud-status-dot" class="hud-status-dot" aria-hidden="true"></div>
        <div>
          <div style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.05em;">
            Interactive Speech Command Deck
          </div>
          <div id="hud-now-playing" style="font-size: 1.05rem; font-weight: 700; color: var(--text-primary);">
            Click any command card below to hear real voice clips from the dataset
          </div>
        </div>
      </div>
      <div style="display: flex; align-items: center; gap: 0.85rem; flex-wrap: wrap;">
        <canvas id="hud-visualizer-canvas" class="hud-visualizer-canvas" width="220" height="38" aria-label="Audio Waveform Display"></canvas>
        <button id="btn-play-tour" class="btn btn-primary" onclick="window.audioManager.startTour()" style="font-size: 0.85rem; padding: 0.45rem 1rem;">
          ▶ Play All (Tour)
        </button>
        <button id="btn-stop-audio" class="btn btn-secondary" onclick="window.audioManager.stopAll()" style="font-size: 0.85rem; padding: 0.45rem 0.85rem;">
          ⏹ Stop
        </button>
      </div>
    </div>

    <!-- 10 Interactive Audio Word Cards -->
    <div class="vocab-grid">
      ${wordClasses.map(w => {
        const m = meta[w] || {};
        const ipa = m.ipa || '';
        const role = m.accessibility_role || 'Navigation';
        const desc = m.sound_desc || 'Acoustic voice command';
        const sampleFiles = m.sample_files || [];
        const initialSpeaker = sampleFiles.length > 0 ? sampleFiles[0].split('_')[0] : 'default';

        return `
          <div class="vocab-card" id="vocab-card-${w}" data-word="${w}">
            <div>
              <div class="vocab-card-header">
                <div>
                  <span class="vocab-category-tag">${role}</span>
                  <div class="vocab-word-title">${w}</div>
                </div>
                <span class="vocab-ipa" title="IPA phonetic transcription">${ipa}</span>
              </div>
              <p class="vocab-sound-desc">${desc}</p>
            </div>

            <div>
              <div style="font-size: 0.78rem; color: var(--text-muted); margin-bottom: 0.65rem;">
                Speaker: <strong id="speaker-label-${w}" style="font-family: monospace;">#${initialSpeaker}</strong> 
                <span style="opacity: 0.75;">(200 clips)</span>
              </div>
              <div class="vocab-footer">
                <button class="btn-play-word" onclick="window.audioManager.playWord('${w}')" aria-label="Listen to audio of word ${w}">
                  <span class="play-icon" aria-hidden="true">▶</span>
                  <span class="play-label">Listen</span>
                  <div class="eq-bars" aria-hidden="true">
                    <span class="eq-bar"></span>
                    <span class="eq-bar"></span>
                    <span class="eq-bar"></span>
                    <span class="eq-bar"></span>
                  </div>
                </button>
                <button class="btn-shuffle-voice" onclick="window.audioManager.shuffleSpeaker('${w}')" title="Audition another volunteer speaker for ${w}" aria-label="Audition another speaker for ${w}">
                  🔀 Next Voice
                </button>
              </div>
            </div>
          </div>
        `;
      }).join('')}
    </div>

    <div style="margin-top: 1.25rem; font-size: 0.85rem; color: var(--text-muted); text-align: center;">
      📁 Dataset Directory: <code>${status.dataset_path}</code> • 200 random clips per word sampled with seed 42
    </div>
  `;
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
              <span class="badge badge-secondary" style="font-size: 0.85rem;">
                <strong style="text-transform: uppercase;">${w.word}</strong>: ${w.percentage}%
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
        <td><strong>${cp.confusability_score.toFixed(3)}</strong></td>
        <td>${(cp.cluster_overlap_sim * 100).toFixed(1)}%</td>
        <td>${cp.acoustic_centroid_distance.toFixed(2)}</td>
        <td>
          <span class="badge ${cp.is_critical_risk ? 'badge-danger' : 'badge-warning'}">
            ${cp.is_critical_risk ? 'CRITICAL SAFETY RISK' : 'Moderate'}
          </span>
          <div style="font-size: 0.8rem; margin-top: 0.25rem;">${cp.critical_warning || 'Phonetically close commands'}</div>
        </td>
        <td style="text-align: center;">
          <button class="btn-compare-pair" onclick="window.audioManager.comparePair('${cp.word1}', '${cp.word2}', this)" title="Listen to ${cp.word1.toUpperCase()} vs ${cp.word2.toUpperCase()} sequential playback" aria-label="Audition comparison between ${cp.word1} and ${cp.word2}">
            🔊 Compare A vs B
          </button>
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

