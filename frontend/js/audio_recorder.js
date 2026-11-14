/**
 * Robust Web Audio API PCM WAV Recorder & Audio Matcher.
 * Records audio directly as raw PCM samples, downsamples to 16,000 Hz,
 * encodes a 100% compliant 16-bit PCM WAV file in the browser,
 * and handles any uploaded audio format via in-browser decodeAudioData.
 */

// Helper: Encodes Float32Array PCM samples into standard 16-bit PCM RIFF WAV
function encodeWAV(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  /* RIFF identifier */
  writeString(view, 0, 'RIFF');
  /* file length */
  view.setUint32(4, 36 + samples.length * 2, true);
  /* RIFF type */
  writeString(view, 8, 'WAVE');
  /* format chunk identifier */
  writeString(view, 12, 'fmt ');
  /* format chunk length */
  view.setUint32(16, 16, true);
  /* sample format (raw PCM) */
  view.setUint16(20, 1, true);
  /* channel count (mono) */
  view.setUint16(22, 1, true);
  /* sample rate */
  view.setUint32(24, sampleRate, true);
  /* byte rate (sample rate * block align) */
  view.setUint32(28, sampleRate * 2, true);
  /* block align (channel count * bytes per sample) */
  view.setUint16(32, 2, true);
  /* bits per sample */
  view.setUint16(34, 16, true);
  /* data chunk identifier */
  writeString(view, 36, 'data');
  /* data chunk length */
  view.setUint32(40, samples.length * 2, true);

  // Write 16-bit signed PCM samples with clipping protection
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([view], { type: 'audio/wav' });
}

// Helper: Downsample Float32Array buffer to target sample rate (16kHz)
function downsampleBuffer(buffer, inputSampleRate, outputSampleRate = 16000) {
  if (inputSampleRate === outputSampleRate) {
    return buffer;
  }
  const sampleRateRatio = inputSampleRate / outputSampleRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? accum / count : buffer[offsetBuffer];
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

class AudioProcessor {
  constructor() {
    this.audioContext = null;
    this.mediaStream = null;
    this.scriptProcessor = null;
    this.recordedBuffers = [];
    this.isRecording = false;
    this.recordTimeout = null;

    this.initElements();
    this.bindEvents();
  }

  initElements() {
    this.recordBtn = document.getElementById('btn-record');
    this.recordStatus = document.getElementById('record-status');
    this.audioPlayer = document.getElementById('audio-preview');
    this.fileInput = document.getElementById('audio-file-input');
    this.uploadZone = document.getElementById('audio-drop-zone');
    this.tryItResult = document.getElementById('try-it-results');
    this.recorderBox = document.getElementById('recorder-box');
  }

  bindEvents() {
    if (this.recordBtn) {
      this.recordBtn.addEventListener('click', () => this.toggleRecording());
    }

    if (this.fileInput) {
      this.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
          this.handleAudioFile(e.target.files[0]);
        }
      });
    }

    if (this.uploadZone) {
      this.uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        this.uploadZone.style.borderColor = 'var(--primary)';
      });
      this.uploadZone.addEventListener('dragleave', () => {
        this.uploadZone.style.borderColor = 'var(--border)';
      });
      this.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        this.uploadZone.style.borderColor = 'var(--border)';
        if (e.dataTransfer.files.length > 0) {
          this.handleAudioFile(e.dataTransfer.files[0]);
        }
      });
    }
  }

  async toggleRecording() {
    if (this.isRecording) {
      this.stopRecording();
    } else {
      await this.startRecording();
    }
  }

  async startRecording() {
    try {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }

      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: true,
        },
        video: false,
      });

      const source = this.audioContext.createMediaStreamSource(this.mediaStream);
      // ScriptProcessor captures raw PCM float32 buffers
      this.scriptProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);
      this.recordedBuffers = [];

      this.scriptProcessor.onaudioprocess = (e) => {
        if (!this.isRecording) return;
        const channelData = e.inputBuffer.getChannelData(0);
        this.recordedBuffers.push(new Float32Array(channelData));
      };

      source.connect(this.scriptProcessor);
      this.scriptProcessor.connect(this.audioContext.destination);

      // Connect live mic analyser for visual feedback
      this.micAnalyser = this.audioContext.createAnalyser();
      this.micAnalyser.fftSize = 32;
      this.micFreqData = new Uint8Array(this.micAnalyser.frequencyBinCount);
      source.connect(this.micAnalyser);

      const vuContainer = document.getElementById('mic-live-vu');
      if (vuContainer) vuContainer.style.display = 'flex';

      const vuBars = vuContainer ? vuContainer.querySelectorAll('.vu-bar') : [];
      const updateVU = () => {
        if (!this.isRecording) return;
        this.micAnalyser.getByteFrequencyData(this.micFreqData);
        for (let i = 0; i < vuBars.length; i++) {
          const val = this.micFreqData[i % this.micFreqData.length] / 255.0;
          const h = Math.max(6, Math.round(val * 34));
          vuBars[i].style.height = `${h}px`;
        }
        this.vuAnimationId = requestAnimationFrame(updateVU);
      };
      this.vuAnimationId = requestAnimationFrame(updateVU);

      this.isRecording = true;
      this.recorderBox.classList.add('recording');
      this.recordBtn.innerHTML = '⏹ Stop Recording';
      this.recordBtn.className = 'btn btn-danger';
      this.recordStatus.innerHTML = '<span style="color: #dc2626; font-weight: 700;">● RECORDING...</span> Speak clearly now: say "STOP", "GO", "YES", or "NO"!';

      // Record for 1.3 seconds (ample time for 1-second spoken command)
      this.recordTimeout = setTimeout(() => {
        if (this.isRecording) {
          this.stopRecording();
        }
      }, 1300);

    } catch (err) {
      console.error('Microphone access error:', err);
      this.recordStatus.innerHTML = `<span style="color: #dc2626;">Microphone access error: ${err.message}. You can still upload any audio file below!</span>`;
    }
  }

  stopRecording() {
    if (this.recordTimeout) clearTimeout(this.recordTimeout);
    if (!this.isRecording) return;

    if (this.vuAnimationId) {
      cancelAnimationFrame(this.vuAnimationId);
      this.vuAnimationId = null;
    }
    const vuContainer = document.getElementById('mic-live-vu');
    if (vuContainer) vuContainer.style.display = 'none';

    this.isRecording = false;
    this.recorderBox.classList.remove('recording');
    this.recordBtn.innerHTML = '🎤 Record Voice Clip';
    this.recordBtn.className = 'btn btn-primary';
    this.recordStatus.innerHTML = '<span class="spinner"></span> Processing your audio recording...';

    // Stop microphone tracks
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
    }

    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
    }

    // Merge buffers into a single Float32Array
    const totalLength = this.recordedBuffers.reduce((acc, b) => acc + b.length, 0);
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const buf of this.recordedBuffers) {
      merged.set(buf, offset);
      offset += buf.length;
    }

    const currentSr = this.audioContext ? this.audioContext.sampleRate : 44100;

    // Downsample to 16,000 Hz
    const resampled = downsampleBuffer(merged, currentSr, 16000);

    // Normalize amplitude to prevent quiet mic issues
    let maxAmp = 0;
    for (let i = 0; i < resampled.length; i++) {
      const absVal = Math.abs(resampled[i]);
      if (absVal > maxAmp) maxAmp = absVal;
    }
    if (maxAmp > 0.001) {
      const gain = 0.92 / maxAmp;
      for (let i = 0; i < resampled.length; i++) {
        resampled[i] *= gain;
      }
    }

    // Encode to real PCM WAV
    const wavBlob = encodeWAV(resampled, 16000);

    // Display audio player preview
    const audioUrl = URL.createObjectURL(wavBlob);
    this.audioPlayer.src = audioUrl;
    this.audioPlayer.style.display = 'block';

    // Submit to backend
    this.submitAudio(wavBlob);
  }

  async handleAudioFile(file) {
    this.recordStatus.innerHTML = `<span class="spinner"></span> Loading and decoding ${file.name}...`;

    try {
      // Decode ANY audio file (WAV, MP3, WebM, OGG, M4A) via in-browser AudioContext
      const arrayBuffer = await file.arrayBuffer();
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

      // Extract mono channel
      const channelData = audioBuffer.getChannelData(0);

      // Downsample to 16,000 Hz
      const resampled = downsampleBuffer(channelData, audioBuffer.sampleRate, 16000);

      // Normalize amplitude
      let maxAmp = 0;
      for (let i = 0; i < resampled.length; i++) {
        const absVal = Math.abs(resampled[i]);
        if (absVal > maxAmp) maxAmp = absVal;
      }
      if (maxAmp > 0.001) {
        const gain = 0.92 / maxAmp;
        for (let i = 0; i < resampled.length; i++) {
          resampled[i] *= gain;
        }
      }

      // Encode to standard 16kHz PCM WAV
      const wavBlob = encodeWAV(resampled, 16000);

      this.audioPlayer.src = URL.createObjectURL(wavBlob);
      this.audioPlayer.style.display = 'block';
      this.recordStatus.textContent = `Loaded and converted: ${file.name} (16kHz mono WAV)`;

      this.submitAudio(wavBlob);

    } catch (err) {
      console.error('Audio decode error:', err);
      // Fallback: send raw file directly if decodeAudioData fails
      this.audioPlayer.src = URL.createObjectURL(file);
      this.audioPlayer.style.display = 'block';
      this.submitAudio(file);
    }
  }

  async submitAudio(blob) {
    this.recordStatus.innerHTML = '<span class="spinner"></span> Extracting 45 acoustic features & identifying closest command words...';
    const formData = new FormData();
    formData.append('audio_file', blob, 'user_clip.wav');

    try {
      const response = await fetch('/api/try_it', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Failed to process audio');
      }

      const result = await response.json();
      this.renderResults(result);
      this.recordStatus.innerHTML = '<strong>✓ Recognition complete!</strong> See your acoustic analysis and matching breakdown below:';

      // Overlay user point onto PCA 2D scatter chart
      if (window.appState && window.appState.pcaData && window.appState.analysis) {
        renderPcaScatter(
          window.appState.pcaData,
          window.appState.analysis.labels,
          window.appState.analysis.words,
          window.appState.pcaColorMode || 'cluster',
          result.pca_coords
        );
      }

    } catch (err) {
      console.error(err);
      this.recordStatus.innerHTML = `<span style="color: #dc2626;">Error: ${err.message}</span>`;
    }
  }

  renderResults(result) {
    this.tryItResult.style.display = 'block';

    const simRows = result.word_similarities.map((ws, i) => {
      const isTop = i === 0;
      const barColor = isTop ? 'var(--success)' : (ws.similarity_percentage > 50 ? 'var(--primary)' : 'var(--accent)');
      return `
        <tr style="${isTop ? 'background: var(--success-light); font-weight: 600;' : ''}">
          <td><strong>#${i + 1}</strong></td>
          <td>
            <span style="font-size: 1.05rem; text-transform: uppercase; font-weight: 700; color: ${isTop ? 'var(--success)' : 'inherit'};">
              ${ws.word}
            </span>
            ${isTop ? ' <span class="badge badge-success" style="margin-left: 0.5rem;">TOP RECOGNITION MATCH</span>' : ''}
          </td>
          <td style="width: 45%;">
            <div style="background: var(--bg-tertiary); border-radius: 6px; overflow: hidden; height: 18px; width: 100%;">
              <div style="background: ${barColor}; width: ${ws.similarity_percentage}%; height: 100%; transition: width 0.5s ease-in-out;"></div>
            </div>
          </td>
          <td>
            <strong>${ws.similarity_percentage}%</strong>
            <span style="font-size: 0.8rem; color: var(--text-muted); margin-left: 0.4rem;">(dist: ${ws.acoustic_distance})</span>
          </td>
        </tr>
      `;
    }).join('');

    this.tryItResult.innerHTML = `
      <div class="card" style="margin-top: 1.5rem; border: 2px solid var(--primary); box-shadow: var(--shadow-lg);">
        <div class="card-header">
          <div class="card-title">
            <span>🎯</span> Voice Recognition & Acoustic Match Result
          </div>
          <span class="badge badge-primary" style="font-size: 0.9rem; padding: 0.4rem 0.8rem;">Assigned: Cluster ${result.predicted_cluster}</span>
        </div>
        
        <div class="grid-3" style="margin-bottom: 1.5rem;">
          <div class="stat-tile" style="border-left: 4px solid var(--success); background: var(--bg-card);">
            <div class="stat-label">Recognized Command Word</div>
            <div class="stat-value" style="color: var(--success); text-transform: uppercase; font-size: 2.2rem;">"${result.top_match_word}"</div>
            <div class="stat-sub">Confidence: <strong>${result.top_match_similarity}% similarity match</strong></div>
          </div>
          <div class="stat-tile" style="border-left: 4px solid var(--primary); background: var(--bg-card);">
            <div class="stat-label">Acoustic Cluster</div>
            <div class="stat-value" style="color: var(--primary);">Cluster ${result.predicted_cluster}</div>
            <div class="stat-sub">Distance to cluster center: ${result.distance_to_cluster_center}</div>
          </div>
          <div class="stat-tile" style="border-left: 4px solid var(--accent); background: var(--bg-card);">
            <div class="stat-label">Audio Normalization</div>
            <div class="stat-value" style="font-size: 1.3rem;">${result.status.toUpperCase()}</div>
            <div class="stat-sub">Speech centered @ 16kHz mono</div>
          </div>
        </div>

        <div style="background: var(--bg-secondary); border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">
          <h4 style="font-size: 0.95rem; margin-bottom: 0.5rem; color: var(--text-primary);">Acoustic Dimensions Detected in Your Voice:</h4>
          <div class="grid-4" style="font-size: 0.85rem;">
            <div><strong>Spectral Centroid:</strong> ${result.raw_features_summary.spectral_centroid} Hz</div>
            <div><strong>Zero-Crossing Rate:</strong> ${result.raw_features_summary.zero_crossing_rate}</div>
            <div><strong>RMS Energy:</strong> ${result.raw_features_summary.rms_energy}</div>
            <div><strong>Spoken Duration:</strong> ${result.raw_features_summary.active_duration} s</div>
          </div>
        </div>

        <h4 style="margin-bottom: 0.75rem; font-size: 1.05rem;">Acoustic Similarity Breakdown Across Core Commands:</h4>
        <div class="table-wrapper">
          <table class="data-table">
            <thead>
              <tr><th>Rank</th><th>Command Word</th><th>Acoustic Similarity Match</th><th>Score</th></tr>
            </thead>
            <tbody>
              ${simRows}
            </tbody>
          </table>
        </div>

        <div class="explainer-box" style="margin-top: 1.25rem;">
          <h4>ℹ️ Exploratory Voice Control Disclaimer</h4>
          <p>${result.exploratory_disclaimer}</p>
        </div>
      </div>
    `;

    // Smooth scroll down to the result card so the user sees it immediately
    this.tryItResult.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// Instantiate on DOM load
window.addEventListener('DOMContentLoaded', () => {
  window.audioProcessor = new AudioProcessor();
});
