/**
 * Acoustic Command Studio & Live Audio Visualizer.
 * Provides real-time Web Audio API waveform and frequency spectrum visualization,
 * interactive sound cards for the 10 core accessibility commands,
 * volunteer speaker cycling, back-to-back confusable pair comparison,
 * and live audio effects.
 */

class AudioStudio {
  constructor() {
    this.wordsInfo = [];
    this.audioElement = new Audio();
    this.audioElement.crossOrigin = 'anonymous';
    this.audioContext = null;
    this.analyser = null;
    this.sourceNode = null;
    this.dataArray = null;
    this.freqArray = null;
    this.animationFrameId = null;

    // State
    this.currentPlayingWord = null;
    this.currentSpeakerIndices = {}; // word -> speakerIndex
    this.isPlaying = false;
    this.playbackSpeed = 1.0;
    this.pairSequenceTimer = null;

    // Elements
    this.canvas = null;
    this.canvasCtx = null;
    this.nowPlayingWordEl = null;
    this.nowPlayingInfoEl = null;
    this.masterPlayBtn = null;

    this.initAudioListeners();
  }

  initAudioContext() {
    if (this.audioContext) {
      if (this.audioContext.state === 'suspended') {
        this.audioContext.resume().catch(() => {});
      }
      return;
    }
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioCtx();
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.8;

      const bufferLength = this.analyser.frequencyBinCount;
      this.dataArray = new Uint8Array(bufferLength);
      this.freqArray = new Uint8Array(bufferLength);

      if (!this.sourceNode) {
        this.sourceNode = this.audioContext.createMediaElementSource(this.audioElement);
        this.sourceNode.connect(this.analyser);
        this.analyser.connect(this.audioContext.destination);
      }
    } catch (e) {
      console.warn('Web Audio API context could not be connected directly to Audio element:', e);
    }
  }

  initAudioListeners() {
    this.audioElement.addEventListener('play', () => {
      this.isPlaying = true;
      this.updateUIPlayStates();
      this.startVisualizer();
    });

    this.audioElement.addEventListener('pause', () => {
      this.isPlaying = false;
      this.updateUIPlayStates();
    });

    this.audioElement.addEventListener('ended', () => {
      this.isPlaying = false;
      this.updateUIPlayStates();
    });

    this.audioElement.addEventListener('error', (e) => {
      console.error('Audio playback error:', e);
      this.isPlaying = false;
      this.updateUIPlayStates();
      this.showToast('Could not load audio sample. Please verify backend is running.', 'error');
    });
  }

  async loadWordsMetadata() {
    try {
      const res = await fetch('/api/audio/words_info');
      if (res.ok) {
        const data = await res.json();
        this.wordsInfo = data.words || [];
        this.wordsInfo.forEach(w => {
          this.currentSpeakerIndices[w.word] = 0;
        });
        return this.wordsInfo;
      }
    } catch (err) {
      console.error('Failed to load words metadata:', err);
    }
    return [];
  }

  attachCanvas(canvasElement) {
    if (!canvasElement) return;
    this.canvas = canvasElement;
    this.canvasCtx = canvasElement.getContext('2d');
    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
    this.drawIdleWaveform();
  }

  resizeCanvas() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(300, Math.floor(rect.width * dpr));
    this.canvas.height = Math.max(100, Math.floor(rect.height * dpr));
    if (this.canvasCtx) {
      this.canvasCtx.scale(dpr, dpr);
    }
  }

  drawIdleWaveform() {
    if (!this.canvas || !this.canvasCtx || this.isPlaying) return;
    const ctx = this.canvasCtx;
    const rect = this.canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    ctx.clearRect(0, 0, width, height);

    // Draw gentle animated resting wave
    const time = Date.now() * 0.002;
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';

    // Ambient background grid lines
    ctx.strokeStyle = isDark ? 'rgba(255, 255, 255, 0.04)' : 'rgba(0, 0, 0, 0.04)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Ambient floating sine ribbon
    ctx.beginPath();
    ctx.lineWidth = 2;
    const gradient = ctx.createLinearGradient(0, 0, width, 0);
    if (isDark) {
      gradient.addColorStop(0, 'rgba(56, 189, 248, 0.2)');
      gradient.addColorStop(0.5, 'rgba(99, 102, 241, 0.6)');
      gradient.addColorStop(1, 'rgba(168, 85, 247, 0.2)');
    } else {
      gradient.addColorStop(0, 'rgba(37, 99, 235, 0.2)');
      gradient.addColorStop(0.5, 'rgba(2, 132, 199, 0.6)');
      gradient.addColorStop(1, 'rgba(79, 70, 229, 0.2)');
    }
    ctx.strokeStyle = gradient;

    for (let x = 0; x < width; x += 3) {
      const y = height / 2 + Math.sin(x * 0.02 + time) * 8 * Math.sin(x * 0.005 + time * 0.5);
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    if (!this.isPlaying) {
      this.animationFrameId = requestAnimationFrame(() => this.drawIdleWaveform());
    }
  }

  startVisualizer() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    this.renderVisualizerFrame();
  }

  renderVisualizerFrame() {
    if (!this.canvas || !this.canvasCtx) return;
    const ctx = this.canvasCtx;
    const rect = this.canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    ctx.clearRect(0, 0, width, height);

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';

    if (this.isPlaying) {
      let hasSignal = false;
      if (this.analyser) {
        this.analyser.getByteFrequencyData(this.freqArray);
        this.analyser.getByteTimeDomainData(this.dataArray);
        for (let i = 0; i < Math.min(30, this.freqArray.length); i++) {
          if (this.freqArray[i] > 5) { hasSignal = true; break; }
        }
      }

      // If playing via speech synthesis or before audio stream fills, simulate lively dynamic frequencies
      if (!hasSignal) {
        const t = Date.now() * 0.008;
        for (let i = 0; i < this.freqArray.length; i++) {
          this.freqArray[i] = Math.floor(Math.abs(Math.sin(t + i * 0.25) * Math.cos(t * 0.6 + i * 0.15)) * 190 + 35);
          this.dataArray[i] = Math.floor(128 + Math.sin(t * 2.5 + i * 0.2) * 40 * Math.cos(t + i * 0.05));
        }
      }

      const bufferLength = this.analyser ? this.analyser.frequencyBinCount : 128;

      // 1. Draw Frequency Spectrum Bars in the background
      const barWidth = (width / bufferLength) * 1.5;
      let barX = 0;
      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (this.freqArray[i] / 255.0) * (height * 0.78);
        const hue = 195 + (i / bufferLength) * 90; // Cyan to purple gradient
        const alpha = isDark ? 0.45 : 0.35;
        ctx.fillStyle = `hsla(${hue}, 90%, 55%, ${alpha})`;
        ctx.fillRect(barX, height - barHeight, barWidth - 1, barHeight);
        barX += barWidth + 1;
        if (barX > width) break;
      }

      // 2. Draw Oscilloscope Waveform Line on top
      ctx.lineWidth = 3;
      const waveGrad = ctx.createLinearGradient(0, 0, width, 0);
      waveGrad.addColorStop(0, '#38bdf8');
      waveGrad.addColorStop(0.5, '#818cf8');
      waveGrad.addColorStop(1, '#c084fc');
      ctx.strokeStyle = waveGrad;
      ctx.shadowBlur = 14;
      ctx.shadowColor = isDark ? 'rgba(56, 189, 248, 0.7)' : 'rgba(37, 99, 235, 0.5)';

      ctx.beginPath();
      const sliceWidth = width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = this.dataArray[i] / 128.0;
        const y = (v * height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }
      ctx.stroke();
      ctx.shadowBlur = 0; // reset shadow
    } else {
      // Resting animation
      this.drawIdleWaveform();
      return;
    }

    if (this.isPlaying) {
      this.animationFrameId = requestAnimationFrame(() => this.renderVisualizerFrame());
    } else {
      this.drawIdleWaveform();
    }
  }

  handleCardClick(event, word) {
    const card = document.getElementById(`card-${word}`) || (event.currentTarget && event.currentTarget.classList.contains('audio-command-card') ? event.currentTarget : null);
    if (card && event) {
      const rect = card.getBoundingClientRect();
      const circle = document.createElement('span');
      const diameter = Math.max(rect.width, rect.height);
      const radius = diameter / 2;
      circle.style.width = circle.style.height = `${diameter}px`;
      circle.style.left = `${(event.clientX || rect.left + radius) - rect.left - radius}px`;
      circle.style.top = `${(event.clientY || rect.top + radius) - rect.top - radius}px`;
      circle.classList.add('audio-ripple');
      const oldRipple = card.querySelector('.audio-ripple');
      if (oldRipple) oldRipple.remove();
      card.appendChild(circle);
      setTimeout(() => circle.remove(), 600);
    }
    this.playWord(word);
  }

  playWord(word, speakerIdx = null) {
    this.initAudioContext();
    if (this.audioContext && this.audioContext.state === 'suspended') {
      this.audioContext.resume().catch(() => {});
    }

    const w = word.toLowerCase().trim();

    if (speakerIdx !== null) {
      this.currentSpeakerIndices[w] = speakerIdx;
    } else if (this.currentSpeakerIndices[w] === undefined) {
      this.currentSpeakerIndices[w] = 0;
    }

    const currentIdx = this.currentSpeakerIndices[w];
    const url = `/api/audio/sample/${w}?speaker_idx=${currentIdx}&t=${Date.now()}`;

    // If already playing this word, stop
    if (this.currentPlayingWord === w && this.isPlaying) {
      this.stopAllAudio();
      return;
    }

    this.stopAllAudio();
    this.currentPlayingWord = w;
    this.updateNowPlayingBanner(w, currentIdx);

    // Try playing authentic dataset WAV clip
    this.audioElement.src = url;
    this.audioElement.playbackRate = this.playbackSpeed;

    const playPromise = this.audioElement.play();
    if (playPromise !== undefined) {
      playPromise.then(() => {
        this.isPlaying = true;
        this.updateUIPlayStates();
        this.startVisualizer();
      }).catch(err => {
        console.warn('Backend audio stream unavailable, falling back to speech synthesis:', err);
        this.playSpeechSynthesisFallback(w);
      });
    } else {
      this.isPlaying = true;
      this.updateUIPlayStates();
      this.startVisualizer();
    }
  }

  stopAllAudio() {
    this.isPlaying = false;
    try {
      this.audioElement.pause();
      this.audioElement.currentTime = 0;
    } catch (e) {}
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    this.updateUIPlayStates();
  }

  playSpeechSynthesisFallback(word) {
    this.isPlaying = true;
    this.updateUIPlayStates();
    this.startVisualizer();

    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(word);
      utter.rate = this.playbackSpeed || 1.0;
      utter.pitch = 1.0;
      utter.lang = 'en-US';
      utter.onend = () => {
        this.isPlaying = false;
        this.updateUIPlayStates();
      };
      utter.onerror = () => {
        this.isPlaying = false;
        this.updateUIPlayStates();
      };
      window.speechSynthesis.speak(utter);
    } else {
      setTimeout(() => {
        this.isPlaying = false;
        this.updateUIPlayStates();
      }, 1000);
    }
  }

  playTour() {
    const tourWords = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'];
    let idx = 0;
    this.showToast('🚀 Starting Guided Acoustic Tour of All 10 Words...', 'info');

    const playNext = () => {
      if (idx >= tourWords.length) {
        this.showToast('✅ Tour complete! All 10 core command sounds played.', 'info');
        return;
      }
      const w = tourWords[idx];
      const card = document.getElementById(`card-${w}`);
      if (card) {
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
      idx++;
      this.playWord(w);

      const onEnd = () => {
        this.audioElement.removeEventListener('ended', onEnd);
        setTimeout(playNext, 450);
      };
      this.audioElement.addEventListener('ended', onEnd);
    };

    playNext();
  }

  playRandom() {
    const words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'];
    const randWord = words[Math.floor(Math.random() * words.length)];
    this.cycleSpeaker(randWord);
  }

  cycleSpeaker(word) {
    const w = word.toLowerCase().trim();
    const current = this.currentSpeakerIndices[w] || 0;
    const meta = this.wordsInfo.find(item => item.word === w);
    const total = meta ? meta.total_clips : 200;

    // Advance to next speaker (wrap around)
    const nextIdx = (current + 1) % total;
    this.currentSpeakerIndices[w] = nextIdx;

    // Update speaker badge on card
    const badge = document.getElementById(`speaker-badge-${w}`);
    if (badge) {
      badge.textContent = `Speaker #${nextIdx + 1} of ${total}`;
    }

    // Play new speaker voice immediately
    this.playWord(w, nextIdx);

    this.showToast(`Switched to voice sample #${nextIdx + 1} for "${w.toUpperCase()}"`, 'info');
  }

  setPlaybackSpeed(speed) {
    this.playbackSpeed = speed;
    this.audioElement.playbackRate = speed;
    document.querySelectorAll('.speed-btn').forEach(b => {
      b.classList.toggle('active', parseFloat(b.dataset.speed) === speed);
    });
  }

  updateNowPlayingBanner(word, speakerIdx) {
    const banner = document.getElementById('audio-studio-banner');
    if (!banner) return;

    const meta = this.wordsInfo.find(m => m.word === word) || {
      display: word.toUpperCase(),
      ipa: `/${word}/`,
      category: 'Command',
      acoustic_summary: 'Acoustic waveform loaded.'
    };

    const wordEl = document.getElementById('studio-banner-word');
    const ipaEl = document.getElementById('studio-banner-ipa');
    const catEl = document.getElementById('studio-banner-category');
    const speakerEl = document.getElementById('studio-banner-speaker');
    const traitsEl = document.getElementById('studio-banner-traits');

    if (wordEl) wordEl.textContent = meta.display;
    if (ipaEl) ipaEl.textContent = meta.ipa;
    if (catEl) catEl.textContent = meta.category;
    if (speakerEl) speakerEl.textContent = `Speaker #${speakerIdx + 1}`;
    if (traitsEl) traitsEl.textContent = meta.acoustic_summary;

    banner.classList.add('playing-glow');
  }

  updateUIPlayStates() {
    // 1. Update Word Cards in Step 1
    document.querySelectorAll('.audio-command-card').forEach(card => {
      const cardWord = card.dataset.word;
      const isThisPlaying = (cardWord === this.currentPlayingWord && this.isPlaying);

      card.classList.toggle('card-playing', isThisPlaying);

      const playBtn = card.querySelector('.btn-card-play');
      if (playBtn) {
        if (isThisPlaying) {
          playBtn.innerHTML = `
            <span class="eq-icon-anim" aria-hidden="true">
              <span class="eq-bar"></span>
              <span class="eq-bar"></span>
              <span class="eq-bar"></span>
            </span>
            <span>Pause</span>
          `;
          playBtn.setAttribute('aria-label', `Pause sound for ${cardWord}`);
        } else {
          playBtn.innerHTML = `
            <span class="play-icon" aria-hidden="true">▶</span>
            <span>Hear Sound</span>
          `;
          playBtn.setAttribute('aria-label', `Play sound for ${cardWord}`);
        }
      }

      // Live animated equalizer in the card corner
      const eqBadge = card.querySelector('.mini-equalizer');
      if (eqBadge) {
        eqBadge.classList.toggle('active', isThisPlaying);
      }
    });

    // 2. Update Master Studio Play/Pause button
    const masterBtn = document.getElementById('master-studio-play');
    if (masterBtn) {
      masterBtn.innerHTML = this.isPlaying ? '⏸ Pause' : '▶ Play';
    }

    // 3. Remove banner glow if paused
    const banner = document.getElementById('audio-studio-banner');
    if (banner && !this.isPlaying) {
      banner.classList.remove('playing-glow');
    }
  }

  async playPairBackToBack(word1, word2, onDone = null) {
    this.initAudioContext();
    if (this.pairSequenceTimer) {
      clearTimeout(this.pairSequenceTimer);
      this.pairSequenceTimer = null;
    }

    const w1 = word1.toLowerCase().trim();
    const w2 = word2.toLowerCase().trim();

    this.showToast(`🎧 Playing comparison pair: "${w1.toUpperCase()}" then "${w2.toUpperCase()}"...`, 'info');

    // Highlight UI comparison state
    const compBanner = document.getElementById('pair-compare-live-status');
    if (compBanner) {
      compBanner.style.display = 'block';
      compBanner.innerHTML = `
        <div class="alert" style="background: var(--primary-light); border-left: 4px solid var(--accent); padding: 0.85rem 1.25rem;">
          <strong>🔊 Sequential Acoustic Playback:</strong> 
          Now playing 1st command <span class="badge badge-primary">${w1.toUpperCase()}</span> ...
        </div>
      `;
    }

    // Step A: Play Word 1
    this.playWord(w1);

    // Step B: Wait for Word 1 to finish, then pause 350ms, then play Word 2
    const handleWord1End = () => {
      this.audioElement.removeEventListener('ended', handleWord1End);

      if (compBanner) {
        compBanner.innerHTML = `
          <div class="alert" style="background: var(--warning-light); border-left: 4px solid var(--warning); padding: 0.85rem 1.25rem;">
            <strong>⏸ Inter-command pause (350ms)...</strong> Notice the acoustic similarity!
          </div>
        `;
      }

      this.pairSequenceTimer = setTimeout(() => {
        if (compBanner) {
          compBanner.innerHTML = `
            <div class="alert" style="background: var(--danger-light); border-left: 4px solid var(--danger); padding: 0.85rem 1.25rem;">
              <strong>🔊 Now playing 2nd command:</strong> <span class="badge badge-danger">${w2.toUpperCase()}</span>
            </div>
          `;
        }

        this.playWord(w2);

        const handleWord2End = () => {
          this.audioElement.removeEventListener('ended', handleWord2End);
          if (compBanner) {
            compBanner.innerHTML = `
              <div class="alert" style="background: var(--success-light); border-left: 4px solid var(--success); padding: 0.85rem 1.25rem;">
                <strong>✅ Pair Comparison Complete.</strong> Did they sound acoustically confusable to your ear?
              </div>
            `;
          }
          if (onDone) onDone();
        };

        this.audioElement.addEventListener('ended', handleWord2End);
      }, 350);
    };

    this.audioElement.addEventListener('ended', handleWord1End);
  }

  showToast(message, type = 'info') {
    let toast = document.getElementById('audio-toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'audio-toast';
      toast.className = 'audio-toast';
      document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.className = `audio-toast show toast-${type}`;
    clearTimeout(this.toastTimeout);
    this.toastTimeout = setTimeout(() => {
      toast.className = 'audio-toast';
    }, 3200);
  }
}

// Global Singleton Instance
window.audioStudio = new AudioStudio();

// Auto-attach canvas as soon as DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  const canvasEl = document.getElementById('studio-audio-canvas');
  if (canvasEl && window.audioStudio) {
    window.audioStudio.attachCanvas(canvasEl);
  }
});
window.addEventListener('load', () => {
  const canvasEl = document.getElementById('studio-audio-canvas');
  if (canvasEl && window.audioStudio) {
    window.audioStudio.attachCanvas(canvasEl);
  }
});

function logger(...args) {
  // console.log('[AudioStudio]', ...args);
}

