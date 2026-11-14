"""
Acoustic Feature Extractor for Speech Commands.
Extracts 45 acoustic features from 1-second audio clips using soundfile and librosa.
"""
import os
import glob
import logging
from typing import Dict, List, Optional, Tuple, Any

# Ensure numba cache is written to local workspace to avoid sandbox permission locks
_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".numba_cache"))
os.makedirs(_cache_dir, exist_ok=True)
os.environ["NUMBA_CACHE_DIR"] = _cache_dir

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CORE_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
TARGET_SR = 16000
TARGET_SAMPLES = 16000  # 1.0 second at 16kHz


def load_and_fix_audio(file_path: str) -> Tuple[np.ndarray, int, str]:
    """
    Loads audio file, enforces mono, and ensures exactly 1.0 second (16000 samples).
    Pads with zeros if shorter, trims if longer.
    Returns: (audio_array, sample_rate, status)
    status: 'normal', 'padded', 'trimmed', or 'corrupted'
    """
    try:
        data, sr = sf.read(file_path)
    except Exception as e:
        logger.warning(f"Error reading {file_path}: {e}")
        return np.zeros(TARGET_SAMPLES, dtype=np.float32), TARGET_SR, "corrupted"

    # Convert stereo to mono if needed
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if sample rate differs from 16kHz
    if sr != TARGET_SR:
        try:
            data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        except Exception:
            pass

    status = "normal"
    curr_len = len(data)
    if curr_len < TARGET_SAMPLES:
        # Pad with zeros at the end
        pad_width = TARGET_SAMPLES - curr_len
        data = np.pad(data, (0, pad_width), mode="constant", constant_values=0)
        status = "padded"
    elif curr_len > TARGET_SAMPLES:
        # Trim to 1 second
        data = data[:TARGET_SAMPLES]
        status = "trimmed"

    return data.astype(np.float32), sr, status


def extract_clip_features(y: np.ndarray, sr: int = TARGET_SR) -> Dict[str, float]:
    """
    Extracts all 45 required acoustic features from a 1D audio waveform:
      - 13 MFCC means (mfcc_mean_1..13)
      - 13 MFCC stds (mfcc_std_1..13)
      - 13 Delta-MFCC means (delta_mfcc_mean_1..13)
      - Spectral Centroid mean
      - Spectral Bandwidth mean
      - Spectral Rolloff mean (0.85 default roll_percent)
      - Zero-Crossing Rate mean
      - RMS Energy mean
      - Active Speech Duration (seconds with energy > 10% of clip max RMS)
    """
    features = {}

    # Handle silent/empty clip edge case
    max_val = np.max(np.abs(y))
    if max_val < 1e-6:
        # Near silent clip: fill with zero / epsilon values
        for i in range(1, 14):
            features[f"mfcc_mean_{i}"] = 0.0
            features[f"mfcc_std_{i}"] = 0.0
            features[f"delta_mfcc_mean_{i}"] = 0.0
        features["spectral_centroid_mean"] = 0.0
        features["spectral_bandwidth_mean"] = 0.0
        features["spectral_rolloff_mean"] = 0.0
        features["zero_crossing_rate_mean"] = 0.0
        features["rms_mean"] = 0.0
        features["active_duration"] = 0.0
        return features

    # 1. 13 MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_mean_{i+1}"] = float(np.mean(mfccs[i, :]))
        features[f"mfcc_std_{i+1}"] = float(np.std(mfccs[i, :]))

    # 2. Delta MFCCs (means)
    delta_mfcc = librosa.feature.delta(mfccs)
    for i in range(13):
        features[f"delta_mfcc_mean_{i+1}"] = float(np.mean(delta_mfcc[i, :]))

    # 3. Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(cent))

    # 4. Spectral Bandwidth
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spectral_bandwidth_mean"] = float(np.mean(bw))

    # 5. Spectral Rolloff
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    features["spectral_rolloff_mean"] = float(np.mean(ro))

    # 6. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y)
    features["zero_crossing_rate_mean"] = float(np.mean(zcr))

    # 7. RMS Energy
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = float(np.mean(rms))

    # 8. Active Speech Duration (energy-based length)
    rms_frames = rms[0]
    frame_dur = librosa.frames_to_time(1, sr=sr)
    threshold = 0.10 * np.max(rms_frames) if np.max(rms_frames) > 0 else 0
    active_frames = np.sum(rms_frames > threshold)
    features["active_duration"] = float(active_frames * frame_dur)

    return features


def extract_dataset(
    dataset_dir: str,
    output_csv: str,
    samples_per_word: int = 200,
    words: Optional[List[str]] = None,
    seed: int = 42,
    force_recompute: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extracts features for all selected words, balanced sampling with fixed seed.
    Caches results to output_csv.
    Returns: (df_features, extraction_stats)
    """
    if words is None:
        words = CORE_WORDS

    if os.path.exists(output_csv) and not force_recompute:
        logger.info(f"Loading existing cached features from {output_csv}")
        df = pd.read_csv(output_csv)
        # Check if requested words and count match
        existing_words = set(df["word"].unique())
        if set(words).issubset(existing_words) and len(df) >= len(words) * samples_per_word:
            logger.info("Cached features match request. Returning cached DataFrame.")
            stats = {
                "total_clips": len(df),
                "words": sorted(list(df["word"].unique())),
                "samples_per_word": samples_per_word,
                "cached": True,
            }
            return df, stats

    logger.info(f"Extracting features from {dataset_dir} (samples_per_word={samples_per_word}, seed={seed})")
    rng = np.random.RandomState(seed)

    records = []
    log_stats = {
        "total_clips": 0,
        "normal_clips": 0,
        "padded_clips": 0,
        "trimmed_clips": 0,
        "corrupted_clips": 0,
        "per_word_counts": {},
    }

    for word in words:
        word_folder = os.path.join(dataset_dir, word)
        if not os.path.isdir(word_folder):
            logger.warning(f"Word folder not found: {word_folder}")
            continue

        wav_files = glob.glob(os.path.join(word_folder, "*.wav"))
        wav_files = [f for f in wav_files if "_background_noise_" not in f]
        wav_files.sort()  # deterministic sort before shuffle

        if len(wav_files) == 0:
            logger.warning(f"No wav files found in {word_folder}")
            continue

        # Balanced random sample
        n_sample = min(samples_per_word, len(wav_files))
        selected_files = rng.choice(wav_files, size=n_sample, replace=False)
        log_stats["per_word_counts"][word] = len(selected_files)

        for file_path in selected_files:
            audio, sr, status = load_and_fix_audio(file_path)
            if status == "normal":
                log_stats["normal_clips"] += 1
            elif status == "padded":
                log_stats["padded_clips"] += 1
            elif status == "trimmed":
                log_stats["trimmed_clips"] += 1
            elif status == "corrupted":
                log_stats["corrupted_clips"] += 1

            feat = extract_clip_features(audio, sr=sr)
            feat["word"] = word
            feat["file_name"] = os.path.basename(file_path)
            feat["pad_trim_status"] = status
            records.append(feat)

    df = pd.DataFrame(records)
    log_stats["total_clips"] = len(df)
    log_stats["words"] = words
    log_stats["samples_per_word"] = samples_per_word

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Extracted {len(df)} clips and saved to {output_csv}")

    return df, log_stats


def extract_features_from_audio_bytes(audio_bytes: bytes, filename: str = "upload.wav") -> Tuple[Dict[str, float], str]:
    """
    Extracts features from an in-memory audio byte stream (for 'Try it yourself').
    Supports standard WAV, FLAC, OGG, and raw PCM with automated maximum-energy 1-second alignment.
    """
    import io
    data = None
    sr = TARGET_SR

    # 1. Try reading with soundfile
    try:
        with io.BytesIO(audio_bytes) as bio:
            data, sr = sf.read(bio)
    except Exception as e_sf:
        logger.warning(f"Soundfile read failed: {e_sf}. Attempting scipy.io.wavfile fallback.")
        try:
            from scipy.io import wavfile
            with io.BytesIO(audio_bytes) as bio:
                sr, raw_data = wavfile.read(bio)
                if raw_data.dtype == np.int16:
                    data = raw_data.astype(np.float32) / 32768.0
                elif raw_data.dtype == np.int32:
                    data = raw_data.astype(np.float32) / 2147483648.0
                elif raw_data.dtype == np.uint8:
                    data = (raw_data.astype(np.float32) - 128.0) / 128.0
                else:
                    data = raw_data.astype(np.float32)
        except Exception as e_scipy:
            logger.error(f"Both soundfile and scipy failed to decode audio: {e_scipy}")
            raise ValueError(f"Could not decode audio. Please ensure the recording is a valid WAV audio file. Error: {e_sf}")

    # Convert stereo to mono if needed
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if sample rate differs from 16kHz
    if sr != TARGET_SR:
        try:
            data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        except Exception as e_resamp:
            logger.warning(f"Resampling failed: {e_resamp}")

    status = "normal"
    curr_len = len(data)
    if curr_len < TARGET_SAMPLES:
        pad_width = TARGET_SAMPLES - curr_len
        data = np.pad(data, (0, pad_width), mode="constant", constant_values=0)
        status = "padded"
    elif curr_len > TARGET_SAMPLES:
        # Find the 1-second window with maximum energy to capture the spoken word
        hop = 512
        max_energy = -1.0
        best_start = 0
        for start_i in range(0, curr_len - TARGET_SAMPLES + 1, hop):
            window = data[start_i : start_i + TARGET_SAMPLES]
            energy = float(np.sum(window**2))
            if energy > max_energy:
                max_energy = energy
                best_start = start_i
        data = data[best_start : best_start + TARGET_SAMPLES]
        status = "trimmed"

    feat = extract_clip_features(data.astype(np.float32), sr=TARGET_SR)
    return feat, status

