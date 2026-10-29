
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from scipy.fft import dct
from scipy.signal import resample_poly
from tqdm import tqdm

from config import (
    BALANCED_DIR,
    CLIP_SAMPLES,
    DATA_DIR,
    N_FFT,
    N_MELS,
    N_MFCC,
    SAMPLE_RATE,
)

MANIFEST_PATH = DATA_DIR / "balanced_manifest.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
ERRORS_PATH = DATA_DIR / "feature_extraction_errors.csv"


def load_audio(path):
    """Load mono audio, resample it, and fit it to one second."""
    audio, sample_rate = sf.read(path, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sample_rate != SAMPLE_RATE:
        divisor = np.gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            sample_rate // divisor,
        ).astype(np.float32)

    if len(audio) < CLIP_SAMPLES:
        audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
    else:
        audio = audio[:CLIP_SAMPLES]

    return audio


def mel_filterbank(sample_rate, n_fft, n_mels):
    """Create triangular filters on the mel frequency scale."""
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    frequencies = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    mel_points = np.linspace(
        hz_to_mel(0),
        hz_to_mel(sample_rate / 2),
        n_mels + 2,
    )
    hz_points = mel_to_hz(mel_points)
    filters = np.zeros((n_mels, len(frequencies)), dtype=np.float32)

    for i in range(n_mels):
        left, center, right = hz_points[i:i + 3]

        filters[i] = np.maximum(
            0,
            np.minimum(
                (frequencies - left) / max(center - left, 1e-12),
                (right - frequencies) / max(right - center, 1e-12),
            ),
        )

    return filters


def extract_features(audio):
    """Extract MFCC, delta MFCC, and spectral summary features."""
    _, _, spectrum = signal.stft(
        audio,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=N_FFT,
        noverlap=N_FFT - 160,
        nfft=N_FFT,
        boundary="zeros",
        padded=True,
    )

    magnitude = np.abs(spectrum)
    power = magnitude ** 2

    filters = mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS)
    mel_power = filters @ power
    log_mel = np.log(np.maximum(mel_power, 1e-10))

    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:N_MFCC]
    delta = np.gradient(mfcc, axis=1) if mfcc.shape[1] > 1 else np.zeros_like(mfcc)

    features = {}

    for i in range(N_MFCC):
        features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))
        features[f"delta_mfcc_{i + 1}_mean"] = float(np.mean(delta[i]))
        features[f"delta_mfcc_{i + 1}_std"] = float(np.std(delta[i]))

    frequencies = np.linspace(0, SAMPLE_RATE / 2, power.shape[0])
    power_sum = np.sum(power, axis=0) + 1e-12
    magnitude_sum = np.sum(magnitude, axis=0) + 1e-12

    centroid = np.sum(frequencies[:, None] * magnitude, axis=0) / magnitude_sum
    bandwidth = np.sqrt(
        np.sum(((frequencies[:, None] - centroid) ** 2) * magnitude, axis=0)
        / magnitude_sum
    )

    cumulative_power = np.cumsum(power, axis=0)
    rolloff_threshold = 0.85 * power_sum
    rolloff_indices = np.argmax(
        cumulative_power >= rolloff_threshold[None, :], axis=0
    )
    rolloff = frequencies[rolloff_indices]

    zcr = np.mean(
        np.abs(np.diff(np.signbit(audio).astype(np.float32)))
    )
    rms = np.sqrt(np.mean(audio ** 2))

    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))
    features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(bandwidth))
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))
    features["zero_crossing_rate"] = float(zcr)
    features["rms_energy"] = float(rms)

    return features


def resolve_audio_path(path_value):
    """Resolve paths from the dataset manifest."""
    path = Path(str(path_value))

    if path.is_absolute() and path.exists():
        return path

    candidates = [
        path,
        Path.cwd() / path,
        BALANCED_DIR / path.name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Audio file not found: {path_value}")


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}. "
            "Run 01_download_data.py first."
        )

    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Manifest columns: {manifest.columns.tolist()}")

    path_column = next(
        (c for c in ["path", "filepath", "file_path", "file"] if c in manifest.columns),
        None,
    )
    if path_column is None:
        raise ValueError("Manifest has no recognised audio-path column.")

    records = []
    errors = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Extracting"):
        audio_path = str(row[path_column])

        try:
            resolved_path = resolve_audio_path(audio_path)
            audio = load_audio(resolved_path)
            features = extract_features(audio)

            record = row.to_dict()
            record["path"] = str(resolved_path)
            record.update(features)
            records.append(record)

        except Exception as exc:
            errors.append({
                "path": audio_path,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

    # Save the error log even if every audio file fails.
    pd.DataFrame(
        errors,
        columns=["path", "error_type", "error_message"],
    ).to_csv(ERRORS_PATH, index=False)

    if not records:
        print("\nNo features were extracted.")
        if errors:
            print(pd.DataFrame(errors)["error_message"].value_counts().head(10))
        raise RuntimeError(f"No features extracted. Check {ERRORS_PATH}")

    feature_df = pd.DataFrame(records)
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    feature_columns = feature_df.select_dtypes(include=[np.number]).columns
    feature_df[feature_columns] = feature_df[feature_columns].fillna(0)

    feature_df.to_csv(FEATURES_PATH, index=False)

    print("\nFeature extraction complete.")
    print(f"Successful files: {len(records)}")
    print(f"Failed files: {len(errors)}")
    print(f"Feature table: {FEATURES_PATH}")
    print(f"Error log: {ERRORS_PATH}")
    print(f"Feature table shape: {feature_df.shape}")


if __name__ == "__main__":
    main()
