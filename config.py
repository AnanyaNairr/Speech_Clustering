
"""Shared configuration for the speech clustering project."""

from pathlib import Path

# Project root.
ROOT_DIR = Path(__file__).resolve().parent

# Data directories.
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
BALANCED_DIR = DATA_DIR / "balanced_subset"

# Model and output directories.
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Official Speech Commands v0.02 archive.
DATASET_URL = (
    "http://download.tensorflow.org/data/"
    "speech_commands_v0.02.tar.gz"
)
ARCHIVE_PATH = RAW_DIR / "speech_commands_v0.02.tar.gz"
EXTRACTED_DIR = RAW_DIR / "speech_commands_v0.02"

# Reproducibility.
RANDOM_SEED = 42

# Audio configuration.
SAMPLE_RATE = 16000
CLIP_DURATION_SECONDS = 1.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_SECONDS)

# Acoustic feature configuration.
N_MFCC = 20
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 40

# Accessibility-related vocabulary.
WORDS = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go",
    "forward", "backward", "follow",
]

# Sample limits per word.
SAMPLES_PER_WORD = 300
LITE_SAMPLES_PER_WORD = 50

# Use fewer recordings when running on limited resources.
LITE_MODE = False

# Limit expensive evaluation computations in later phases.
METRIC_SAMPLE_SIZE = 2500

# Download chunk size: 1 MB.
DOWNLOAD_CHUNK_SIZE = 1024 * 1024

# Ensure required directories exist.
for directory in (
    DATA_DIR,
    RAW_DIR,
    BALANCED_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


def samples_per_word(lite=None):
    """Return the selected sample limit for each word."""
    use_lite = LITE_MODE if lite is None else lite
    return LITE_SAMPLES_PER_WORD if use_lite else SAMPLES_PER_WORD


def get_dataset_root():
    """Locate the extracted dataset in supported locations."""
    candidates = [EXTRACTED_DIR, RAW_DIR, DATA_DIR]

    for candidate in candidates:
        if all((candidate / word).is_dir() for word in WORDS):
            return candidate

    raise FileNotFoundError(
        "Speech Commands dataset was not found. "
        "Run 01_download_data.py or place the extracted word "
        f"folders under {EXTRACTED_DIR}, {RAW_DIR}, or {DATA_DIR}."
    )
