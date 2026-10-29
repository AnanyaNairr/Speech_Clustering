
"""Download Google Speech Commands and create a balanced subset."""

import argparse
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    ARCHIVE_PATH,
    BALANCED_DIR,
    DATA_DIR,
    DATASET_URL,
    DOWNLOAD_CHUNK_SIZE,
    RANDOM_SEED,
    RAW_DIR,
    WORDS,
    get_dataset_root,
    samples_per_word,
)


def download_archive(url, destination):
    """Download the dataset with a progress bar."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using existing archive: {destination}")
        return

    print("Downloading the official Speech Commands v0.02 archive.")
    print("The archive is large, so this may take some time.")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SpeechClusteringProject/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))

            with open(destination, "wb") as output:
                with tqdm(
                    total=total if total > 0 else None,
                    unit="B",
                    unit_scale=True,
                    desc="Download",
                ) as progress:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)
                        progress.update(len(chunk))

    except Exception:
        if destination.exists():
            destination.unlink()
        raise

    print(f"Download completed: {destination}")


def extract_archive(archive_path, destination):
    """Extract the archive after checking member paths."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    print(f"Extracting archive into: {destination}")

    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()

            if target != root and root not in target.parents:
                raise ValueError(
                    f"Unsafe archive path: {member.name}"
                )

        archive.extractall(destination)

    print("Extraction completed.")


def find_dataset_root():
    """Find the extracted dataset without downloading it again."""
    try:
        return get_dataset_root()
    except FileNotFoundError:
        return None


def build_balanced_subset(dataset_root, lite=False, force=False):
    """Copy a reproducible, balanced subset into data/balanced_subset."""
    limit = samples_per_word(lite=lite)
    manifest_path = DATA_DIR / "balanced_manifest.csv"

    # Reuse an existing valid manifest unless rebuilding was requested.
    if manifest_path.exists() and not force:
        try:
            old = pd.read_csv(manifest_path)
            if (
                not old.empty
                and all(Path(p).is_file() for p in old["path"])
            ):
                print(f"Reusing existing manifest: {manifest_path}")
                print(f"Existing clips: {len(old)}")
                print("Use --force to rebuild the subset.")
                return
        except (OSError, ValueError, KeyError):
            print("Existing manifest is invalid; rebuilding.")

    # Remove old selected word folders to avoid stale files.
    for word in WORDS:
        word_dir = BALANCED_DIR / word
        if word_dir.exists():
            shutil.rmtree(word_dir)

    rng = random.Random(RANDOM_SEED)
    records = []

    for word in WORDS:
        source_dir = dataset_root / word

        if not source_dir.is_dir():
            print(f"WARNING: Missing word folder: {source_dir}")
            continue

        files = sorted(source_dir.glob("*.wav"))
        rng.shuffle(files)
        selected = files[:limit]

        target_dir = BALANCED_DIR / word
        target_dir.mkdir(parents=True, exist_ok=True)

        for source in tqdm(
            selected,
            desc=f"Selecting {word}",
            unit="clip",
        ):
            target = target_dir / source.name
            shutil.copy2(source, target)

            # Speaker ID is retained only for later evaluation.
            speaker_id = source.stem.split("_nohash_")[0]

            records.append({
                "word": word,
                "speaker_id": speaker_id,
                "path": str(target.resolve()),
                "source_path": str(source.resolve()),
            })

        print(f"{word}: {len(selected)} clips selected")

    if not records:
        raise RuntimeError(
            "No recordings were selected. Check the dataset folders."
        )

    manifest = pd.DataFrame(records)
    manifest.to_csv(manifest_path, index=False)

    counts = (
        manifest.groupby("word")
        .size()
        .reindex(WORDS, fill_value=0)
        .rename("clip_count")
    )
    counts.to_csv(DATA_DIR / "balanced_word_counts.csv")

    print("\nSubset creation complete.")
    print(counts.to_string())
    print(f"Total clips: {len(manifest)}")
    print(f"Manifest: {manifest_path}")
    print(f"Word counts: {DATA_DIR / 'balanced_word_counts.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare Speech Commands v0.02."
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Select at most 50 clips per word.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the balanced subset.",
    )
    parser.add_argument(
        "--manual-only",
        action="store_true",
        help="Use an extracted dataset without downloading.",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataset_root = find_dataset_root()

    if dataset_root is None:
        if args.manual_only:
            raise FileNotFoundError(
                "No extracted dataset found. Place the extracted "
                "word folders under data/raw/speech_commands_v0.02/ "
                "or data/raw/."
            )

        download_archive(DATASET_URL, ARCHIVE_PATH)
        extract_archive(ARCHIVE_PATH, RAW_DIR)
        dataset_root = find_dataset_root()

    if dataset_root is None:
        raise FileNotFoundError(
            "Dataset extraction finished, but the expected word "
            "folders were not found. Check the extracted structure."
        )

    print(f"Dataset root: {dataset_root}")

    build_balanced_subset(
        dataset_root,
        lite=args.lite,
        force=args.force,
    )


if __name__ == "__main__":
    main()
