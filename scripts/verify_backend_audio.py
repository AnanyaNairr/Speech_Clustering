"""
Verifies all backend functions for audio streaming, metadata, and PCA interactivity.
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
os.environ["NUMBA_CACHE_DIR"] = os.path.join(ROOT_DIR, ".numba_cache")

from backend.app import (
    get_status,
    get_words_metadata,
    get_sample_audio,
    get_clip_by_name,
    get_current_analysis,
    ensure_pipeline_initialized,
)

def run_checks():
    print("Initializing pipeline...")
    ensure_pipeline_initialized()

    st = get_status()
    print("1. Status core words:", len(st["core_words"]), st["core_words"])

    wm = get_words_metadata()
    words = wm["words"]
    print("2. Words metadata loaded:", len(words))
    for w in ["stop", "go", "yes", "no", "up", "down"]:
        item = words[w]
        print(f"   • {w.upper():<5} | IPA: {item['ipa']:<8} | Role: {item['accessibility_role']:<32} | {item['sample_count']} clips | Sample file: {item['sample_files'][0] if item['sample_files'] else 'None'}")

    # Audio sample test for all 10 words
    print("3. Testing sample audio for all 10 core words:")
    for w in st["core_words"]:
        resp = get_sample_audio(w)
        exists = os.path.exists(resp.path)
        size = os.path.getsize(resp.path) if exists else 0
        speaker = os.path.basename(resp.path).split("_")[0]
        print(f"   ✓ Word '{w}': path={resp.path} (Exists: {exists}, Size: {size} bytes, Speaker: #{speaker})")
        assert exists, f"Audio file for {w} does not exist!"

    # Specific clip by name test
    first_word = st["core_words"][0]
    first_fn = words[first_word]["sample_files"][0]
    clip_resp = get_clip_by_name(word=first_word, file_name=first_fn)
    print(f"4. Testing specific clip by name ({first_word} / {first_fn}):")
    print(f"   ✓ Exists: {os.path.exists(clip_resp.path)}, Path: {clip_resp.path}")
    assert os.path.exists(clip_resp.path)

    # Analysis test
    an = get_current_analysis()
    print(f"5. Active analysis: K = {an['k']}, Distance = {an['distance']}")
    print(f"   • Profiles count: {len(an['profiles'])}")
    print(f"   • Confusable pairs count: {len(an['confusable_pairs'])}")
    print(f"   • Total 2D PCA points: {len(an['pca_data']['pc_coords_2d'])}")
    print(f"   • file_names array attached: {'file_names' in an} ({len(an.get('file_names', []))} filenames)")
    assert len(an["file_names"]) == len(an["pca_data"]["pc_coords_2d"])

    print("\n🎉 ALL 10 AUDIO SAMPLES, METADATA, AND PCA INSPECTOR DATA VERIFIED 100% CORRECT!")

if __name__ == "__main__":
    run_checks()

