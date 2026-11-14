"""
Test Live Server Endpoints (Threaded Uvicorn + urllib).
Verifies all API endpoints and audio streaming functionality.
"""
import os
import sys
import threading
import time
import urllib.request
import json

# Setup environment
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
os.environ["NUMBA_CACHE_DIR"] = os.path.join(ROOT_DIR, ".numba_cache")

import uvicorn
from backend.app import app

def run_srv():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

if __name__ == "__main__":
    t = threading.Thread(target=run_srv, daemon=True)
    t.start()
    time.sleep(2.0)

    endpoints = [
        ("Status API", "http://127.0.0.1:8000/api/status"),
        ("Audio Words Metadata", "http://127.0.0.1:8000/api/audio/words_metadata"),
        ("Audio Sample 'STOP'", "http://127.0.0.1:8000/api/audio/sample/stop"),
        ("Audio Sample 'YES'", "http://127.0.0.1:8000/api/audio/sample/yes"),
        ("Audio Sample 'GO'", "http://127.0.0.1:8000/api/audio/sample/go"),
        ("Analysis API", "http://127.0.0.1:8000/api/analysis"),
        ("Frontend HTML Index", "http://127.0.0.1:8000/"),
    ]

    all_ok = True
    for name, url in endpoints:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
                headers = dict(resp.headers)
                file_hdr = headers.get("x-file-name") or headers.get("X-File-Name") or "N/A"
                print(f"✓ [{resp.status}] {name}: {len(data)} bytes (Clip: {file_hdr})")
        except Exception as e:
            print(f"✗ FAILED {name}: {e}")
            all_ok = False

    if all_ok:
        print("\n🎉 ALL SERVER & AUDIO ENDPOINTS FUNCTIONING PERFECTLY!")
    else:
        print("\n⚠️ SOME ENDPOINTS FAILED")
        sys.exit(1)

