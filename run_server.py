"""
Local Web Server Launcher for Acoustic Feature Clustering Application.
Starts Uvicorn server hosting FastAPI backend and static frontend.
"""
import os
import sys
import uvicorn

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
os.environ["NUMBA_CACHE_DIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".numba_cache"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("=" * 70)
    print("  ACOUSTIC FEATURE CLUSTERING FOR ACCESSIBILITY APPLICATIONS")
    print(f"  Starting web application on: http://127.0.0.1:{port}")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 70)
    uvicorn.run("backend.app:app", host="127.0.0.1", port=port, reload=True, log_level="info")

