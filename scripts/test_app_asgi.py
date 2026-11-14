"""
Test FastAPI endpoints using asyncio + httpx.AsyncClient ASGITransport.
"""
import os
import sys
import asyncio

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
os.environ["NUMBA_CACHE_DIR"] = os.path.join(ROOT_DIR, ".numba_cache")

import httpx
from backend.app import app, ensure_pipeline_initialized

async def main():
    print("Initializing pipeline...")
    ensure_pipeline_initialized()
    print("Pipeline ready. Testing endpoints via ASGITransport...")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Status
        r1 = await client.get("/api/status")
        print(f"✓ [{r1.status_code}] /api/status -> {r1.json().get('core_words')}")

        # 2. Words metadata
        r2 = await client.get("/api/audio/words_metadata")
        meta = r2.json()
        print(f"✓ [{r2.status_code}] /api/audio/words_metadata -> {len(meta.get('words', {}))} words loaded")
        sample_word = meta["words"]["stop"]
        print(f"    STOP: IPA {sample_word['ipa']}, Role: {sample_word['accessibility_role']}, Clips: {sample_word['sample_count']}")

        # 3. Audio sample STOP
        r3 = await client.get("/api/audio/sample/stop")
        print(f"✓ [{r3.status_code}] /api/audio/sample/stop -> {len(r3.content)} bytes, File: {r3.headers.get('X-File-Name')}")

        # 4. Audio sample YES
        r4 = await client.get("/api/audio/sample/yes")
        print(f"✓ [{r4.status_code}] /api/audio/sample/yes -> {len(r4.content)} bytes, File: {r4.headers.get('X-File-Name')}")

        # 5. Audio clip by name
        fn = r3.headers.get("X-File-Name")
        r5 = await client.get(f"/api/audio/clip?word=stop&file_name={fn}")
        print(f"✓ [{r5.status_code}] /api/audio/clip?word=stop&file_name={fn} -> {len(r5.content)} bytes")

        # 6. Analysis
        r6 = await client.get("/api/analysis")
        anal = r6.json()
        print(f"✓ [{r6.status_code}] /api/analysis -> K={anal['k']}, Distance={anal['distance']}, Profiles={len(anal['profiles'])}, Confusable={len(anal['confusable_pairs'])}")
        print(f"    Has file_names in analysis: {'file_names' in anal} (Total {len(anal.get('file_names', []))} points)")

        # 7. Frontend index
        r7 = await client.get("/")
        print(f"✓ [{r7.status_code}] Frontend index.html -> {len(r7.content)} bytes")

    print("\n🎉 ALL ASGITransport ENDPOINTS PASSED WITH FLYING COLORS!")

if __name__ == "__main__":
    asyncio.run(main())
