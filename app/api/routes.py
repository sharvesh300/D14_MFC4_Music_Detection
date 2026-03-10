"""
app/api/routes.py — HTTP REST endpoints
========================================

GET  /api/health       Liveness probe
GET  /api/songs        List all indexed songs
POST /api/match        Upload an audio file and get match results
"""

import os
import tempfile

from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.core.fingerprint import AudioFingerprinter
from app.db.fingerprint_repo import get_all_songs
from app.db.redis import get_connection
from app.services.recognition_service import match

router = APIRouter(prefix="/api")

_fp = AudioFingerprinter()  # shared, stateless — safe to reuse across requests


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/songs")
def list_songs() -> dict[str, Any]:
    """Return every song registered in Redis."""
    r = get_connection()
    songs = get_all_songs(r)
    return {
        "count": len(songs),
        "songs": [{"id": sid, "name": name} for sid, name in songs],
    }


@router.post("/match")
async def match_song(
    file: UploadFile = File(...),
    phone_mode: bool = Query(
        False, description="Apply 300–3400 Hz bandpass (phone/mic simulation)"
    ),
    top_n: int = Query(5, description="Maximum number of results to return"),
) -> dict[str, Any]:
    """
    Upload an audio file and return the top matching songs.

    Accepts: mp3, wav, flac, ogg
    """
    suffix = os.path.splitext(file.filename or "audio")[1].lower()
    if suffix not in {".mp3", ".wav", ".flac", ".ogg"}:
        raise HTTPException(
            status_code=400, detail=f"Unsupported audio format: {suffix}"
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        r = get_connection()
        response = match(r, _fp, tmp_path, is_phone_mode=phone_mode, top_n=top_n)
    finally:
        os.unlink(tmp_path)

    return {
        "matched": response.matched,
        "n_hashes": response.n_hashes,
        "results": [
            {
                "song_id": m.song_id,
                "song_name": m.song_name,
                "confidence": round(m.confidence, 4),
            }
            for m in response.results
        ],
    }
