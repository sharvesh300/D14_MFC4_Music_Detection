"""
app/services/fingerprint_service.py — Song indexing business logic
==================================================================

Wraps audio loading → fingerprinting → Redis insertion into a clean service
function used by the CLI indexing scripts.
"""

import redis
from typing import Any

from app.core.fingerprint import AudioFingerprinter
from app.core.matcher import bandpass
from app.db.fingerprint_repo import insert_fingerprints_bulk
from app.utils.audio import find_audio_file


def fingerprint_song(
    r: redis.Redis[Any],
    song_id: int,
    song_name: str,
    songs_dir: str,
    fp: AudioFingerprinter | None = None,
) -> dict[str, Any]:
    """
    Compute and store fingerprint hashes for a single song.

    Audio is loaded **once** and bandpass is applied on top of the already-
    loaded signal, so we avoid double disk I/O.

    Parameters:
        r          : open Redis client
        song_id    : integer id already stored in Redis
        song_name  : filename stem (without extension)
        songs_dir  : directory that contains the audio files
        fp         : reusable AudioFingerprinter instance (created if None)

    Returns:
        dict with keys: status, song_id, song_name,
                        normal_hashes (int), phone_hashes (int)
        status values: "done" | "skipped" | "not_found" | "error"
    """
    if fp is None:
        fp = AudioFingerprinter()

    if r.exists(f"song:{song_id}:fingerprinted"):
        return {"status": "skipped", "song_id": song_id, "song_name": song_name}

    audio_path = find_audio_file(songs_dir, song_name)
    if audio_path is None:
        return {"status": "not_found", "song_id": song_id, "song_name": song_name}

    try:
        y_raw, sr = fp.preprocess(audio_path, is_phone_mode=False)

        # Broadband hashes
        S_db = fp.generate_spectrogram(y_raw)
        peaks = fp.find_peaks(S_db)
        hashes = fp.generate_hashes(peaks)

        # Phone-mode hashes — bandpass on the already-loaded signal, no re-read
        y_bp = bandpass(y_raw, sr)
        S_db_bp = fp.generate_spectrogram(y_bp)
        peaks_bp = fp.find_peaks(S_db_bp)
        hashes_bp = fp.generate_hashes(peaks_bp)

        insert_fingerprints_bulk(r, song_id, hashes + hashes_bp)
        r.set(f"song:{song_id}:fingerprinted", "1")

        return {
            "status": "done",
            "song_id": song_id,
            "song_name": song_name,
            "normal_hashes": len(hashes),
            "phone_hashes": len(hashes_bp),
        }

    except Exception as exc:
        return {
            "status": "error",
            "song_id": song_id,
            "song_name": song_name,
            "error": str(exc),
        }
