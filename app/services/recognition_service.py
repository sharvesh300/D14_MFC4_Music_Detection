"""
app/services/recognition_service.py — Audio recognition business logic
=======================================================================

Wraps the low-level fingerprinting + Redis lookup into a clean service
function used by both the REST API and CLI scripts.
"""

from collections import defaultdict

from typing import Any

import redis

from app.config import MIN_CONFIDENCE
from app.core.fingerprint import AudioFingerprinter
from app.db.fingerprint_repo import match_fingerprints_bulk, song_name_from_id
from app.models.response import MatchResponse, MatchResult


def match(
    r: redis.Redis[Any],
    fp: AudioFingerprinter,
    audio_path: str,
    is_phone_mode: bool = False,
    top_n: int = 5,
    min_confidence: float = MIN_CONFIDENCE,
) -> MatchResponse:
    """
    Fingerprint an audio file and return ranked match results.

    Parameters:
        r             : open Redis client
        fp            : AudioFingerprinter instance
        audio_path    : path to the query audio file
        is_phone_mode : apply 300–3400 Hz bandpass before fingerprinting
        top_n         : maximum number of results to return
        min_confidence: discard results below this normalised score

    Returns:
        MatchResponse with a ranked list of MatchResult objects.
    """
    y, _ = fp.preprocess(audio_path, is_phone_mode=is_phone_mode)
    S_db = fp.generate_spectrogram(y)
    peaks = fp.find_peaks(S_db)
    hashes = fp.generate_hashes(peaks)

    n_hashes = len(hashes)
    if n_hashes == 0:
        return MatchResponse(query_path=audio_path, n_hashes=0, matched=False)

    hash_values = [int(h) for h, _ in hashes]
    hash_to_query_times = defaultdict(list)
    for h, t in hashes:
        hash_to_query_times[int(h)].append(t)

    db_rows = match_fingerprints_bulk(r, hash_values)

    votes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for hash_value, song_id, db_t in db_rows:
        for query_t in hash_to_query_times[hash_value]:
            votes[song_id][db_t - query_t] += 1

    scores = {sid: max(buckets.values()) / n_hashes for sid, buckets in votes.items()}
    scores = {sid: s for sid, s in scores.items() if s >= min_confidence}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = [
        MatchResult(
            song_id=sid,
            song_name=song_name_from_id(r, sid),
            confidence=conf,
        )
        for sid, conf in ranked
    ]

    return MatchResponse(
        query_path=audio_path,
        n_hashes=n_hashes,
        matched=bool(results),
        results=results,
    )
