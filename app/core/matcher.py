"""
app/core/matcher.py — Bandpass filter, offset-alignment voting, and worker
==========================================================================

bandpass(y, sr, low=300, high=3400) -> np.ndarray
    4th-order Butterworth bandpass filter.  Called by ``AudioFingerprinter``
    when ``is_phone_mode=True`` to simulate telephone / GSM channel conditions
    (300–3400 Hz passband, 8 kHz sample rate).

match_sample(r, sample_hashes) -> (song_id, best_offset, best_score)
    Core matching function.

    1. Batch-queries Redis ``fp:{hash_value}`` lists for every hash in
       ``sample_hashes`` using a single pipeline round-trip.
    2. For every matching entry, increments a vote counter:
           votes[song_id][db_time_offset − query_time_offset] += 1
       Offset-alignment makes matching invariant to playback position.
    3. Returns the (song_id, offset_bucket, vote_count) triple for the
       peak of the vote histogram.  Returns ``(None, None, 0)`` on no match.

match_audio(r, fp, y) -> (song_id | None, confidence: float)
    Fingerprint one audio chunk and return the best matching song.

matcher_worker(audio_queue, stop_flag, on_match, on_no_match) -> None
    Consume chunks from audio_queue and invoke result callbacks.

start_matcher_worker(audio_queue, stop_flag, on_match, on_no_match) -> threading.Thread
    Convenience wrapper — starts matcher_worker in a daemon Thread and
    returns the Thread object.
"""

import queue
import threading
import time
from collections import defaultdict

import numpy as np
from scipy.signal import butter, lfilter


def bandpass(y: np.ndarray, sr: int, low: int = 300, high: int = 3400) -> np.ndarray:
    """4th-order Butterworth bandpass filter (300–3400 Hz by default)."""
    nyq = sr / 2
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return lfilter(b, a, y)


def match_sample(r, sample_hashes: list) -> tuple:
    """
    Match sample hashes against the Redis fingerprint store.

    Uses a pipeline to fetch all fp:{hash} lists in one round-trip, then
    performs offset-alignment voting to find the best matching song.

    Parameters:
        r             : redis.Redis client (decode_responses=True)
        sample_hashes : list of (hash_value: int, time_offset: int)

    Returns:
        (best_song_id, best_offset, best_score)
        All three are None / 0 when no fingerprints matched.
    """
    if not sample_hashes:
        return None, None, 0

    sample_time_map: dict = defaultdict(list)
    for h, t in sample_hashes:
        sample_time_map[int(h)].append(t)

    unique_hashes = list(sample_time_map.keys())

    # Single pipeline round-trip
    pipe = r.pipeline()
    for hv in unique_hashes:
        pipe.lrange(f"fp:{hv}", 0, -1)
    raw_results = pipe.execute()

    votes: dict = defaultdict(lambda: defaultdict(int))
    for hv, entries in zip(unique_hashes, raw_results):
        for entry in entries:
            sid_str, t_str = entry.split(":", 1)
            song_id = int(sid_str)
            db_time = int(t_str)
            for sample_time in sample_time_map[hv]:
                delta = int((db_time - sample_time) / 2)
                votes[song_id][delta] += 1

    if not votes:
        return None, None, 0

    best_song_id = None
    best_offset  = None
    best_score   = 0

    for song_id, delta_map in votes.items():
        local_offset, local_score = max(delta_map.items(), key=lambda x: x[1])
        if local_score > best_score:
            best_score   = local_score
            best_song_id = song_id
            best_offset  = local_offset

    return best_song_id, best_offset, best_score


def fingerprint_only(fp, y) -> tuple:
    """
    Pure-CPU step: spectrogram → peaks → hashes.

    Returns:
        (hashes, hash_to_query_times)
        hashes               : list of (hash_value: int, time_offset: int)
        hash_to_query_times  : dict[int, list[int]]
    """
    from collections import defaultdict

    S_db   = fp.generate_spectrogram(y)
    peaks  = fp.find_peaks(S_db)
    hashes = fp.generate_hashes(peaks)

    if not hashes:
        return [], {}

    hash_to_query_times = defaultdict(list)
    for h, t in hashes:
        hash_to_query_times[int(h)].append(t)

    return hashes, hash_to_query_times


def score_matches(hashes, hash_to_query_times, db_rows) -> tuple:
    """
    Vote-and-score step (pure CPU, no I/O).

    Parameters:
        hashes               : list of (hash_value, time_offset)
        hash_to_query_times  : dict[int, list[int]]
        db_rows              : list of (hash_value, song_id, db_time)

    Returns:
        (best_id: int | None, confidence: float, best_offset_bins: int)
        best_offset_bins is the winning delta-time bucket (db_t − query_t) in
        spectrogram frames.  Convert to wall-clock seconds with:
            offset_s = best_offset_bins * hop_length / sample_rate
        e.g. hop_length=256, sample_rate=8000 → multiply by 0.032.
    """
    from collections import defaultdict
    from app.config import MIN_CONFIDENCE

    if not hashes or not db_rows:
        return None, 0.0, 0

    votes: dict = defaultdict(lambda: defaultdict(int))
    for hash_value, song_id, db_t in db_rows:
        for query_t in hash_to_query_times.get(hash_value, []):
            votes[song_id][db_t - query_t] += 1

    scores = {
        sid: max(buckets.values()) / len(hashes)
        for sid, buckets in votes.items()
    }
    scores = {sid: s for sid, s in scores.items() if s >= MIN_CONFIDENCE}

    if not scores:
        return None, 0.0, 0

    best_id          = max(scores, key=scores.get)
    best_offset_bins = max(votes[best_id], key=votes[best_id].get)
    return best_id, scores[best_id], best_offset_bins


def match_audio(r, fp, y) -> tuple:
    """
    Fingerprint one audio chunk and return the best matching song.

    Returns:
        (song_id: int | None, confidence: float, best_offset_bins: int)
    """
    from app.db.fingerprint_repo import match_fingerprints_bulk

    hashes, hash_to_query_times = fingerprint_only(fp, y)
    if not hashes:
        return None, 0.0, 0

    hash_values = [int(h) for h, _ in hashes]
    db_rows     = match_fingerprints_bulk(r, hash_values)
    return score_matches(hashes, hash_to_query_times, db_rows)


def matcher_worker(
    audio_queue: queue.Queue,
    stop_flag:   threading.Event,
    on_match=None,
    on_no_match=None,
) -> None:
    """
    Consume audio chunks, run match_audio, and invoke result callbacks.

    Parameters:
        audio_queue  : queue holding (pcm_array, timestamp) tuples
        stop_flag    : threading.Event — worker exits when set
        on_match     : callable(result: dict) — called on every successful match
                       result keys: name, confidence, start_time, end_time, duration_s
        on_no_match  : callable() — called when a chunk yields no match
    """
    from app.core.fingerprint import AudioFingerprinter
    from app.db.redis import get_connection

    r  = get_connection()
    fp = AudioFingerprinter()

    current_song_id: int | None   = None
    song_start_time: float | None = None

    while not stop_flag.is_set():
        try:
            y, chunk_start = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        best_id, conf, offset_bins = match_audio(r, fp, y)
        offset_s = round(offset_bins * fp.hop_length / fp.sample_rate, 2)

        if best_id is not None:
            name = r.hget(f"song:{best_id}", "name") or "Unknown"

            if best_id != current_song_id:
                current_song_id = best_id
                song_start_time = chunk_start

            end_wall   = time.time()
            duration_s = end_wall - song_start_time

            if on_match:
                on_match({
                    "name":       name,
                    "confidence": conf,
                    "offset_s":   offset_s,
                    "start_time": song_start_time,
                    "end_time":   end_wall,
                    "duration_s": duration_s,
                })
        else:
            current_song_id = None
            song_start_time = None
            if on_no_match:
                on_no_match()

    if hasattr(r, "close"):
        r.close()


def start_matcher_worker(
    audio_queue: queue.Queue,
    stop_flag:   threading.Event,
    on_match=None,
    on_no_match=None,
) -> threading.Thread:
    """
    Start matcher_worker in a daemon thread and return the Thread object.

    Example::

        stop = threading.Event()
        t = start_matcher_worker(q, stop, on_match=handle_match)
        # ... later ...
        stop.set()
        t.join()
    """
    t = threading.Thread(
        target=matcher_worker,
        args=(audio_queue, stop_flag, on_match, on_no_match),
        daemon=True,
    )
    t.start()
    return t
