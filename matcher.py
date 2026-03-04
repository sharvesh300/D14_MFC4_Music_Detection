"""
matcher.py — Hash Lookup and Offset-Alignment Voting
=====================================================

Exposes two public functions used by the matching pipeline:

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
       This offset-alignment step makes matching invariant to where in the
       song the clip was recorded.
    3. Returns the (song_id, offset_bucket, vote_count) triple for the
       peak of the vote histogram.  Returns ``(None, None, 0)`` if no
       hashes matched.

    Arguments
    ---------
    r             : redis.Redis — an open Redis client (decode_responses=True)
    sample_hashes : list — [(hash_value: int, time_offset: int), ...]
                          as returned by ``AudioFingerprinter.generate_hashes``

    Returns
    -------
    song_id    : int | None  — primary key stored as song:{id} in Redis
    best_offset: int | None  — most-voted time-delta bucket
    best_score : int         — raw vote count for the winning (song, offset)

Usage
-----
    from fingerprint import AudioFingerprinter
    from matcher import match_sample
    from database import get_connection

    r      = get_connection()
    fp     = AudioFingerprinter()
    y, sr  = fp.preprocess("clip.wav")
    hashes = fp.generate_hashes(fp.find_peaks(fp.generate_spectrogram(y)))
    song_id, offset, score = match_sample(r, hashes)

Dependencies
------------
    collections, redis, scipy.signal
"""

from collections import defaultdict
from scipy.signal import butter, lfilter


def bandpass(y, sr, low=300, high=3400):
    nyq = sr / 2
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return lfilter(b, a, y)


def match_sample(r, sample_hashes):
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

    sample_time_map = defaultdict(list)
    for h, t in sample_hashes:
        sample_time_map[int(h)].append(t)

    unique_hashes = list(sample_time_map.keys())

    # Fetch all matching fingerprint entries in a single pipeline round-trip
    pipe = r.pipeline()
    for hv in unique_hashes:
        pipe.lrange(f"fp:{hv}", 0, -1)
    raw_results = pipe.execute()

    votes = defaultdict(lambda: defaultdict(int))
    for hv, entries in zip(unique_hashes, raw_results):
        for entry in entries:
            sid_str, t_str = entry.split(":", 1)
            song_id  = int(sid_str)
            db_time  = int(t_str)
            for sample_time in sample_time_map[hv]:
                delta = int((db_time - sample_time) / 2)
                votes[song_id][delta] += 1

    if not votes:
        return None, None, 0

    best_song_id = None
    best_offset  = None
    best_score   = 0

    for song_id, delta_map in votes.items():
        local_best_offset, local_best_score = max(delta_map.items(), key=lambda x: x[1])
        if local_best_score > best_score:
            best_score   = local_best_score
            best_song_id = song_id
            best_offset  = local_best_offset

    return best_song_id, best_offset, best_score