"""
matcher.py — Hash Lookup and Offset-Alignment Voting
=====================================================

Exposes two public functions used by the matching pipeline:

bandpass(y, sr, low=300, high=3400) -> np.ndarray
    4th-order Butterworth bandpass filter.  Called by ``AudioFingerprinter``
    when ``is_phone_mode=True`` to simulate telephone / GSM channel conditions
    (300–3400 Hz passband, 8 kHz sample rate).

match_sample(db_path, sample_hashes) -> (song_id, best_offset, best_score)
    Core matching function.

    1. Batch-queries the SQLite ``fingerprints`` table for every hash in
       ``sample_hashes`` using a single ``WHERE hash_value IN (...)`` call.
    2. For every matching DB row, increments a vote counter:
           votes[song_id][db_time_offset − query_time_offset] += 1
       This offset-alignment step makes matching invariant to where in the
       song the clip was recorded.
    3. Returns the (song_id, offset_bucket, vote_count) triple for the
       peak of the vote histogram.  Returns ``(None, None, 0)`` if no
       hashes matched.

    Arguments
    ---------
    db_path       : str  — path to the SQLite database file
    sample_hashes : list — [(hash_value: int, time_offset: int), ...]
                          as returned by ``AudioFingerprinter.generate_hashes``

    Returns
    -------
    song_id    : int | None  — primary key in the ``songs`` table
    best_offset: int | None  — most-voted time-delta bucket
    best_score : int         — raw vote count for the winning (song, offset)

Usage
-----
    from fingerprint import AudioFingerprinter
    from matcher import match_sample

    fp     = AudioFingerprinter()
    y, sr  = fp.preprocess("clip.wav")
    hashes = fp.generate_hashes(fp.find_peaks(fp.generate_spectrogram(y)))
    song_id, offset, score = match_sample("database/fingerprints.db", hashes)

Dependencies
------------
    collections, sqlite3, scipy.signal
"""

from collections import defaultdict
import sqlite3
from scipy.signal import butter, lfilter


def bandpass(y, sr, low=300, high=3400):
    nyq = sr / 2
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return lfilter(b, a, y)


def match_sample(db_path, sample_hashes):

    if not sample_hashes:
        return None, None, 0

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    hash_values = list(set(h for h, _ in sample_hashes))

    sample_time_map = defaultdict(list)
    for h, t in sample_hashes:
        sample_time_map[h].append(t)

    placeholders = ",".join("?" for _ in hash_values)

    query = f"""
        SELECT hash_value, song_id, time_offset
        FROM fingerprints
        WHERE hash_value IN ({placeholders})
    """

    cursor.execute(query, hash_values)
    db_rows = cursor.fetchall()
    conn.close()

    # Proper nested vote structure
    votes = defaultdict(lambda: defaultdict(int))

    for hash_value, song_id, db_time in db_rows:
        for sample_time in sample_time_map[hash_value]:
            delta = int((db_time - sample_time) / 2)  # bucketed delta
            votes[song_id][delta] += 1

    if not votes:
        return None, None, 0

    best_song_id = None
    best_offset = None
    best_score = 0

    for song_id, delta_map in votes.items():
        if not delta_map:
            continue
        local_best_offset, local_best_score = max(
            delta_map.items(), key=lambda x: x[1]
        )
        if local_best_score > best_score:
            best_score = local_best_score
            best_song_id = song_id
            best_offset = local_best_offset

    return best_song_id, best_offset, best_score