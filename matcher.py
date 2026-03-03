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