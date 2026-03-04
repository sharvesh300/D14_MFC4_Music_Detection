"""
Match a single audio clip against the fingerprint database.

Usage:
    python scripts/match_song.py <path_to_audio_file> [--phone-mode]

Example:
    python scripts/match_song.py songs/samples/Neelothi_sample_2.wav
    python scripts/match_song.py songs/samples_noisy/Neelothi_sample_2_heavy.wav
    python scripts/match_song.py songs/test/thee_koluthi.wav --phone-mode
"""

import os
import sys
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fingerprint import AudioFingerprinter
from database import get_connection, match_fingerprints_bulk

MIN_CONFIDENCE = 0.02
TOP_N          = 5


def match(r, fp, audio_path, is_phone_mode=False):
    y, sr  = fp.preprocess(audio_path, is_phone_mode=is_phone_mode)
    S_db   = fp.generate_spectrogram(y)
    peaks  = fp.find_peaks(S_db)
    hashes = fp.generate_hashes(peaks)

    n_hashes = len(hashes)
    if n_hashes == 0:
        return [], 0

    hash_values = [int(h) for h, _ in hashes]

    hash_to_query_times = defaultdict(list)
    for h, t in hashes:
        hash_to_query_times[int(h)].append(t)

    db_rows = match_fingerprints_bulk(r, hash_values)

    votes = defaultdict(lambda: defaultdict(int))
    for hash_value, song_id, db_t in db_rows:
        for query_t in hash_to_query_times[hash_value]:
            votes[song_id][db_t - query_t] += 1

    scores = {
        sid: max(buckets.values()) / n_hashes
        for sid, buckets in votes.items()
    }
    scores = {sid: s for sid, s in scores.items() if s >= MIN_CONFIDENCE}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:TOP_N], n_hashes


def song_name_from_id(r, song_id):
    return r.hget(f"song:{song_id}", "name") or "Unknown"


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/match_song.py <path_to_audio_file> [--phone-mode]")
        sys.exit(1)

    audio_path    = sys.argv[1]
    is_phone_mode = "--phone-mode" in sys.argv

    if not os.path.isfile(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)

    r  = get_connection()
    if not r.exists("songs:counter"):
        print("No songs found in Redis — run insert_songs.py and fingerprint_songs.py first.")
        sys.exit(1)

    print(f"Query      : {audio_path}")
    print(f"Phone mode : {'ON' if is_phone_mode else 'OFF'}")
    print(f"DB         : Redis {r.connection_pool.connection_kwargs['host']}:"
          f"{r.connection_pool.connection_kwargs['port']}\n")

    fp = AudioFingerprinter()

    ranked, n_hashes = match(r, fp, audio_path, is_phone_mode=is_phone_mode)

    print(f"Hashes generated : {n_hashes}")
    print(f"Matches found    : {len(ranked)}\n")

    if not ranked:
        print("No match found (confidence below threshold).")
        return

    print(f"  {'Rank':<6} {'Song':<35} {'Confidence':>12}")
    print(f"  {'-'*55}")
    for rank, (song_id, conf) in enumerate(ranked, start=1):
        name   = song_name_from_id(r, song_id)
        marker = "  <-- BEST MATCH" if rank == 1 else ""
        print(f"  {rank:<6} {name:<35} {conf:>11.4f}{marker}")


if __name__ == "__main__":
    main()
