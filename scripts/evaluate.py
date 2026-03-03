"""
Evaluate the audio fingerprinting system against clean and noisy sample clips.

Clean  samples: songs/samples/          (e.g. Neelothi_sample_2.wav)
Noisy  samples: songs/samples_noisy/    (e.g. Neelothi_sample_2_medium.wav)

Expected song name is derived from the filename:
  "Neelothi_sample_2.wav"         → "Neelothi"
  "Neelothi_sample_2_medium.wav"  → "Neelothi"

Output: per-sample results + aggregate metrics per set (clean / per noise level).
"""

import os
import sys
import sqlite3
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fingerprint import AudioFingerprinter
from database import match_fingerprints_bulk

DB_PATH      = os.path.join(os.path.dirname(__file__), "..", "database", "fingerprints.db")
SAMPLES_DIR  = os.path.join(os.path.dirname(__file__), "..", "songs", "samples")
NOISY_DIR    = os.path.join(os.path.dirname(__file__), "..", "songs", "samples_noisy")

NOISE_LEVELS   = ["light", "medium", "heavy"]
MIN_CONFIDENCE = 0.02   # normalised score threshold — reject matches below this


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def query(conn, fp, audio_path, top_n=3):
    """
    Fingerprint a clip and return ranked (song_id, confidence) pairs.

    Optimizations applied:
      - Query optimization : single batched SQL IN (...) instead of N individual queries
      - Score normalization: raw vote count / total query hashes  →  0–1 confidence
      - Thresholding       : results below MIN_CONFIDENCE are discarded as no-match
    """
    y, sr  = fp.preprocess(audio_path)
    S_db   = fp.generate_spectrogram(y)
    peaks  = fp.find_peaks(S_db)
    hashes = fp.generate_hashes(peaks)

    n_hashes = len(hashes)
    if n_hashes == 0:
        return [], 0

    # --- Query optimization: batch lookup ---
    hash_values = [int(h) for h, _ in hashes]

    # Map hash_value → all query frame indices (a hash can appear at multiple times)
    hash_to_query_times = defaultdict(list)
    for h, t in hashes:
        hash_to_query_times[int(h)].append(t)

    db_rows = match_fingerprints_bulk(conn, hash_values)  # (hash_value, song_id, db_t)

    # --- Offset-alignment voting ---
    votes = defaultdict(lambda: defaultdict(int))
    for hash_value, song_id, db_t in db_rows:
        for query_t in hash_to_query_times[hash_value]:
            votes[song_id][db_t - query_t] += 1

    # --- Score normalization ---
    scores = {
        sid: max(buckets.values()) / n_hashes
        for sid, buckets in votes.items()
    }

    # --- Thresholding ---
    scores = {sid: s for sid, s in scores.items() if s >= MIN_CONFIDENCE}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n], n_hashes


def song_name_from_id(conn, song_id):
    row = conn.execute("SELECT name FROM songs WHERE id = ?", (song_id,)).fetchone()
    return row[0] if row else "Unknown"


def expected_name_from_filename(filename, noise_levels):
    """
    Strip '_sample_N[_noiselevel].wav' to recover the original song name.
    Works for both clean and noisy filenames.
    """
    base = os.path.splitext(filename)[0]
    # Strip noise suffix first (e.g. _medium)
    for level in noise_levels:
        if base.endswith(f"_{level}"):
            base = base[: -len(f"_{level}")]
            break
    # Strip _sample_N
    parts = base.rsplit("_sample_", 1)
    return parts[0] if len(parts) == 2 else base


# ---------------------------------------------------------------------------
# Evaluation of a single folder
# ---------------------------------------------------------------------------

def evaluate_folder(conn, fp, folder, label, noise_levels):
    files = sorted([
        f for f in os.listdir(folder)
        if f.endswith(".wav") or f.endswith(".mp3")
    ])

    if not files:
        print(f"  No files found in {folder}\n")
        return {}

    print(f"\n{'='*120}")
    print(f"  {label}  ({len(files)} samples)")
    print(f"{'='*120}")
    print(f"  {'Sample':<50} {'Expected':<28} {'Predicted':<28} {'Conf':>7}  Result")
    print(f"  {'-'*118}")

    total = top1 = top3 = no_match = 0
    wrong = []

    for filename in files:
        audio_path = os.path.join(folder, filename)
        expected   = expected_name_from_filename(filename, noise_levels)
        total     += 1

        try:
            ranked, _ = query(conn, fp, audio_path)
        except Exception as e:
            print(f"  ERROR {filename}: {e}")
            total -= 1
            continue

        if not ranked:
            no_match += 1
            print(f"  {filename:<50} {expected:<28} {'—':<28} {'—':>7}  NO MATCH")
            continue

        top1_id, top1_conf = ranked[0]
        predicted  = song_name_from_id(conn, top1_id)
        top3_names = [song_name_from_id(conn, sid) for sid, _ in ranked]

        is_top1 = predicted == expected
        is_top3 = expected in top3_names

        if is_top1:
            top1 += 1
            status = "✓"
        elif is_top3:
            status = "TOP3"
        else:
            status = "✗"
            wrong.append((filename, expected, predicted, top1_conf))

        if is_top3:
            top3 += 1

        print(f"  {filename:<50} {expected:<28} {predicted:<28} {top1_conf:>6.3f}  {status}")

    # Summary
    print(f"\n  Total  : {total}")
    print(f"  Top-1  : {top1} ({100*top1/total:.1f}%)")
    print(f"  Top-3  : {top3} ({100*top3/total:.1f}%)")
    print(f"  No match : {no_match}")
    if wrong:
        print(f"\n  Mismatches:")
        for fname, exp, pred, score in wrong:
            print(f"    {fname}  →  expected '{exp}', got '{pred}' (score {score})")

    return {"label": label, "total": total, "top1": top1, "top3": top3, "no_match": no_match}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isfile(DB_PATH):
        print(f"Database not found: {DB_PATH}  — run fingerprint_songs.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA query_only = ON")
    fp = AudioFingerprinter()

    results = []

    # --- Clean samples ---
    if os.path.isdir(SAMPLES_DIR):
        r = evaluate_folder(conn, fp, SAMPLES_DIR, "CLEAN SAMPLES", NOISE_LEVELS)
        if r:
            results.append(r)

    # --- Noisy samples per noise level ---
    if os.path.isdir(NOISY_DIR):
        for level in NOISE_LEVELS:
            # Collect only files for this noise level
            level_files = [
                f for f in os.listdir(NOISY_DIR)
                if f.endswith(f"_{level}.wav")
            ]
            if not level_files:
                continue

            # Evaluate using a temporary subfolder view (pass the full dir + filter)
            # We do this inline to avoid creating physical subfolders
            print(f"\n{'='*120}")
            print(f"  NOISY SAMPLES — {level.upper()}  ({len(level_files)} samples)")
            print(f"{'='*120}")
            print(f"  {'Sample':<55} {'Expected':<28} {'Predicted':<28} {'Conf':>7}  Result")
            print(f"  {'-'*118}")

            total = top1 = top3 = no_match = 0
            wrong = []

            for filename in sorted(level_files):
                audio_path = os.path.join(NOISY_DIR, filename)
                expected   = expected_name_from_filename(filename, NOISE_LEVELS)
                total     += 1

                try:
                    ranked, _ = query(conn, fp, audio_path)
                except Exception as e:
                    print(f"  ERROR {filename}: {e}")
                    total -= 1
                    continue

                if not ranked:
                    no_match += 1
                    print(f"  {filename:<55} {expected:<28} {'—':<28} {'—':>7}  NO MATCH")
                    continue

                top1_id, top1_conf = ranked[0]
                predicted  = song_name_from_id(conn, top1_id)
                top3_names = [song_name_from_id(conn, sid) for sid, _ in ranked]

                is_top1 = predicted == expected
                is_top3 = expected in top3_names

                if is_top1:
                    top1 += 1
                    status = "✓"
                elif is_top3:
                    status = "TOP3"
                else:
                    status = "✗"
                    wrong.append((filename, expected, predicted, top1_conf))

                if is_top3:
                    top3 += 1

                print(f"  {filename:<55} {expected:<28} {predicted:<28} {top1_conf:>6.3f}  {status}")

            print(f"\n  Total  : {total}")
            print(f"  Top-1  : {top1} ({100*top1/total:.1f}%)" if total else "  Top-1  : —")
            print(f"  Top-3  : {top3} ({100*top3/total:.1f}%)" if total else "  Top-3  : —")
            print(f"  No match : {no_match}")
            if wrong:
                print(f"\n  Mismatches:")
                for fname, exp, pred, score in wrong:
                    print(f"    {fname}  →  expected '{exp}', got '{pred}' (score {score})")

            results.append({"label": f"noisy/{level}", "total": total, "top1": top1, "top3": top3, "no_match": no_match})

    conn.close()

    # ---------------------------------------------------------------------------
    # Overall comparison table
    # ---------------------------------------------------------------------------
    if results:
        print(f"\n\n{'='*70}")
        print("  OVERALL COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Set':<22} {'Total':>6} {'Top-1 %':>9} {'Top-3 %':>9} {'No match':>9}")
        print(f"  {'-'*68}")
        for r in results:
            t = r["total"] or 1
            print(f"  {r['label']:<22} {r['total']:>6} {100*r['top1']/t:>8.1f}% {100*r['top3']/t:>8.1f}% {r['no_match']:>9}")
        print()


if __name__ == "__main__":
    main()
