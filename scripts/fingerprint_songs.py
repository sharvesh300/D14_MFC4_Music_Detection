import os
import sys
import sqlite3

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fingerprint import AudioFingerprinter
from database import create_database, insert_fingerprints_bulk

DB_PATH   = os.path.join(os.path.dirname(__file__), "..", "database", "fingerprints.db")
SONGS_DIR = os.path.join(os.path.dirname(__file__), "..", "songs")


def find_audio_file(songs_dir, song_name):
    """Find the audio file for a given song name (tries common extensions)."""
    for ext in (".mp3", ".wav", ".flac", ".ogg"):
        path = os.path.join(songs_dir, song_name + ext)
        if os.path.isfile(path):
            return path
    return None


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    create_database(DB_PATH)

    # Single connection for the entire run
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    try:
        songs = conn.execute("SELECT id, name FROM songs ORDER BY id").fetchall()

        if not songs:
            print("No songs found in the database. Run insert_songs.py first.")
            return

        print(f"Found {len(songs)} song(s) in database.\n")

        fp = AudioFingerprinter()

        for song_id, song_name in songs:
            print(f"[{song_id:>3}] {song_name}", end=" ... ")

            # Pre-compute check: skip if already fingerprinted
            already = conn.execute(
                "SELECT 1 FROM fingerprints WHERE song_id = ? LIMIT 1", (song_id,)
            ).fetchone()
            if already:
                print("already fingerprinted, skipping.")
                continue

            audio_path = find_audio_file(SONGS_DIR, song_name)
            if audio_path is None:
                print("audio file not found, skipping.")
                continue

            try:
                # --- Normal hashes (broadband) ---
                y, sr       = fp.preprocess(audio_path, is_phone_mode=False)
                S_db        = fp.generate_spectrogram(y)
                peaks       = fp.find_peaks(S_db)
                hashes      = fp.generate_hashes(peaks)

                # --- Phone-mode hashes (bandpass filtered) ---
                y_bp, sr_bp = fp.preprocess(audio_path, is_phone_mode=True)
                S_db_bp     = fp.generate_spectrogram(y_bp)
                peaks_bp    = fp.find_peaks(S_db_bp)
                hashes_bp   = fp.generate_hashes(peaks_bp)

                all_hashes = hashes + hashes_bp
                insert_fingerprints_bulk(conn, song_id, all_hashes)
                print(f"done ({len(hashes)} normal + {len(hashes_bp)} phone-mode hashes stored).")

            except Exception as e:
                print(f"ERROR: {e}")

    finally:
        conn.close()

    print("\nFingerprinting complete.")


if __name__ == "__main__":
    main()
