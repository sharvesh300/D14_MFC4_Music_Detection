import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fingerprint import AudioFingerprinter
from database import get_connection, create_database, insert_fingerprints_bulk

SONGS_DIR = os.path.join(os.path.dirname(__file__), "..", "songs")


def find_audio_file(songs_dir, song_name):
    """Find the audio file for a given song name (tries common extensions)."""
    for ext in (".mp3", ".wav", ".flac", ".ogg"):
        path = os.path.join(songs_dir, song_name + ext)
        if os.path.isfile(path):
            return path
    return None


def main():
    r = get_connection()
    create_database(r)  # no-op for Redis, kept for consistency

    n = int(r.get("songs:counter") or 0)
    if n == 0:
        print("No songs found in Redis. Run insert_songs.py first.")
        return

    # Fetch all song names in one pipeline round-trip
    pipe = r.pipeline()
    for i in range(1, n + 1):
        pipe.hget(f"song:{i}", "name")
    names = pipe.execute()
    songs = [(i, name) for i, name in enumerate(names, start=1) if name]

    print(f"Found {len(songs)} song(s) in Redis.\n")

    fp = AudioFingerprinter()

    for song_id, song_name in songs:
        print(f"[{song_id:>3}] {song_name}", end=" ... ")

        # Skip if already fingerprinted
        if r.exists(f"song:{song_id}:fingerprinted"):
            print("already fingerprinted, skipping.")
            continue

        audio_path = find_audio_file(SONGS_DIR, song_name)
        if audio_path is None:
            print("audio file not found, skipping.")
            continue

        try:
            # Load and preprocess the audio once (shared by both modes)
            y_raw, sr = fp.preprocess(audio_path, is_phone_mode=False)

            # --- Normal hashes (broadband) ---
            S_db    = fp.generate_spectrogram(y_raw)
            peaks   = fp.find_peaks(S_db)
            hashes  = fp.generate_hashes(peaks)

            # --- Phone-mode hashes (apply bandpass on the already-loaded signal) ---
            from matcher import bandpass
            y_bp    = bandpass(y_raw, sr)
            S_db_bp = fp.generate_spectrogram(y_bp)
            peaks_bp  = fp.find_peaks(S_db_bp)
            hashes_bp = fp.generate_hashes(peaks_bp)

            all_hashes = hashes + hashes_bp
            insert_fingerprints_bulk(r, song_id, all_hashes)
            r.set(f"song:{song_id}:fingerprinted", "1")
            print(f"done ({len(hashes)} normal + {len(hashes_bp)} phone-mode hashes stored).")

        except Exception as e:
            print(f"ERROR: {e}")

    print("\nFingerprinting complete.")


if __name__ == "__main__":
    main()
