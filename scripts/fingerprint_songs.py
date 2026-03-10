import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import SONGS_DIR
from app.core.fingerprint import AudioFingerprinter
from app.db.redis import get_connection
from app.db.fingerprint_repo import create_database, get_all_songs
from app.services.fingerprint_service import fingerprint_song


def main() -> None:
    r = get_connection()
    create_database(r)

    songs = get_all_songs(r)
    if not songs:
        print("No songs found in Redis. Run insert_songs.py first.")
        return

    print(f"Found {len(songs)} song(s) in Redis.\n")

    fp = AudioFingerprinter()

    for song_id, song_name in songs:
        print(f"[{song_id:>3}] {song_name}", end=" ... ", flush=True)
        result = fingerprint_song(r, song_id, song_name, SONGS_DIR, fp)

        if result["status"] == "skipped":
            print("already fingerprinted, skipping.")
        elif result["status"] == "not_found":
            print("audio file not found, skipping.")
        elif result["status"] == "error":
            print(f"ERROR: {result['error']}")
        else:
            n = result["normal_hashes"]
            p = result["phone_hashes"]
            print(f"done ({n} normal + {p} phone-mode hashes stored).")

    print("\nFingerprinting complete.")


if __name__ == "__main__":
    main()
