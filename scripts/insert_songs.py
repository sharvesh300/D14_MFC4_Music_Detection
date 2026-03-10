import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import SONGS_DIR
from app.db.redis import get_connection
from app.db.fingerprint_repo import create_database, insert_song
from app.utils.audio import list_audio_files


def main() -> None:
    r = get_connection()
    create_database(r)

    if not os.path.isdir(SONGS_DIR):
        print(f"Songs directory not found: {SONGS_DIR}")
        return

    song_files = list_audio_files(SONGS_DIR)

    if not song_files:
        print("No audio files found in the songs folder.")
        return

    print(f"Found {len(song_files)} song(s). Inserting into Redis...\n")

    inserted = 0
    for filename in song_files:
        song_name = os.path.splitext(filename)[0]

        if r.exists(f"song:name:{song_name}"):
            print(f"  [skip] {song_name}")
            continue

        song_id = insert_song(r, song_name)
        print(f"  [{song_id:>3}] {song_name}")
        inserted += 1

    print(f"\nDone. {inserted} new song(s) inserted.")


if __name__ == "__main__":
    main()
