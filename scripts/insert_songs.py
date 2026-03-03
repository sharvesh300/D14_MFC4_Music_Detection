import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from database import create_database, insert_song

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "fingerprints.db")
SONGS_DIR = os.path.join(os.path.dirname(__file__), "..", "songs")

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    create_database(DB_PATH)

    song_files = [
        f for f in os.listdir(SONGS_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not song_files:
        print("No audio files found in the songs folder.")
        return

    print(f"Found {len(song_files)} song(s). Inserting into database...\n")

    for filename in sorted(song_files):
        song_name = os.path.splitext(filename)[0]
        song_id = insert_song(DB_PATH, song_name)
        print(f"  [{song_id:>3}] {song_name}")

    print(f"\nDone. {len(song_files)} song(s) inserted into {DB_PATH}")


if __name__ == "__main__":
    main()
