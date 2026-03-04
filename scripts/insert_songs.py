import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from database import get_connection, create_database, insert_song

SONGS_DIR = os.path.join(os.path.dirname(__file__), "..", "songs")

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}


def main():
    r = get_connection()
    create_database(r)

    song_files = sorted(
        f for f in os.listdir(SONGS_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )

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

