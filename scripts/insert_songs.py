import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from database import create_database, insert_song
from extract_metadata import extract_metadata

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "fingerprints.db")
SONGS_DIR = os.path.join(os.path.dirname(__file__), "..", "songs")

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    create_database(DB_PATH)

    song_files = sorted(
        f for f in os.listdir(SONGS_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )

    if not song_files:
        print("No audio files found in the songs folder.")
        return

    print(f"Found {len(song_files)} song(s). Extracting metadata and inserting into database...\n")

    for filename in song_files:
        filepath = os.path.join(SONGS_DIR, filename)
        song_name = os.path.splitext(filename)[0]

        meta = extract_metadata(filepath)
        song_id = insert_song(DB_PATH, song_name, meta=meta)

        title   = meta.get("title") or song_name
        artist  = meta.get("artist") or "(unknown)"
        duration = meta.get("duration_formatted") or "?"
        cover   = meta.get("cover_image_path") or "(none)"
        print(f"  [{song_id:>3}] {title}")
        print(f"         Artist  : {artist}")
        print(f"         Duration: {duration}")
        print(f"         Cover   : {cover}")
        print()

    print(f"Done. {len(song_files)} song(s) inserted into {DB_PATH}")


if __name__ == "__main__":
    main()
