import os
import sys
import csv
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from mutagen import File as MutagenFile
    from mutagen.id3 import ID3NoHeaderError
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False

import librosa

SONGS_DIR = os.path.join(os.path.dirname(__file__), "..", "songs")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "metadata.csv")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "..", "metadata.json")

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}


def format_duration(seconds):
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes}:{secs:02d}"


def extract_metadata(filepath):
    """Extract metadata from an audio file using mutagen (tags) and librosa (duration)."""
    filename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(filename)[0]
    file_size_kb = round(os.path.getsize(filepath) / 1024, 1)

    meta = {
        "filename": filename,
        "title": name_no_ext,
        "artist": "",
        "album": "",
        "genre": "",
        "year": "",
        "track_number": "",
        "duration_seconds": None,
        "duration_formatted": "",
        "sample_rate_hz": None,
        "channels": None,
        "bitrate_kbps": None,
        "file_size_kb": file_size_kb,
    }

    # --- Tag extraction via mutagen ---
    if HAS_MUTAGEN:
        try:
            audio = MutagenFile(filepath, easy=True)
            if audio is not None:
                def tag(key):
                    vals = audio.tags.get(key) if audio.tags else None
                    return str(vals[0]) if vals else ""

                meta["title"] = tag("title") or name_no_ext
                meta["artist"] = tag("artist")
                meta["album"] = tag("album")
                meta["genre"] = tag("genre")
                meta["year"] = tag("date")
                meta["track_number"] = tag("tracknumber")

                if audio.info:
                    meta["duration_seconds"] = round(audio.info.length, 2)
                    meta["duration_formatted"] = format_duration(audio.info.length)
                    if hasattr(audio.info, "sample_rate"):
                        meta["sample_rate_hz"] = audio.info.sample_rate
                    if hasattr(audio.info, "channels"):
                        meta["channels"] = audio.info.channels
                    if hasattr(audio.info, "bitrate"):
                        meta["bitrate_kbps"] = round(audio.info.bitrate / 1000, 1)
        except Exception as e:
            print(f"  [mutagen] Warning for {filename}: {e}")

    # --- Fallback: duration via librosa if not already set ---
    if meta["duration_seconds"] is None:
        try:
            y, sr = librosa.load(filepath, sr=None, mono=False)
            duration = librosa.get_duration(y=y, sr=sr)
            meta["duration_seconds"] = round(duration, 2)
            meta["duration_formatted"] = format_duration(duration)
            meta["sample_rate_hz"] = sr
            meta["channels"] = y.shape[0] if y.ndim > 1 else 1
        except Exception as e:
            print(f"  [librosa] Warning for {filename}: {e}")

    return meta


def main():
    if not HAS_MUTAGEN:
        print("Warning: 'mutagen' is not installed. Tag extraction will be skipped.")
        print("Install it with:  pip install mutagen\n")

    song_files = sorted(
        f for f in os.listdir(SONGS_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )

    if not song_files:
        print("No audio files found in the songs folder.")
        return

    print(f"Extracting metadata for {len(song_files)} song(s)...\n")

    all_metadata = []
    for filename in song_files:
        filepath = os.path.join(SONGS_DIR, filename)
        meta = extract_metadata(filepath)
        all_metadata.append(meta)

        print(f"  {meta['filename']}")
        print(f"    Title    : {meta['title']}")
        print(f"    Artist   : {meta['artist'] or '(unknown)'}")
        print(f"    Album    : {meta['album'] or '(unknown)'}")
        print(f"    Genre    : {meta['genre'] or '(unknown)'}")
        print(f"    Year     : {meta['year'] or '(unknown)'}")
        print(f"    Duration : {meta['duration_formatted'] or '(unknown)'}")
        bitrate_str = f"{meta['bitrate_kbps']} kbps" if meta['bitrate_kbps'] else '(unknown)'
        sample_str = f"{meta['sample_rate_hz']} Hz" if meta['sample_rate_hz'] else '(unknown)'
        print(f"    Bitrate  : {bitrate_str}")
        print(f"    Sample   : {sample_str}")
        print(f"    Size     : {meta['file_size_kb']} KB")
        print()

    # --- Write CSV ---
    fieldnames = list(all_metadata[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metadata)

    # --- Write JSON ---
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print(f"Metadata saved to:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
