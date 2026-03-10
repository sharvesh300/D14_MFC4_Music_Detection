"""
Match a single audio clip against the fingerprint database.

Usage:
    python scripts/match_song.py <path_to_audio_file> [--phone-mode]

Example:
    python scripts/match_song.py data/songs/samples/Neelothi_sample_2.wav
    python scripts/match_song.py data/songs/test/thee_koluthi.wav --phone-mode
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.fingerprint import AudioFingerprinter
from app.db.redis import get_connection
from app.services.recognition_service import match


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/match_song.py <path_to_audio_file> [--phone-mode]")
        sys.exit(1)

    audio_path = sys.argv[1]
    is_phone_mode = "--phone-mode" in sys.argv

    if not os.path.isfile(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)

    r = get_connection()
    if not r.exists("songs:counter"):
        print(
            "No songs found in Redis — run insert_songs.py and fingerprint_songs.py first."
        )
        sys.exit(1)

    print(f"Query      : {audio_path}")
    print(f"Phone mode : {'ON' if is_phone_mode else 'OFF'}\n")

    fp = AudioFingerprinter()
    response = match(r, fp, audio_path, is_phone_mode=is_phone_mode)

    print(f"Hashes generated : {response.n_hashes}")
    print(f"Matches found    : {len(response.results)}\n")

    if not response.matched:
        print("No match found (confidence below threshold).")
        return

    print(f"  {'Rank':<6} {'Song':<35} {'Confidence':>12}")
    print(f"  {'-'*55}")
    for rank, result in enumerate(response.results, start=1):
        marker = "  <-- BEST MATCH" if rank == 1 else ""
        print(f"  {rank:<6} {result.song_name:<35} {result.confidence:>11.4f}{marker}")


if __name__ == "__main__":
    main()
