"""
Evaluate the audio fingerprinting system against clean and noisy sample clips.

Clean  samples: data/samples/          (e.g. Neelothi_sample_2.wav)
Noisy  samples: data/samples_noisy/    (e.g. Neelothi_sample_2_medium.wav)

Expected song name is derived from the filename:
  "Neelothi_sample_2.wav"         → "Neelothi"
  "Neelothi_sample_2_medium.wav"  → "Neelothi"

Output: per-sample results + aggregate metrics per set (clean / per noise level).

Usage:
    python scripts/evaluate_system.py
"""

import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import SAMPLES_DIR, SAMPLES_NOISY_DIR
from app.core.fingerprint import AudioFingerprinter
from app.db.redis import get_connection
from app.services.recognition_service import match

NOISE_LEVELS = ["light", "medium", "heavy"]


def expected_name_from_filename(filename: str) -> str:
    """Strip '_sample_N[_noiselevel].ext' to recover the original song name."""
    base = os.path.splitext(filename)[0]
    for level in NOISE_LEVELS:
        if base.endswith(f"_{level}"):
            base = base[: -len(f"_{level}")]
            break
    parts = base.rsplit("_sample_", 1)
    return parts[0] if len(parts) == 2 else base


def evaluate_folder(
    r: Any,
    fp: Any,
    folder: str,
    label: str,
    file_list: list[str] | None = None,
) -> dict[str, Any]:
    files = (
        sorted(file_list)
        if file_list is not None
        else sorted(f for f in os.listdir(folder) if f.endswith((".wav", ".mp3")))
    )

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
        expected = expected_name_from_filename(filename)
        total += 1

        try:
            response = match(r, fp, audio_path, top_n=3, min_confidence=0.02)
        except Exception as exc:
            print(f"  ERROR {filename}: {exc}")
            total -= 1
            continue

        if not response.matched:
            no_match += 1
            print(f"  {filename:<50} {expected:<28} {'—':<28} {'—':>7}  NO MATCH")
            continue

        top1_result = response.results[0]
        predicted = top1_result.song_name
        top3_names = [m.song_name for m in response.results]

        is_top1 = predicted == expected
        is_top3 = expected in top3_names

        if is_top1:
            top1 += 1
            status = "✓"
        elif is_top3:
            status = "TOP3"
        else:
            status = "✗"
            wrong.append((filename, expected, predicted, top1_result.confidence))

        if is_top3:
            top3 += 1

        print(
            f"  {filename:<50} {expected:<28} {predicted:<28} {top1_result.confidence:>6.3f}  {status}"
        )

    print(f"\n  Total    : {total}")
    print(f"  Top-1    : {top1} ({100*top1/total:.1f}%)")
    print(f"  Top-3    : {top3} ({100*top3/total:.1f}%)")
    print(f"  No match : {no_match}")
    if wrong:
        print("\n  Mismatches:")
        for fname, exp, pred, score in wrong:
            print(f"    {fname}  →  expected '{exp}', got '{pred}' (score {score:.4f})")

    return {
        "label": label,
        "total": total,
        "top1": top1,
        "top3": top3,
        "no_match": no_match,
    }


def main() -> None:
    r = get_connection()
    if not r.exists("songs:counter"):
        print(
            "No songs found in Redis — run insert_songs.py and fingerprint_songs.py first."
        )
        return

    fp = AudioFingerprinter()
    results = []

    if os.path.isdir(SAMPLES_DIR):
        ev = evaluate_folder(r, fp, SAMPLES_DIR, "CLEAN SAMPLES")
        if ev:
            results.append(ev)

    if os.path.isdir(SAMPLES_NOISY_DIR):
        for level in NOISE_LEVELS:
            level_files = [
                f for f in os.listdir(SAMPLES_NOISY_DIR) if f.endswith(f"_{level}.wav")
            ]
            if not level_files:
                continue

            ev = evaluate_folder(
                r,
                fp,
                SAMPLES_NOISY_DIR,
                f"NOISY SAMPLES — {level.upper()}",
                file_list=level_files,
            )
            if ev:
                results.append(ev)

    # Overall summary
    if results:
        print(f"\n{'='*60}")
        print("  OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Set':<40} {'Top-1':>7}  {'Top-3':>7}")
        print(f"  {'-'*58}")
        for ev in results:
            t = ev["total"]
            print(
                f"  {ev['label']:<40} {100*ev['top1']/t:>6.1f}%  {100*ev['top3']/t:>6.1f}%"
            )


if __name__ == "__main__":
    main()
