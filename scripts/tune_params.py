"""
Grid-search tuning for AudioFingerprinter parameters.

Tries every combination of fan_value, delta_t_max, freq_bin_size, and
min_confidence against the clean + noisy probe clips in songs/samples/ and
songs/samples_noisy/, then reports the best parameter set for each probe split.

Usage
-----
    python scripts/tune_params.py                          # default grid
    python scripts/tune_params.py --split clean            # clean only
    python scripts/tune_params.py --split noisy            # noisy only
    python scripts/tune_params.py --split light medium     # specific levels
    python scripts/tune_params.py --max-probes 30          # limit probes (faster)
    python scripts/tune_params.py --top 5                  # show top-5 combos

Grid defaults (edit the constants below to widen / narrow the search):
    FAN_VALUES          = (5, 10, 15, 20)
    DELTA_T_MAX_VALUES  = (100, 200, 300)
    FREQ_BIN_SIZE_VALUES= (5, 10)
    MIN_CONF_VALUES     = (0.01, 0.02, 0.05)
"""

import argparse
import itertools
import os
import sys
from collections import defaultdict
from typing import Any, cast

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.core.fingerprint import AudioFingerprinter
from app.db.redis import get_connection
from app.db.fingerprint_repo import match_fingerprints_bulk

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from app.config import SAMPLES_DIR, SAMPLES_NOISY_DIR as NOISY_DIR

NOISE_LEVELS = ["light", "medium", "heavy"]

# ---------------------------------------------------------------------------
# Default grid (edit here or pass CLI arguments)
# ---------------------------------------------------------------------------
FAN_VALUES = (5, 10, 15, 20)
DELTA_T_MAX_VALUES = (100, 200, 300)
FREQ_BIN_SIZE_VALUES = (5, 10)
MIN_CONF_VALUES = (0.01, 0.02, 0.05)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def expected_name_from_filename(filename: str) -> str:
    """Strip '_sample_N[_noiselevel].ext' to recover the original song name."""
    base = os.path.splitext(filename)[0]
    for level in NOISE_LEVELS:
        if base.endswith(f"_{level}"):
            base = base[: -len(f"_{level}")]
            break
    parts = base.rsplit("_sample_", 1)
    return parts[0] if len(parts) == 2 else base


def song_name_from_id(r: Any, song_id: int) -> str:
    return r.hget(f"song:{song_id}", "name") or "Unknown"


def collect_probes(
    folder: str,
    suffix_filter: str | None = None,
    max_probes: int | None = None,
) -> list[tuple[str, str]]:
    """
    Return list of (audio_path, expected_song_name) from a folder.
    suffix_filter: if set, only include files whose base ends with this suffix
                   (e.g. '_light' for light-noise files).
    """
    if not os.path.isdir(folder):
        return []
    files = sorted(
        f for f in os.listdir(folder) if f.endswith(".wav") or f.endswith(".mp3")
    )
    if suffix_filter:
        files = [f for f in files if os.path.splitext(f)[0].endswith(suffix_filter)]
    if max_probes:
        files = files[:max_probes]
    return [(os.path.join(folder, f), expected_name_from_filename(f)) for f in files]


# ---------------------------------------------------------------------------
# Core: fingerprint one clip with given params
# ---------------------------------------------------------------------------


def fingerprint_clip(
    fp: Any, audio_path: str, fan_value: int, delta_t_max: int, freq_bin_size: int
) -> list[Any]:
    y, _ = fp.preprocess(audio_path)
    S_db = fp.generate_spectrogram(y)
    peaks = fp.find_peaks(S_db)
    return cast(
        list[Any],
        fp.generate_hashes(
            peaks,
            fan_value=fan_value,
            delta_t_max=delta_t_max,
            freq_bin_size=freq_bin_size,
        ),
    )


# ---------------------------------------------------------------------------
# Core: match hashes against DB with offset-alignment voting
# ---------------------------------------------------------------------------


def match_hashes(
    r: Any,
    hashes: list[tuple[int, int]],
    min_confidence: float,
    top_n: int = 3,
) -> list[tuple[int, float]]:
    """
    Return ranked list of (song_name, confidence) using offset-alignment voting.
    """
    n_hashes = len(hashes)
    if n_hashes == 0:
        return []

    hash_values = [int(h) for h, _ in hashes]
    hash_to_query_times = defaultdict(list)
    for h, t in hashes:
        hash_to_query_times[int(h)].append(t)

    db_rows = match_fingerprints_bulk(r, hash_values)

    votes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for hash_value, song_id, db_t in db_rows:
        for query_t in hash_to_query_times[hash_value]:
            votes[song_id][db_t - query_t] += 1

    scores = {sid: max(buckets.values()) / n_hashes for sid, buckets in votes.items()}
    scores = {sid: s for sid, s in scores.items() if s >= min_confidence}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(song_id, conf) for song_id, conf in ranked]


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def run_grid_search(
    r: Any,
    probes: list[tuple[str, str]],
    label: str,
    fan_values: list[int],
    delta_t_max_values: list[int],
    freq_bin_size_values: list[int],
    min_conf_values: list[float],
    top_n_results: int,
) -> list[dict[str, Any]]:
    fp = AudioFingerprinter(sample_rate=8000)
    grid = list(
        itertools.product(
            fan_values, delta_t_max_values, freq_bin_size_values, min_conf_values
        )
    )

    print(f"\n{'='*80}")
    print(f"  Tuning on: {label}  ({len(probes)} probes, {len(grid)} combinations)")
    print(f"{'='*80}")
    print(
        f"  {'fan':>4} {'dt_max':>7} {'fbsz':>5} {'min_conf':>9}  "
        f"{'Top-1':>7}  {'Top-3':>7}  {'No-match':>9}"
    )
    print(f"  {'-'*75}")

    results = []

    for fan_value, delta_t_max, freq_bin_size, min_conf in grid:
        top1 = top3 = no_match = total = 0

        for audio_path, expected in probes:
            try:
                hashes = fingerprint_clip(
                    fp, audio_path, fan_value, delta_t_max, freq_bin_size
                )
                ranked = match_hashes(r, hashes, min_conf)
            except Exception as exc:
                print(f"    ERROR {os.path.basename(audio_path)}: {exc}")
                continue

            total += 1
            if not ranked:
                no_match += 1
                continue

            top_names = [song_name_from_id(r, sid) for sid, _ in ranked]
            if top_names[0] == expected:
                top1 += 1
            if expected in top_names:
                top3 += 1

        top1_acc = top1 / total if total else 0.0
        top3_acc = top3 / total if total else 0.0

        results.append(
            {
                "fan_value": fan_value,
                "delta_t_max": delta_t_max,
                "freq_bin_size": freq_bin_size,
                "min_confidence": min_conf,
                "top1_acc": top1_acc,
                "top3_acc": top3_acc,
                "no_match": no_match,
                "total": total,
            }
        )
        print(
            f"  {fan_value:>4} {delta_t_max:>7} {freq_bin_size:>5} {min_conf:>9.3f}  "
            f"{top1_acc:>6.1%}  {top3_acc:>6.1%}  {no_match:>9}"
        )

    # Sort: best top1_acc first, tiebreak top3_acc
    results.sort(key=lambda x: (x["top1_acc"], x["top3_acc"]), reverse=True)

    print(f"\n  Top-{top_n_results} parameter combinations for [{label}]:")
    print(
        f"  {'Rank':>5}  {'fan':>4} {'dt_max':>7} {'fbsz':>5} {'min_conf':>9}  "
        f"{'Top-1':>7}  {'Top-3':>7}"
    )
    print(f"  {'-'*65}")
    for rank, row in enumerate(results[:top_n_results], 1):
        print(
            f"  {rank:>5}  {row['fan_value']:>4} {row['delta_t_max']:>7} "
            f"{row['freq_bin_size']:>5} {row['min_confidence']:>9.3f}  "
            f"{row['top1_acc']:>6.1%}  {row['top3_acc']:>6.1%}"
        )

    if results:
        best = results[0]
        print(f"\n  >>> Best for [{label}]:")
        print(f"        fan_value      = {best['fan_value']}")
        print(f"        delta_t_max    = {best['delta_t_max']}")
        print(f"        freq_bin_size  = {best['freq_bin_size']}")
        print(f"        min_confidence = {best['min_confidence']}")
        print(f"        Top-1 acc      = {best['top1_acc']:.1%}")
        print(f"        Top-3 acc      = {best['top3_acc']:.1%}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune fingerprinter parameters via grid search."
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["clean", "light", "medium", "heavy"],
        help="Which probe splits to evaluate. Options: clean light medium heavy",
    )
    parser.add_argument(
        "--max-probes",
        type=int,
        default=None,
        help="Max number of probe clips per split (default: all).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many top combos to display per split (default: 5).",
    )
    # Grid overrides
    parser.add_argument("--fan-values", nargs="+", type=int, default=list(FAN_VALUES))
    parser.add_argument(
        "--delta-t-max", nargs="+", type=int, default=list(DELTA_T_MAX_VALUES)
    )
    parser.add_argument(
        "--freq-bin-size", nargs="+", type=int, default=list(FREQ_BIN_SIZE_VALUES)
    )
    parser.add_argument(
        "--min-conf", nargs="+", type=float, default=list(MIN_CONF_VALUES)
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    r = get_connection()
    if not r.exists("songs:counter"):
        print("No songs found in Redis.\nRun scripts/fingerprint_songs.py first.")
        sys.exit(1)

    all_best = {}

    for split in args.split:
        if split == "clean":
            probes = collect_probes(SAMPLES_DIR, max_probes=args.max_probes)
            label = "CLEAN"
        else:
            probes = collect_probes(
                NOISY_DIR, suffix_filter=f"_{split}", max_probes=args.max_probes
            )
            label = f"NOISY / {split.upper()}"

        if not probes:
            print(f"\nNo probes found for split '{split}', skipping.")
            continue

        results = run_grid_search(
            r,
            probes,
            label,
            fan_values=args.fan_values,
            delta_t_max_values=args.delta_t_max,
            freq_bin_size_values=args.freq_bin_size,
            min_conf_values=args.min_conf,
            top_n_results=args.top,
        )
        if results:
            all_best[split] = results[0]

    if len(all_best) > 1:
        print(f"\n\n{'='*80}")
        print("  SUMMARY — best params per split")
        print(f"{'='*80}")
        print(
            f"  {'Split':<18} {'fan':>4} {'dt_max':>7} {'fbsz':>5} "
            f"{'min_conf':>9}  {'Top-1':>7}  {'Top-3':>7}"
        )
        print(f"  {'-'*72}")
        for split, row in all_best.items():
            print(
                f"  {split:<18} {row['fan_value']:>4} {row['delta_t_max']:>7} "
                f"{row['freq_bin_size']:>5} {row['min_confidence']:>9.3f}  "
                f"{row['top1_acc']:>6.1%}  {row['top3_acc']:>6.1%}"
            )
        print()


if __name__ == "__main__":
    main()
