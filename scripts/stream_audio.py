import queue
import sys
import os
import time
import threading
import argparse
from typing import Any

import sounddevice as sd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.redis import get_connection
from app.core.streaming import audio_producer, matcher_worker


def _resolve_device(mic: str) -> int | None:
    """Return a sounddevice device index for the chosen mic."""
    if mic == "system":
        return None  # sounddevice default
    # Search for AirPods (or any external) by name fragment
    devices = sd.query_devices()
    name_lower = mic.lower()
    for idx, dev in enumerate(devices):
        if name_lower in dev["name"].lower() and dev["max_input_channels"] > 0:
            return idx
    print(
        f"Warning: no input device matching '{mic}' found — falling back to system default."
    )
    return None


def _print_match(result: dict[str, Any]) -> None:
    start_str = time.strftime("%H:%M:%S", time.localtime(result["start_time"]))
    end_str = time.strftime("%H:%M:%S", time.localtime(result["end_time"]))
    d = result["duration_s"]
    dur = f"{int(d // 60):02d}:{int(d % 60):02d}"

    print()
    print("=" * 50)
    print("  🎵  SONG IDENTIFIED")
    print("=" * 50)
    print(f"  Name       : {result['name']}")
    print(f"  Confidence : {result['confidence']:.3f}")
    print(f"  Start time : {start_str}")
    print(f"  End time   : {end_str}")
    print(f"  Duration   : {dur}")
    print("=" * 50)


def _print_no_match() -> None:
    print("...")


def keyboard_listener(stop_flag: threading.Event) -> None:
    print("Press 's' + Enter to stop")
    while True:
        if input().strip().lower() == "s":
            stop_flag.set()
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream audio and identify songs.")
    parser.add_argument(
        "--mic",
        default="system",
        help=(
            "Microphone to use. Use 'system' for the default system mic, "
            "'airpods' for AirPods, or any substring of a device name "
            "(run with --list-devices to see options). Default: system"
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available input devices and exit.",
    )
    args = parser.parse_args()

    if args.list_devices:
        print("Available input devices:")
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                print(f"  [{idx:2d}] {dev['name']}")
        return

    device = _resolve_device(args.mic)
    device_label = f"device [{device}]" if device is not None else "system default"

    r = get_connection()
    if not r.exists("songs:counter"):
        print("No songs in Redis — run insert_songs.py and fingerprint_songs.py first.")
        return

    print(f"🎤 Listening continuously... (mic: {device_label})")
    print("Processing 2-second chunks.")

    audio_queue: queue.Queue[tuple[Any, float]] = queue.Queue(maxsize=5)
    stop_flag = threading.Event()

    producer = threading.Thread(
        target=audio_producer,
        args=(audio_queue, stop_flag),
        kwargs={"device": device},
        daemon=True,
    )
    consumer = threading.Thread(
        target=matcher_worker,
        args=(audio_queue, stop_flag),
        kwargs={"on_match": _print_match, "on_no_match": _print_no_match},
        daemon=True,
    )
    keyboard = threading.Thread(
        target=keyboard_listener, args=(stop_flag,), daemon=True
    )

    producer.start()
    consumer.start()
    keyboard.start()

    producer.join()
    consumer.join()

    print("Stopped.")


if __name__ == "__main__":
    main()
