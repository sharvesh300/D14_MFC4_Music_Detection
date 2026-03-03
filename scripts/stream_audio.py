import sounddevice as sd
import numpy as np
import sqlite3
import threading
import queue
import sys
import os
import time
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fingerprint import AudioFingerprinter
from database import match_fingerprints_bulk

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "fingerprints.db")

SAMPLE_RATE = 8000
CHUNK_DURATION = 2  # seconds
MIN_CONFIDENCE = 0.02

audio_queue = queue.Queue(maxsize=5)
stop_flag = threading.Event()


# ---------------------------------------------------------
# AUDIO PRODUCER THREAD
# ---------------------------------------------------------
def audio_producer():
    blocksize = int(SAMPLE_RATE * CHUNK_DURATION)

    def callback(indata, frames, time_info, status):
        if stop_flag.is_set():
            raise sd.CallbackStop()

        chunk = indata[:, 0].copy()

        try:
            audio_queue.put_nowait(chunk)
        except queue.Full:
            pass  # drop if processing too slow

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=blocksize,
        callback=callback,
    ):
        while not stop_flag.is_set():
            sd.sleep(100)


# ---------------------------------------------------------
# MATCHING CONSUMER THREAD
# ---------------------------------------------------------
def matcher_worker():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA query_only = ON")
    fp = AudioFingerprinter()

    while not stop_flag.is_set():
        try:
            y = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        best_id, conf = match_audio(conn, fp, y)

        if best_id is not None:
            name = song_name_from_id(conn, best_id)
            print()
            print("=" * 50)
            print("  🎵  SONG IDENTIFIED")
            print("=" * 50)
            print(f"  Name       : {name}")
            print(f"  Confidence : {conf:.3f}")
            print(f"  Chunks     : processed until match")
            print("=" * 50)
        else:
            print("...")

    conn.close()


# ---------------------------------------------------------
# MATCH LOGIC
# ---------------------------------------------------------
def match_audio(conn, fp, y):
    S_db = fp.generate_spectrogram(y)
    peaks = fp.find_peaks(S_db)
    hashes = fp.generate_hashes(peaks)

    if not hashes:
        return None, 0

    hash_values = [int(h) for h, _ in hashes]

    hash_to_query_times = defaultdict(list)
    for h, t in hashes:
        hash_to_query_times[int(h)].append(t)

    db_rows = match_fingerprints_bulk(conn, hash_values)

    votes = defaultdict(lambda: defaultdict(int))
    for hash_value, song_id, db_t in db_rows:
        for query_t in hash_to_query_times[hash_value]:
            votes[song_id][db_t - query_t] += 1

    scores = {
        sid: max(buckets.values()) / len(hashes)
        for sid, buckets in votes.items()
    }

    scores = {sid: s for sid, s in scores.items()
              if s >= MIN_CONFIDENCE}

    if not scores:
        return None, 0

    best_id = max(scores, key=scores.get)
    return best_id, scores[best_id]


def song_name_from_id(conn, song_id):
    row = conn.execute(
        "SELECT name FROM songs WHERE id = ?",
        (song_id,)
    ).fetchone()
    return row[0] if row else "Unknown"


# ---------------------------------------------------------
# KEYBOARD LISTENER
# ---------------------------------------------------------
def keyboard_listener():
    print("Press 's' + Enter to stop\n")
    while True:
        cmd = input().strip().lower()
        if cmd == "s":
            stop_flag.set()
            break


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    if not os.path.isfile(DB_PATH):
        print("Database not found.")
        return

    print("🎤 Listening continuously...")
    print("Processing 4-second chunks.\n")

    producer = threading.Thread(target=audio_producer)
    consumer = threading.Thread(target=matcher_worker)
    keyboard = threading.Thread(target=keyboard_listener)

    producer.start()
    consumer.start()
    keyboard.start()

    producer.join()
    consumer.join()
    keyboard.join()

    print("Stopped.")


if __name__ == "__main__":
    main()