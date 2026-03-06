"""
scripts/client.py — WebSocket microphone client
================================================

Captures microphone audio in real time and streams it to /ws/stream.
Disconnects automatically as soon as the server returns a match.

Usage:
    uv run client
    python scripts/client.py
"""

import json
import time

import numpy as np
import sounddevice as sd
import websocket

WS_URL          = "ws://localhost:8000/ws/stream"
SAMPLE_RATE     = 8000
PACKET_DURATION = 0.5                               # seconds per packet
PACKET_SAMPLES  = int(SAMPLE_RATE * PACKET_DURATION)  # 4000 samples


def stream():
    ws = websocket.WebSocket()
    ws.connect(WS_URL)
    print(f"Connected to {WS_URL}")
    print(f"Listening via microphone — {PACKET_DURATION}s packets at {SAMPLE_RATE} Hz")
    print("Press Ctrl+C to stop.\n")

    packets_sent = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=PACKET_SAMPLES,
        ) as stream_in:
            while True:
                pcm, _ = stream_in.read(PACKET_SAMPLES)
                audio  = pcm[:, 0]                     # mono, shape (PACKET_SAMPLES,)

                ws.send_binary(audio.tobytes())
                packets_sent += 1

                ws.settimeout(1.0)
                try:
                    msg  = ws.recv()
                    data = json.loads(msg)
                    if data.get("matched"):
                        offset_s   = data.get("offset_s", 0)
                        offset_min = int(offset_s) // 60
                        offset_sec = offset_s % 60
                        print(
                            f"\nMATCH → {data['name']}"
                            f"  confidence={data['confidence']}"
                            f"  offset={offset_min}m{offset_sec:05.2f}s"
                            f"  {data['timestamp']}"
                        )
                        return   # server already closed; exit cleanly
                except websocket.WebSocketTimeoutException:
                    pass        # no response yet — keep streaming

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        try:
            ws.close()
        except Exception:
            pass
        print(f"Done — {packets_sent} packets sent")


if __name__ == "__main__":
    stream()

