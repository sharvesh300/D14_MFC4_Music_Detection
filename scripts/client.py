"""
scripts/client.py — WebSocket microphone client
================================================

Captures microphone audio in real time and streams it to /ws/stream.
Disconnects automatically as soon as the server returns a match, or after
LISTEN_TIMEOUT seconds without a match.

Usage:
    uv run client
    python scripts/client.py
"""

import json
import time

import sounddevice as sd
import websocket

WS_URL = "ws://localhost:8000/ws/stream"
SAMPLE_RATE = 8000
PACKET_DURATION = 0.5  # seconds per packet
PACKET_SAMPLES = int(SAMPLE_RATE * PACKET_DURATION)  # 4000 samples
LISTEN_TIMEOUT = 30  # seconds before giving up


def _status(msg: str) -> None:
    """Overwrite the current terminal line in place."""
    print(f"\r{msg:<60}", end="", flush=True)


def stream() -> None:
    ws = websocket.WebSocket()
    ws.connect(WS_URL)  # type: ignore[no-untyped-call]
    print(f"Connected to {WS_URL}")
    print(f"Listening via microphone — {PACKET_DURATION}s packets at {SAMPLE_RATE} Hz")
    print(f"Will give up after {LISTEN_TIMEOUT}s without a match.\n")

    packets_sent = 0
    start_time = time.monotonic()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=PACKET_SAMPLES,
        ) as stream_in:
            while True:
                elapsed = time.monotonic() - start_time
                if elapsed >= LISTEN_TIMEOUT:
                    print(f"\nNo match found after {LISTEN_TIMEOUT}s — giving up.")
                    return

                pcm, _ = stream_in.read(PACKET_SAMPLES)
                audio = pcm[:, 0]  # mono, shape (PACKET_SAMPLES,)

                ws.send_binary(audio.tobytes())  # int16 LE bytes
                packets_sent += 1

                ws.settimeout(1.0)
                try:
                    msg = ws.recv()
                    data = json.loads(msg)
                    if data.get("matched"):
                        offset_s = data.get("offset_s", 0)
                        offset_min = int(offset_s) // 60
                        offset_sec = offset_s % 60
                        print(
                            f"\nMATCH → {data['name']}"
                            f"  confidence={data['confidence']}"
                            f"  offset={offset_min}m{offset_sec:05.2f}s"
                            f"  {data['timestamp']}"
                        )
                        return  # server already closed; exit cleanly
                    else:
                        # No match yet — overwrite status line so terminal stays clean
                        remaining = max(0, LISTEN_TIMEOUT - elapsed)
                        _status(
                            f"Listening… {elapsed:4.1f}s elapsed  ({remaining:.0f}s remaining)"
                        )
                except websocket.WebSocketTimeoutException:
                    pass  # no response yet — keep streaming

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        try:
            ws.close()
        except Exception:
            pass
        print(f"\nDone — {packets_sent} packets sent")


if __name__ == "__main__":
    stream()
