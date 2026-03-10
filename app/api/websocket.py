"""
app/api/websocket.py — Real-time WebSocket streaming endpoint
=============================================================

Endpoint
--------
WS  /ws/stream

Wire protocol
-------------
Client → Server : Raw PCM binary frames, int16 little-endian, mono, 8 kHz.
                  Each frame covers PACKET_DURATION seconds (default 0.5 s,
                  i.e. 4 000 samples = 8 000 bytes).

Server → Client : UTF-8 JSON, one message per processed window.

    No match yet::

        {"matched": false}

    Timeout (30 s elapsed, no confirmed song)::

        {"matched": false, "reason": "timeout"}

    Confirmed match::

        {
          "matched":    true,
          "name":       "Song Title",
          "confidence": 0.23,        # average over confirming windows
          "offset_s":   34.56,       # playback position in the original song
          "timestamp":  "14:05:01"   # wall-clock time of confirmation
        }

Pipeline (per window)
---------------------
1. ``RingBuffer`` accumulates int16 PCM packets and emits sliding windows
   of WINDOW_SIZE samples (1.2 s) with a STEP_SIZE step (0.3 s).

2. ``fingerprint_only`` (CPU, thread executor)
   STFT → log-magnitude spectrogram → peak detection → 64-bit landmark hashes.

3. ``match_fingerprints_bulk`` (I/O, thread executor)
   Single Redis pipeline round-trip — fetches all ``fp:{hash}`` lists.

4. ``score_matches`` (CPU, inline)
   Offset-alignment voting → best_id, confidence, offset_bins.

5. ``ConsensusVoter`` (stateful, per-connection)
   Requires the same song to win *threshold* (default 3) successive windows
   before a result is considered genuine.  Suppresses single-window false
   positives from noise bursts.

6. ``SongTracker`` (stateful, per-connection)
   Holds the last confirmed song for HOLD_TIME seconds (default 4 s) even
   when intervening windows yield no consensus.  Prevents brief dropout
   windows from breaking a valid match chain.

7. 30-second hard timeout — if no match is confirmed within MAX_STREAM_TIME
   the server sends a timeout JSON and closes the connection.
"""

import asyncio
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import MIN_CONFIDENCE, SAMPLE_RATE, WINDOW_SIZE
from app.core.buffer import RingBuffer
from app.core.fingerprint import AudioFingerprinter
from app.core.matcher import (
    fingerprint_only,
    score_matches,
    ConsensusVoter,
    SongTracker,
)
from app.db.fingerprint_repo import match_fingerprints_bulk
from app.db.redis import get_connection
from app.utils.logging import get_logger

ws_router = APIRouter()

_fp = AudioFingerprinter()
logger = get_logger(__name__)
logger.setLevel(10)  # DEBUG


@ws_router.websocket("/ws/stream")
async def stream_audio(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("WebSocket connection accepted from %s", websocket.client)
    r = get_connection()
    buf = RingBuffer()
    voter = ConsensusVoter(threshold=3)
    tracker = SongTracker(hold_time=4.0)
    loop = asyncio.get_event_loop()

    MAX_STREAM_TIME = 30
    start_time = time.time()
    chunk_count = 0
    window_count = 0

    try:
        while True:
            if time.time() - start_time > MAX_STREAM_TIME:
                logger.info(
                    "Stream timeout after %ds — no match found", MAX_STREAM_TIME
                )
                await websocket.send_json({"matched": False, "reason": "timeout"})
                await websocket.close()
                return

            data = await websocket.receive_bytes()
            chunk_count += 1
            pcm_int16 = np.frombuffer(data, dtype=np.int16)
            pcm = pcm_int16.astype(np.float32) / 32768.0
            buf.extend(pcm)
            logger.debug(
                "[chunk %d] %d bytes | %d samples | buffer=%d/%d",
                chunk_count,
                len(data),
                len(pcm),
                buf.buffered,
                WINDOW_SIZE,
            )

            for window in buf.windows():
                window_count += 1
                logger.debug(
                    "[window %d] fingerprinting %d samples (%.2f s)",
                    window_count,
                    WINDOW_SIZE,
                    WINDOW_SIZE / SAMPLE_RATE,
                )

                # CPU fingerprinting — off event loop
                hashes, hash_to_query_times = await loop.run_in_executor(
                    None, fingerprint_only, _fp, window
                )

                if not hashes:
                    logger.debug("[window %d] no hashes generated", window_count)
                    await websocket.send_json({"matched": False})
                    continue

                # Redis lookup — sync client in executor
                hash_values = [int(h) for h, _ in hashes]
                db_rows = await loop.run_in_executor(
                    None, match_fingerprints_bulk, r, hash_values
                )

                # CPU scoring — fast, no I/O
                best_id, confidence, offset_bins = score_matches(
                    hashes, hash_to_query_times, db_rows
                )

                logger.info(
                    "[window %d] best_id=%s confidence=%s votes=%s threshold=%s",
                    window_count,
                    best_id,
                    confidence,
                    voter._counts.get(best_id, 0) + (1 if best_id is not None else 0),  # type: ignore[arg-type]
                    MIN_CONFIDENCE,
                )

                # Consensus gate — require threshold windows to agree before confirming
                confirmed_id, confirmed_conf, confirmed_offset = voter.vote(
                    best_id, confidence, offset_bins
                )
                offset_s = round(confirmed_offset * _fp.hop_length / _fp.sample_rate, 2)

                # Resolve name only on a fresh consensus hit (avoids a Redis call every window)
                name = None
                if confirmed_id is not None:
                    raw_name: str | None = r.hget(f"song:{confirmed_id}", "name")
                    name = raw_name if raw_name is not None else "Unknown"

                # Temporal smoothing — hold the song across brief dropout windows
                active_id, active_name, active_conf, active_off_s = tracker.update(
                    confirmed_id, confirmed_conf, offset_s, name
                )

                if active_id is not None and active_conf >= MIN_CONFIDENCE:
                    logger.info(
                        "[window %d] MATCH — song_id=%s name=%r confidence=%.4f offset=%.2fs",
                        window_count,
                        active_id,
                        active_name,
                        active_conf,
                        active_off_s,
                    )
                    await websocket.send_json(
                        {
                            "matched": True,
                            "name": active_name,
                            "confidence": round(active_conf, 4),
                            "offset_s": active_off_s,
                            "timestamp": time.strftime("%H:%M:%S"),
                        }
                    )
                    await websocket.close()
                    return

                await websocket.send_json({"matched": False})

    except WebSocketDisconnect:
        logger.info(
            "Client disconnected — chunks=%d windows=%d", chunk_count, window_count
        )
    except Exception as exc:
        logger.exception("[chunk %d] Unhandled error: %s", chunk_count, exc)
        try:
            await websocket.send_json({"matched": False, "error": str(exc)})
        except Exception:
            pass
