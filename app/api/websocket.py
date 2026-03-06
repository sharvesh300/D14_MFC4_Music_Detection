"""
app/api/websocket.py — WebSocket streaming endpoint
=====================================================

WebSocket /ws/stream

Protocol
--------
Client → Server : raw PCM bytes, float32 little-endian, mono, 8 kHz.
                  Each message = one audio packet (PACKET_DURATION seconds worth).

Server → Client : JSON messages, sent once per detection window step:
    {"matched": true,  "name": "...", "confidence": 0.23, "timestamp": "14:05:01"}
    {"matched": false}
"""

import asyncio
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import MIN_CONFIDENCE, SAMPLE_RATE, WINDOW_SIZE
from app.core.buffer import RingBuffer
from app.core.fingerprint import AudioFingerprinter
from app.core.matcher import fingerprint_only, score_matches
from app.db.fingerprint_repo import match_fingerprints_bulk
from app.db.redis import get_connection
from app.utils.logging import get_logger

ws_router = APIRouter()

_fp    = AudioFingerprinter()
logger = get_logger(__name__)
logger.setLevel(10)  # DEBUG


@ws_router.websocket("/ws/stream")
async def stream_audio(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted from %s", websocket.client)
    r    = get_connection()
    buf  = RingBuffer()
    loop = asyncio.get_event_loop()

    chunk_count  = 0
    window_count = 0

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk_count += 1
            pcm = np.frombuffer(data, dtype=np.float32)
            buf.extend(pcm)

            logger.debug(
                "[chunk %d] %d bytes | %d samples | buffer=%d/%d",
                chunk_count, len(data), len(pcm), buf.buffered, WINDOW_SIZE,
            )

            for window in buf.windows():
                window_count += 1
                logger.debug(
                    "[window %d] fingerprinting %d samples (%.2f s)",
                    window_count, WINDOW_SIZE, WINDOW_SIZE / SAMPLE_RATE,
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
                db_rows     = await loop.run_in_executor(
                    None, match_fingerprints_bulk, r, hash_values
                )

                # CPU scoring — fast, no I/O
                best_id, confidence, offset_bins = score_matches(hashes, hash_to_query_times, db_rows)
                # Convert spectrogram frames → seconds: frames × hop_length / sample_rate
                offset_s = round(offset_bins * _fp.hop_length / _fp.sample_rate, 2)

                logger.info(
                    "[window %d] best_id=%s confidence=%s offset=%.2fs threshold=%s",
                    window_count, best_id, confidence, offset_s, MIN_CONFIDENCE,
                )

                if best_id and confidence >= MIN_CONFIDENCE:
                    raw_name  = r.hget(f"song:{best_id}", "name")
                    song_name = raw_name.decode() if isinstance(raw_name, bytes) else (raw_name or "Unknown")
                    logger.info(
                        "[window %d] MATCH — song_id=%s name=%r confidence=%.4f offset=%.2fs",
                        window_count, best_id, song_name, confidence, offset_s,
                    )
                    await websocket.send_json({
                        "matched":    True,
                        "name":       song_name,
                        "confidence": round(confidence, 4),
                        "offset_s":   offset_s,
                        "timestamp":  time.strftime("%H:%M:%S"),
                    })
                    await websocket.close()
                    return

                await websocket.send_json({"matched": False})

    except WebSocketDisconnect:
        logger.info("Client disconnected — chunks=%d windows=%d", chunk_count, window_count)
    except Exception as exc:
        logger.exception("[chunk %d] Unhandled error: %s", chunk_count, exc)
        try:
            await websocket.send_json({"matched": False, "error": str(exc)})
        except Exception:
            pass




