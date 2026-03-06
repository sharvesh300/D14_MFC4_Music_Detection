"""
app/core/streaming.py — Live audio producer / consumer engine
=============================================================

Provides three building blocks used by the stream_audio CLI script and the
WebSocket API endpoint:

  audio_producer(audio_queue, stop_flag)
      Captures microphone input in ``CHUNK_DURATION``-second blocks and puts
      ``(pcm_array, absolute_timestamp)`` tuples onto *audio_queue*.

  match_audio(r, fp, y) -> (song_id | None, confidence: float)
      Fingerprints one audio chunk and returns the best matching song_id and
      a normalised confidence score.

  matcher_worker(audio_queue, stop_flag, on_match, on_no_match)
      Consumes chunks from *audio_queue*, calls match_audio, and invokes
      callbacks so the caller decides how to present results.
"""

import threading
import queue
import time

import numpy as np
import sounddevice as sd

from app.config import SAMPLE_RATE, CHUNK_DURATION
from app.core.matcher import match_audio, matcher_worker, start_matcher_worker


# ---------------------------------------------------------------------------
# Audio producer
# ---------------------------------------------------------------------------

def audio_producer(audio_queue: queue.Queue, stop_flag: threading.Event) -> None:
    """
    Capture microphone input and push ``(chunk, timestamp)`` onto *audio_queue*.

    The timestamp is an absolute Unix epoch time computed by adding the ADC
    relative time (``time_info.inputBufferAdcTime``) to the wall-clock instant
    captured just before the InputStream opens.  This gives true audio-hardware
    timing rather than Python scheduling delays.
    """
    blocksize = int(SAMPLE_RATE * CHUNK_DURATION)

    def callback(indata, frames, time_info, status):
        if stop_flag.is_set():
            raise sd.CallbackStop()
        chunk       = indata[:, 0].copy()
        chunk_start = stream_open_wall_time + time_info.inputBufferAdcTime
        try:
            audio_queue.put_nowait((chunk, chunk_start))
        except queue.Full:
            pass  # drop oldest chunk if consumer is lagging

    stream_open_wall_time = time.time()
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=blocksize,
        callback=callback,
    ):
        while not stop_flag.is_set():
            sd.sleep(100)





# ---------------------------------------------------------------------------
# Matching consumer — matcher_worker / start_matcher_worker live in app.core.matcher
# ---------------------------------------------------------------------------
