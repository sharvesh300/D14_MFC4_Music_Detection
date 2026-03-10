"""
app/core/buffer.py — Ring-buffer windowing for streaming audio
==============================================================

RingBuffer
    Pre-allocated fixed-capacity float32 ring buffer.  Accepts raw PCM chunks
    and yields sliding windows of a fixed length with a configurable step.

    Design goals
    ~~~~~~~~~~~~
    * Zero per-window allocation — ``windows()`` yields numpy views into the
      pre-allocated backing array, not copies.
    * O(n) append — a single C-level memcpy per packet via numpy slice assignment.
    * O(1) slide — ``read_pos`` is advanced by an integer add; no data is moved.
    * Bounded compaction — the backing array is only shifted when the tail
      space cannot fit the next incoming packet.  With ``capacity_multiplier=4``
      this happens at most once every ``3 × window_size`` samples.

    Usage
    ~~~~~
    ::

        buf = RingBuffer(window_size=9600, step_size=2400)

        # On each incoming packet
        buf.extend(pcm)               # write samples — O(n), no allocation

        for window in buf.windows():  # iterate ready windows — zero-copy views
            process(window)

    Windowing strategy
    ~~~~~~~~~~~~~~~~~~
    The default is an overlapping **sliding window**::

        window_size=9600, step_size=2400  (WINDOW_DURATION=1.2s, STEP_DURATION=0.3s)

        Window 1: samples   0 –  9599   (0.0 – 1.2 s)
        Window 2: samples 2400 – 11999   (0.3 – 1.5 s)
        Window 3: samples 4800 – 14399   (0.6 – 1.8 s)
        ...

    To change the strategy (tumbling windows, multi-resolution, etc.) only
    this file needs editing — websocket.py is unaffected.
"""

import numpy as np
from typing import Generator

from app.config import WINDOW_SIZE, STEP_SIZE


class RingBuffer:
    """
    Pre-allocated numpy ring buffer with sliding-window iteration.

    Parameters
    ----------
    window_size : int
        Number of samples per window (default: WINDOW_SIZE from config).
    step_size : int
        Number of samples to advance after each window (default: STEP_SIZE).
    capacity_multiplier : int
        Buffer capacity expressed as a multiple of window_size.  4× gives
        ample headroom — compaction triggers at most once every
        (capacity_multiplier - 1) × window_size samples.
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        capacity_multiplier: int = 4,
    ) -> None:
        self.window_size = window_size
        self.step_size = step_size
        self._cap = window_size * capacity_multiplier
        self._buf = np.zeros(self._cap, dtype=np.float32)
        self._write = 0  # next write position
        self._read = 0  # start of oldest unprocessed sample

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def buffered(self) -> int:
        """Number of samples currently available for windowing."""
        return self._write - self._read

    def extend(self, pcm: np.ndarray) -> None:
        """
        Append *pcm* (float32 array) to the buffer.

        Compaction — shifting the live region to the front — is triggered
        only when the remaining tail space cannot fit the incoming chunk.
        Cost is O(buffered) and is bounded by the buffer capacity.
        """
        n = len(pcm)
        if self._write + n > self._cap:
            live = self.buffered
            self._buf[:live] = self._buf[self._read : self._write]
            self._write = live
            self._read = 0

        self._buf[self._write : self._write + n] = pcm
        self._write += n

    def windows(self) -> Generator[np.ndarray, None, None]:
        """
        Yield every ready window as a zero-copy numpy view, advancing by
        *step_size* samples after each one.

        Yields
        ------
        np.ndarray
            Shape (window_size,), dtype float32.  The view is valid until
            the next call to extend() that triggers compaction, so callers
            that need to keep the data must copy it (e.g. ``window.copy()``).
            For read-only processing (fingerprinting) the view is sufficient.
        """
        while self.buffered >= self.window_size:
            yield self._buf[self._read : self._read + self.window_size]
            self._read += self.step_size

    def reset(self) -> None:
        """Discard all buffered samples (e.g. after a connection closes)."""
        self._write = 0
        self._read = 0
