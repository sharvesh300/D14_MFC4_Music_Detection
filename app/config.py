"""
app/config.py — Centralised environment settings
=================================================
All tuneable knobs live here. Override any value with an environment variable.
"""

import os

# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB   = int(os.environ.get("REDIS_DB",   0))

# ---------------------------------------------------------------------------
# Audio / fingerprinting
# ---------------------------------------------------------------------------
SAMPLE_RATE    = int(os.environ.get("SAMPLE_RATE",    8000))
CHUNK_DURATION = float(os.environ.get("CHUNK_DURATION", 2.0))   # seconds
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", 0.1))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
SONGS_DIR       = os.path.join(DATA_DIR, "songs")
SAMPLES_DIR     = os.path.join(DATA_DIR, "samples")
SAMPLES_NOISY_DIR = os.path.join(DATA_DIR, "samples_noisy")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}

# ---------------------------------------------------------------------------
# Streaming matching
# ---------------------------------------------------------------------------
SAMPLE_RATE     = 8000
PACKET_DURATION = 0.5
WINDOW_DURATION = 1.2  # reduced from 2.0 — first window ready 0.8 s sooner
STEP_DURATION   = 0.3  # reduced from 0.5 — denser overlap, faster detection

PACKET_SIZE = int(SAMPLE_RATE * PACKET_DURATION)   # 4000
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)   # 9600  (≈1.2 s)
STEP_SIZE   = int(SAMPLE_RATE * STEP_DURATION)     # 2400  (≈0.3 s)
