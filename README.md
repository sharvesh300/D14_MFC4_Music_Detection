# Shazam-Style Audio Fingerprinting System

A Python implementation of Shazam-style audio fingerprinting for song identification from short clips (~4 seconds). Uses spectral peak detection, anchor-target constellation mapping, and offset-alignment voting for robust matching — even under noise or telephone-quality audio.

## 📋 Table of Contents

- [Overview](#-overview)
- [Why Preprocessing?](#-why-does-audio-need-preprocessing)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Pipeline](#-pipeline)
- [Scripts Reference](#-scripts-reference)
- [Configuration](#-configuration)
- [Evaluation](#-evaluation)
- [Robustness](#-robustness-features)
- [Docker Setup](#-docker-setup)
- [References](#-references)

## 🎯 Overview

This system implements industry-standard Shazam-style audio fingerprinting featuring:

- **Audio Preprocessing**: Mono conversion, resampling to 8000 Hz, DC removal, amplitude normalisation, optional bandpass filtering (phone mode)
- **Spectral Analysis**: STFT with configurable FFT size and hop length
- **Peak Detection**: Local maxima via `maximum_filter` with percentile-based amplitude thresholding
- **64-bit Fingerprint Hashing**: Anchor-target pairing packed as `[f1·9-bit][f2·9-bit][Δt·8-bit]`
- **Dual Hash Indexing**: Each song is stored with both broadband and phone-mode (300–3400 Hz bandpass) hashes for robust telephone/GSM matching
- **SQLite Database**: Songs table + fingerprints table with `idx_hash` index for fast lookups
- **Offset-Alignment Voting**: Time-invariant matching robust to recording offset
- **Score Normalisation & Thresholding**: Confidence scores (0–1) with rejection below threshold
- **Live Microphone Matching**: Real-time stream matching via `stream_audio.py`
- **Parameter Tuning**: Grid-search script to find optimal fingerprinter parameters

---

## 🔥 Why Does Audio Need Preprocessing?

Real-world audio recordings are rarely clean. Noise and distortion are introduced at multiple stages of the recording chain. Common sources include:

| Source | Description |
|---|---|
| **Cheap Microphones** | Low-quality capsules introduce self-noise and frequency coloration |
| **ADC Imperfections** | Quantization noise, clipping, and non-linearity errors during analog → digital conversion |
| **Recording Devices** | Power supply hum (50/60 Hz), poor shielding bleed into the signal |
| **Compression Artifacts** | Lossy codecs (MP3, AAC) discard high-frequency detail and introduce ringing artifacts |
| **Telephone / GSM Channels** | Band-limited to 300–3400 Hz, heavily compressed, low bitrate |

### 🛠️ How Preprocessing Fixes This

1. **Mono Conversion** — Collapses stereo to a single channel; eliminates channel imbalance.
2. **Resampling to 8000 Hz** — Standardises the sample rate; 8 kHz captures up to 4 kHz — sufficient for telephone-quality fingerprinting.
3. **Amplitude Normalisation** — Scales peak amplitude to ±1; removes recording level differences.
4. **DC Removal** — Subtracts the mean; eliminates DC offset from cheap ADCs.
5. **Pre-emphasis** — Boosts high-frequency energy to compensate for spectral roll-off.
6. **Bandpass Filter (phone mode)** — 4th-order Butterworth 300–3400 Hz; simulates telephone/GSM channel conditions.
7. **Percentile Peak Thresholding** — Retains only the top 15% of spectral energy peaks as landmarks.

---

## 📁 Project Structure

```
music-detection/
├── fingerprint.py             # AudioFingerprinter class (preprocess, spectrogram, peaks, hashes, tune_parameters)
├── matcher.py                 # bandpass() filter + match_sample()
├── database.py                # SQLite helpers (create, insert, bulk match)
├── experiment.ipynb           # Interactive exploration notebook
├── toy_example.ipynb          # Minimal worked example
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── README.md                  # This file
├── scripts/
│   ├── insert_songs.py        # Register songs from songs/ into the DB
│   ├── fingerprint_songs.py   # Compute & store fingerprints (broadband + phone-mode)
│   ├── create_samples.py      # Generate clean + Gaussian-noisy test clips
│   ├── generate_audio_variations.sh  # GSM & low-bitrate MP3 conversion of samples
│   ├── match_song.py          # Match a single audio file (supports --phone-mode)
│   ├── evaluate.py            # Batch evaluation across clean and noisy samples
│   ├── tune_params.py         # Grid-search parameter tuning
│   ├── stream_audio.py        # Real-time microphone matching
│   ├── process_variation.sh   # Process a single audio variation
│   └── drop_tables.py         # Drop all DB tables (reset)
├── songs/                     # Full-length audio files (MP3/WAV/FLAC/OGG)
│   ├── SongName.mp3
│   ├── samples/               # Clean 4s WAV clips
│   ├── samples_noisy/         # Gaussian-noisy variants (light / medium / heavy)
│   ├── gsm/
│   │   ├── samples/           # GSM-encoded clean clips
│   │   └── samples_noisy/     # GSM-encoded noisy clips
│   ├── low_mp3/
│   │   ├── samples/           # 64 kbps MP3 clean clips
│   │   └── samples_noisy/     # 64 kbps MP3 noisy clips
│   └── test/                  # Manual test files
└── database/
    └── fingerprints.db        # SQLite database (auto-created, git-ignored)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (or Docker)
- **ffmpeg** — required for GSM/MP3 conversion (`brew install ffmpeg`)
- Audio files placed in `songs/` (MP3, WAV, FLAC, OGG)

### Local Setup

```bash
# Create and activate virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Full Pipeline (run in order)

```bash
# 1. Register song names in the database
python scripts/insert_songs.py

# 2. Compute and store fingerprints (broadband + phone-mode hashes per song)
python scripts/fingerprint_songs.py

# 3. Generate test clips (clean + Gaussian-noisy variants)
python scripts/create_samples.py

# 4. (Optional) Generate GSM and low-bitrate MP3 variations
bash scripts/generate_audio_variations.sh

# 5. Match a single clip
python scripts/match_song.py songs/samples/SongName_sample_2.wav

# 5a. Match a telephone/GSM recording
python scripts/match_song.py songs/gsm/samples/SongName_sample_2.gsm --phone-mode

# 6. Evaluate the whole system
python scripts/evaluate.py

# 7. (Optional) Tune parameters via grid search
python scripts/tune_params.py
```

---

## 🏗️ System Architecture

The system is built around three core Python modules. Scripts in `scripts/` are thin orchestration wrappers that wire these modules together.

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `fingerprint.py` | `AudioFingerprinter` class — preprocess audio, build spectrogram, detect spectral peaks, produce 64-bit landmark hashes |
| `matcher.py` | `bandpass()` filter (phone mode) + `match_sample()` — hash lookup in SQLite and offset-alignment voting to identify a song |
| `database.py` | SQLite helpers — create tables, insert songs/fingerprints, bulk hash queries |

### Two Operating Paths

```
 INDEXING (run once, offline)
 ──────────────────────────────────────────────────────────
  songs/*.mp3/wav
       │
       ▼
  fingerprint.py         preprocess → spectrogram → peaks → hashes
  (AudioFingerprinter)   run twice per song: broadband + phone-mode
       │
       ▼
  database.py            INSERT INTO fingerprints (song_id, hash_value, time_offset)
       │
       ▼
  database/fingerprints.db   (idx_hash index for fast lookup)


 QUERYING (real-time / batch)
 ──────────────────────────────────────────────────────────
  audio clip / microphone
       │
       ▼
  fingerprint.py         same pipeline as indexing
       │  hashes
       ▼
  matcher.py             batch SELECT WHERE hash_value IN (...)
                         offset-alignment voting → confidence score
       │
       ▼
  Ranked results         [(song_name, confidence), ...]
```

### Detailed Data Flow

```
Audio File / Mic Stream
        │
        ▼
┌─────────────────────────┐
│  preprocess()           │  mono · resample 8 kHz · normalise · DC-remove
│  fingerprint.py         │  · pre-emphasis · [phone mode: bandpass 300–3400 Hz]
└────────────┬────────────┘
             │  waveform y
             ▼
┌─────────────────────────┐
│  generate_spectrogram() │  STFT  →  |magnitude|  →  log dB
│  fingerprint.py         │
└────────────┬────────────┘
             │  S_db  (freq × time matrix)
             ▼
┌─────────────────────────┐
│  find_peaks()           │  maximum_filter (20×20 neighbourhood)
│  fingerprint.py         │  → local maxima · percentile threshold (top 15%)
└────────────┬────────────┘
             │  peaks  [(freq_idx, time_idx), ...]
             ▼
┌─────────────────────────┐
│  generate_hashes()      │  anchor-target pairing  →  packed 64-bit hash
│  fingerprint.py         │  hash = [f1 · 9-bit][f2 · 9-bit][Δt · 8-bit]
└────────────┬────────────┘
             │  hashes  [(hash_value, time_offset), ...]
             │
      ┌──────┴──────┐
      │             │
  (indexing)    (querying)
      │             │
      ▼             ▼
 database.py   match_sample()          batch IN(...) query
  INSERT         matcher.py        →   votes[song_id][db_t − query_t] += 1
                                   →   confidence = peak_votes / n_hashes
                                   →   discard if confidence < MIN_CONFIDENCE
                                   →   ranked [(song_name, confidence), ...]
```

---

## 📜 Scripts Reference

### `scripts/insert_songs.py`
Scans `songs/` for audio files and inserts each song name into the `songs` table. Safe to re-run — skips duplicates.

```bash
python scripts/insert_songs.py
```

---

### `scripts/fingerprint_songs.py`
For each song in the DB, computes **two sets of hashes** and stores both:
- **Broadband hashes** — standard preprocessing
- **Phone-mode hashes** — with 300–3400 Hz bandpass (simulates telephone/GSM)

Skips songs already fingerprinted.

```bash
python scripts/fingerprint_songs.py
```

---

### `scripts/create_samples.py`
Generates short test clips from each song in `songs/`. The song is divided into `N_SAMPLES` equal segments with a random offset within each, ensuring clips are spread across the full duration.

| Setting | Value |
|---|---|
| Clip duration | 4 seconds |
| Clips per song | 10 |
| Sample rate | 8000 Hz |
| Noise type | Gaussian noise |
| Noise levels | light (SNR +10 dB), medium (SNR 0 dB), heavy (SNR −5 dB) |

```bash
python scripts/create_samples.py
```

Output:
- `songs/samples/SongName_sample_N.wav` — clean clips
- `songs/samples_noisy/SongName_sample_N_light.wav` — light noise
- `songs/samples_noisy/SongName_sample_N_medium.wav` — medium noise
- `songs/samples_noisy/SongName_sample_N_heavy.wav` — heavy noise

---

### `scripts/generate_audio_variations.sh`
Runs `create_samples.py` first, then converts all clips to GSM and low-bitrate MP3 formats using ffmpeg.

```bash
bash scripts/generate_audio_variations.sh
```

Output:
- `songs/gsm/samples/` — GSM-encoded (8 kHz, 13 kbps) clean clips
- `songs/gsm/samples_noisy/` — GSM-encoded noisy clips
- `songs/low_mp3/samples/` — 64 kbps MP3 clean clips
- `songs/low_mp3/samples_noisy/` — 64 kbps MP3 noisy clips

---

### `scripts/match_song.py`
Match a single audio file against the fingerprint database.

```bash
python scripts/match_song.py <path_to_audio> [--phone-mode]

# Examples
python scripts/match_song.py songs/samples/Neelothi_sample_2.wav
python scripts/match_song.py songs/samples_noisy/Neelothi_sample_3_heavy.wav --phone-mode
python scripts/match_song.py songs/gsm/samples/Neelothi_sample_2.gsm --phone-mode
```

`--phone-mode` applies the 300–3400 Hz bandpass filter before fingerprinting, matching against the phone-mode hashes stored in the DB.

Output:
```
Query      : songs/samples/Neelothi_sample_2.wav
Phone mode : OFF
DB         : database/fingerprints.db

Hashes generated : 2847
Matches found    : 3

  Rank   Song                                Confidence
  -------------------------------------------------------
  1      Neelothi                                0.1823  <-- BEST MATCH
  2      Cheenikkallu                            0.0031
  3      Mannichiru                              0.0021
```

---

### `scripts/evaluate.py`
Batch evaluation across all clean and noisy samples. Reports Top-1 accuracy, Top-3 accuracy, and no-match rate per noise level with a final comparison table.

```bash
python scripts/evaluate.py
```

Output sections:
- **CLEAN SAMPLES** — baseline accuracy
- **NOISY SAMPLES — LIGHT** — Gaussian noise SNR +10 dB
- **NOISY SAMPLES — MEDIUM** — Gaussian noise SNR 0 dB
- **NOISY SAMPLES — HEAVY** — Gaussian noise SNR −5 dB
- **OVERALL COMPARISON** — side-by-side accuracy table

---

### `scripts/tune_params.py`
Grid-search over fingerprinter parameters to find the best combination for a given probe set.

```bash
# Full grid on all splits
python scripts/tune_params.py

# Clean split only, show top-3 combos
python scripts/tune_params.py --split clean --top 3

# Noisy splits, limit probes for a quick run
python scripts/tune_params.py --split light medium --max-probes 20

# Custom grid
python scripts/tune_params.py --fan-values 10 15 20 --delta-t-max 150 200 --min-conf 0.01 0.02
```

Default grid:

| Parameter | Values searched |
|---|---|
| `fan_value` | 5, 10, 15, 20 |
| `delta_t_max` | 100, 200, 300 |
| `freq_bin_size` | 5, 10 |
| `min_confidence` | 0.01, 0.02, 0.05 |

---

### `scripts/stream_audio.py`
Continuously captures audio from the microphone in 4-second chunks and matches each chunk against the DB in a background thread.

```bash
python scripts/stream_audio.py
```

- Prints `...` for each chunk with no match
- Prints a result card when a song is identified and keeps listening
- Press `s` + Enter to stop

Output on match:
```
==================================================
  🎵  SONG IDENTIFIED
==================================================
  Name       : Neelothi
  Confidence : 0.087
  Chunks     : processed until match
==================================================
```

---

### `scripts/drop_tables.py`
Drops all tables in the database (requires manual confirmation). Use to reset and re-run the pipeline from scratch.

```bash
python scripts/drop_tables.py
# → Type 'yes' to confirm
```

---

## 🔧 Configuration

### `AudioFingerprinter` defaults (`fingerprint.py`)

| Parameter | Default | Description |
|---|---|---|
| `sample_rate` | 8000 Hz | Target sample rate |
| `n_fft` | 2048 | STFT window size |
| `hop_length` | 512 | STFT hop (frame step) |
| `fan_value` | 10 | Target peaks per anchor |
| `delta_t_min` | 1 | Min frame gap anchor→target |
| `delta_t_max` | 200 | Max frame gap |
| `freq_bin_size` | 10 | Frequency quantisation |

### Matching defaults (`evaluate.py`, `match_song.py`)

| Parameter | Default | Description |
|---|---|---|
| `MIN_CONFIDENCE` | 0.02 | Reject matches below this normalised score |
| `TOP_N` | 5 | Number of candidates returned |

### Bandpass filter (`matcher.py`)

| Parameter | Value |
|---|---|
| Filter type | 4th-order Butterworth |
| Low cutoff | 300 Hz |
| High cutoff | 3400 Hz |
| Applied when | `is_phone_mode=True` |

---

## 📊 Evaluation

### Matching Algorithm

1. **Batch DB query** — all query hashes fetched in a single `WHERE hash_value IN (...)` call
2. **Offset-alignment voting** — `votes[song_id][db_offset − query_offset] += 1`; robust to where in the song the clip was taken
3. **Score normalisation** — `confidence = peak_votes / total_query_hashes` → comparable 0–1 scale
4. **Thresholding** — results below `MIN_CONFIDENCE` are discarded as no-match

### Expected Performance

| Condition | Top-1 Accuracy |
|---|---|
| Clean samples | ~95–100% |
| Light noise (SNR +10 dB) | ~90–98% |
| Medium noise (SNR 0 dB) | ~75–90% |
| Heavy noise (SNR −5 dB) | ~50–75% |

---

## 🔐 Robustness Features

- ✅ Volume changes — amplitude normalisation before hashing
- ✅ Recording offset — offset-alignment voting is position-independent
- ✅ Compression artifacts — high fan-out creates overlapping, redundant hashes
- ✅ Background noise — percentile thresholding retains only dominant spectral peaks
- ✅ Device/sample-rate variation — all audio resampled to 8000 Hz
- ✅ Telephone/GSM channels — dual hash indexing (broadband + bandpass) + `--phone-mode` query path

---

## 🐳 Docker Setup

```bash
# Build
docker build -t music-detection .

# Run Jupyter Lab
docker run -p 8888:8888 -v $(pwd):/workspace music-detection
```

Access at: `http://localhost:8888`

---

## 📚 References

- Wang, A. (2003). *An Industrial-Strength Audio Search Algorithm.* Shazam Entertainment.
- Librosa documentation: https://librosa.org/
- STFT: https://en.wikipedia.org/wiki/Short-time_Fourier_transform

---

## 📄 License

Educational project — S4 MFC Course.

---

**Last Updated**: March 2026 · **Python**: 3.10+ · **Sample Rate**: 8000 Hz · **FFT Size**: 2048

