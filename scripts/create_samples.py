"""
Create short audio samples (clean + noisy) from songs for query testing.

Sampling strategy:
  The song is divided into N_SAMPLES equal-width segments. A random start
  point is chosen within each segment, so clips are both randomized and
  equally distributed across the full duration.

Output structure:
    songs/
    ├── samples/           <- clean clips
    │   ├── Neelothi_sample_1.wav
    │   └── ...
    └── samples_noisy/     <- same clips with added noise/distortion
        ├── Neelothi_sample_1_noisy.wav
        └── ...
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import librosa
import soundfile as sf
import numpy as np

SONGS_DIR    = os.path.join(os.path.dirname(__file__), "..", "songs")
SAMPLES_DIR  = os.path.join(SONGS_DIR, "samples")
NOISY_DIR    = os.path.join(SONGS_DIR, "samples_noisy")

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}

# Clip settings
CLIP_DURATION  = 4       # seconds per sample
TARGET_SR      = 8000   # output sample rate

# Number of clips to extract per song.
# The song is split into this many equal segments and one clip is picked
# at a random position within each segment.
N_SAMPLES = 10

# Noise profiles applied to each clip (name, SNR dB)
# Lower SNR = more noise = harder match
NOISE_PROFILES = [
    ("light",  10),   # barely audible noise
    ("medium", 0),   # clearly noisy, like a phone recording
    ("heavy",   -5),   # heavy background noise
]


def add_noise(y, snr_db):
    """Add Gaussian noise at the given SNR (dB)."""
    signal_power = np.mean(y ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.normal(0, np.sqrt(noise_power), size=y.shape)
    return np.clip(y + noise, -1.0, 1.0)


def load_clip(audio_path, start, duration):
    y, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True,
        offset=start,
        duration=duration
    )
    return librosa.util.normalize(y)


def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(NOISY_DIR,   exist_ok=True)

    song_files = [
        f for f in os.listdir(SONGS_DIR)
        if os.path.isfile(os.path.join(SONGS_DIR, f))
        and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not song_files:
        print("No audio files found in the songs folder.")
        return

    total_clean = 0
    total_noisy = 0

    print(f"Found {len(song_files)} song(s).")
    print(f"Samples : {N_SAMPLES} per song (randomised + equally distributed)")
    print(f"Noise   : {[p[0] for p in NOISE_PROFILES]}\n")

    for filename in sorted(song_files):
        song_name  = os.path.splitext(filename)[0]
        audio_path = os.path.join(SONGS_DIR, filename)

        print(f"  {filename}")

        try:
            duration = librosa.get_duration(path=audio_path)

            usable = duration - CLIP_DURATION
            if usable <= 0:
                print(f"    Song too short ({duration:.1f}s < {CLIP_DURATION}s clip), skipping.")
                continue

            # Divide usable range into equal segments, pick a random point in each
            segment_size = usable / N_SAMPLES
            offsets = [
                float(np.clip(
                    np.random.uniform(i * segment_size, (i + 1) * segment_size),
                    0, usable
                ))
                for i in range(N_SAMPLES)
            ]

            print(f"    Duration: {duration:.1f}s | Offsets: {[f'{o:.1f}s' for o in offsets]}")

            for i, start in enumerate(offsets, start=1):
                actual_dur = CLIP_DURATION  # always full clip length

                # --- Clean clip ---
                clean_path = os.path.join(SAMPLES_DIR, f"{song_name}_sample_{i}.wav")
                if os.path.isfile(clean_path):
                    print(f"    [{i}] clean already exists, skipping.")
                    y_clean = load_clip(audio_path, start, actual_dur)
                else:
                    y_clean = load_clip(audio_path, start, actual_dur)
                    sf.write(clean_path, y_clean, TARGET_SR)
                    total_clean += 1
                    print(f"    [{i}] clean saved ({actual_dur:.1f}s from {start:.1f}s).")

                # --- Noisy clips ---
                for profile_name, snr_db in NOISE_PROFILES:
                    noisy_path = os.path.join(
                        NOISY_DIR,
                        f"{song_name}_sample_{i}_{profile_name}.wav"
                    )
                    if os.path.isfile(noisy_path):
                        print(f"    [{i}] {profile_name} noise already exists, skipping.")
                        continue

                    y_noisy = add_noise(y_clean, snr_db)
                    sf.write(noisy_path, y_noisy, TARGET_SR)
                    total_noisy += 1
                    print(f"    [{i}] {profile_name} noise saved (SNR {snr_db} dB).")

        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\nDone.")
    print(f"  Clean  samples : {total_clean}  →  {SAMPLES_DIR}")
    print(f"  Noisy  samples : {total_noisy}  →  {NOISY_DIR}")


if __name__ == "__main__":
    main()
