"""
app/core/fingerprint.py — Audio Fingerprinting Engine
======================================================

Provides the ``AudioFingerprinter`` class, which converts a raw audio file
into a compact set of 64-bit landmark hashes that can be stored in or
queried against the Redis fingerprint store.

Pipeline (per audio file)
--------------------------
1. ``preprocess(file_path, is_phone_mode)``
       Load audio, convert to mono, resample to 8 kHz, normalise amplitude,
       remove DC offset, apply pre-emphasis.  When ``is_phone_mode=True``,
       also applies a 300–3400 Hz Butterworth bandpass via ``app.core.matcher.bandpass``
       to simulate telephone / GSM channel conditions.

2. ``generate_spectrogram(y)``
       Short-Time Fourier Transform (STFT) → magnitude → log dB scaling.

3. ``find_peaks(S_db)``
       Detect dominant spectral landmarks with a 20×20 maximum filter,
       keeping only peaks above the per-frame 85th-percentile amplitude threshold.

4. ``generate_hashes(peaks)``
       Pair each anchor peak with up to ``fan_value`` target peaks in a
       forward time window (the "constellation map") and pack
       ``(f1, f2, Δt)`` into a single 64-bit integer.
       Returns ``[(hash_value, time_offset), ...]``.

5. ``tune_parameters(...)``  (static)
       Grid-search over fan_value, delta_t_max, freq_bin_size, and
       min_confidence to maximise match accuracy on a labelled probe set.
"""

import librosa
import numpy as np
from scipy.ndimage import maximum_filter
from typing import Any, Callable

from app.core.matcher import bandpass


class AudioFingerprinter:
    def __init__(self, sample_rate: int = 8000):
        self.sample_rate = sample_rate
        self.n_fft = 1024
        self.hop_length = 256
        self.fan_value = 10
        self.target_zone_time = 50

    def preprocess(
        self, file_path: str, is_phone_mode: bool = False
    ) -> tuple[np.ndarray, int]:
        """
        Load audio and standardise it.

        is_phone_mode : bool
            When True, apply a 300–3400 Hz bandpass filter after standard
            preprocessing to simulate telephone / GSM channel conditions.

        Returns:
            y  -> normalised mono waveform (np.ndarray)
            sr -> fixed sample rate (int)
        """
        y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        y = librosa.util.normalize(y)
        y = y - np.mean(y)
        y = librosa.effects.preemphasis(y)

        if is_phone_mode:
            y = bandpass(y, int(sr))

        return y, int(sr)

    def generate_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        Generate log-magnitude spectrogram via STFT.

        Returns:
            S_db -> log-scaled magnitude spectrogram (np.ndarray)
        """
        S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        S_db = np.asarray(librosa.amplitude_to_db(np.abs(S), ref=np.max))
        return S_db

    def find_peaks(
        self, S_db: np.ndarray, amp_min: float = -50
    ) -> list[tuple[int, int]]:
        """
        Detect spectral peaks from a log-magnitude spectrogram.

        Uses a 20×20 maximum filter to find local maxima, then keeps only
        peaks above ``amp_min`` dB.

        Parameters:
            S_db    : log-magnitude spectrogram
            amp_min : minimum amplitude (dB) threshold
        """
        local_max = maximum_filter(S_db, size=(20, 20))
        detected = S_db == local_max
        peaks_mask = detected & (S_db > amp_min)
        freq_idx, time_idx = np.where(peaks_mask)
        return [(int(f), int(t)) for f, t in zip(freq_idx, time_idx)]

    @staticmethod
    def _make_hash(
        f1_bin: int, f2_bin: int, delta_t: int, freq_bin_size: int = 5
    ) -> int | None:
        """
        Pack two frequency bins and a time delta into a single 64-bit hash.

        Layout: [f1 coarsened][f2 coarsened][Δt/2] using fixed bit widths.
        Returns the hash integer, or None if any field overflows.
        """
        f1_coarse = f1_bin // freq_bin_size
        f2_coarse = f2_bin // freq_bin_size
        delta_t_bin = delta_t // 2

        if f1_coarse >= 65536 or f2_coarse >= 65536 or delta_t_bin >= 65536:
            return None

        FREQ_BITS = 9
        DELTA_BITS = 8
        return (
            (f1_coarse << (FREQ_BITS + DELTA_BITS))
            | (f2_coarse << DELTA_BITS)
            | delta_t_bin
        )

    def generate_hashes(
        self,
        peaks: list[tuple[int, int]],
        fan_value: int = 5,
        delta_t_min: int = 1,
        delta_t_max: int = 200,
        freq_bin_size: int = 10,
    ) -> list[tuple[int, int]]:
        """
        Industry-style 64-bit landmark hashing with Target Zones (Shazam-style).

        Returns:
            list of (hash_value: int, time_offset: int)
        """
        if not peaks:
            return []

        peaks_sorted = sorted(peaks, key=lambda x: x[1])
        hashes = []
        total_peaks = len(peaks_sorted)

        for i in range(total_peaks):
            f1, t1 = peaks_sorted[i]
            valid_pairs = 0

            for j in range(1, total_peaks - i):
                f2, t2 = peaks_sorted[i + j]
                delta_t = t2 - t1

                if delta_t < delta_t_min:
                    continue
                if delta_t > delta_t_max:
                    break

                hash_value = self._make_hash(f1, f2, delta_t, freq_bin_size)
                if hash_value is not None:
                    hashes.append((hash_value, t1))
                    valid_pairs += 1

                if valid_pairs >= fan_value:
                    break

        return hashes

    @staticmethod
    def tune_parameters(
        audio_paths_and_labels: list[tuple[str, str]],
        match_fn: Callable[..., Any],
        fan_values: tuple[int, ...] = (5, 10, 15, 20),
        delta_t_max_values: tuple[int, ...] = (100, 200, 300),
        freq_bin_size_values: tuple[int, ...] = (5, 10),
        min_confidence_values: tuple[float, ...] = (0.01, 0.02, 0.05),
        sample_rate: int = 8000,
        verbose: bool = True,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Grid-search over fan_value, delta_t_max, freq_bin_size, and
        min_confidence to maximise Top-1 accuracy on a labelled probe set.

        Parameters
        ----------
        audio_paths_and_labels : list of (str, str)
            Pairs of (audio_file_path, expected_song_name).
        match_fn : callable
            Signature: match_fn(hashes, min_confidence=float)
              -> list of (song_name, confidence) ranked best-first.

        Returns
        -------
        best_params : dict
        all_results : list of dict  (every combination, sorted by top1_acc desc)
        """
        import itertools

        grid = list(
            itertools.product(
                fan_values,
                delta_t_max_values,
                freq_bin_size_values,
                min_confidence_values,
            )
        )

        if verbose:
            print(
                f"\nParameter tuning: {len(grid)} combinations × {len(audio_paths_and_labels)} probes"
            )
            print(
                f"{'fan':>5} {'dt_max':>7} {'fbsz':>6} {'min_conf':>9}  {'Top-1':>7}  {'Top-3':>7}"
            )
            print("-" * 60)

        all_results = []
        fp = AudioFingerprinter(sample_rate=sample_rate)

        for fan_value, delta_t_max, freq_bin_size, min_conf in grid:
            top1 = top3 = total = 0

            for audio_path, expected in audio_paths_and_labels:
                try:
                    y, _ = fp.preprocess(audio_path)
                    S_db = fp.generate_spectrogram(y)
                    peaks = fp.find_peaks(S_db)
                    hashes = fp.generate_hashes(
                        peaks,
                        fan_value=fan_value,
                        delta_t_max=delta_t_max,
                        freq_bin_size=freq_bin_size,
                    )
                    ranked = match_fn(hashes, min_confidence=min_conf)
                except Exception:
                    continue

                total += 1
                if not ranked:
                    continue

                top_names = [name for name, _ in ranked]
                if top_names[0] == expected:
                    top1 += 1
                if expected in top_names:
                    top3 += 1

            top1_acc = top1 / total if total else 0.0
            top3_acc = top3 / total if total else 0.0

            result = {
                "fan_value": fan_value,
                "delta_t_max": delta_t_max,
                "freq_bin_size": freq_bin_size,
                "min_confidence": min_conf,
                "top1_acc": top1_acc,
                "top3_acc": top3_acc,
                "total": total,
            }
            all_results.append(result)

            if verbose:
                print(
                    f"{fan_value:>5} {delta_t_max:>7} {freq_bin_size:>6} {min_conf:>9.3f}"
                    f"  {top1_acc:>7.3f}  {top3_acc:>7.3f}"
                )

        all_results.sort(key=lambda r: r["top1_acc"], reverse=True)
        best = all_results[0] if all_results else {}

        if verbose and best:
            print("\nBest combination:")
            for k, v in best.items():
                print(f"  {k}: {v}")

        return best, all_results
