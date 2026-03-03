import librosa
import numpy as np
from scipy.ndimage import maximum_filter
from matcher import bandpass


class AudioFingerprinter:

    def __init__(self, sample_rate=8000):
        self.sample_rate = sample_rate
        self.n_fft = 2048
        self.hop_length = 512
        self.fan_value = 10
        self.target_zone_time = 50

    def preprocess(self, file_path, is_phone_mode=False):
        """
        Load audio and standardize it.

        is_phone_mode : bool
            When True, apply a 300–3400 Hz bandpass filter after standard
            preprocessing to simulate telephone / GSM channel conditions.

        Returns:
            y  -> normalized mono waveform
            sr -> fixed sample rate
        """

        y, sr = librosa.load(
            file_path,
            sr=self.sample_rate,
            mono=True
        )

        y = librosa.util.normalize(y)

        y = y - np.mean(y)

        y = librosa.effects.preemphasis(y)

        if is_phone_mode:
            y = bandpass(y, sr)

        return y, sr

    def generate_spectrogram(self, y):
        """
        Generate log-magnitude spectrogram.
        Returns:
            S_db -> log-scaled magnitude spectrogram
        """

        S = librosa.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        S_mag = np.abs(S)

        S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

        return S_db
    
    def find_peaks(self, S_db, amp_min=-50):
        """
        Detect spectral peaks from log spectrogram.

        amp_min: minimum amplitude (dB) threshold
        """

        neighborhood_size = (20, 20)
        local_max = maximum_filter(S_db, size=neighborhood_size)

        # 2️ Keep points that equal local max
        detected_peaks = (S_db == local_max)

        # 3️ Apply amplitude threshold
        threshold = np.percentile(S_db, 85, axis=0)
        threshold_mask = S_db > threshold[np.newaxis, :]
        peaks_mask = detected_peaks & (S_db > amp_min)

        # 4️ Combine masks
        # peaks_mask = detected_peaks & threshold_mask
        peaks_mask = detected_peaks & (S_db > amp_min)

        # 5️ Extract peak indices
        freq_indices, time_indices = np.where(peaks_mask)

        peaks = list(zip(freq_indices, time_indices))

        return peaks
    
    @staticmethod
    def _make_hash(f1_bin, f2_bin, delta_t, freq_bin_size=5):
        """
        Pack two frequency bins and a time delta into a single 64-bit hash.

        Layout: [f1 16-bit][f2 16-bit][dt 16-bit][reserved 16-bit]

        Frequency bins are coarsened by 2x before packing so the hash is
        tolerant to slight pitch shifts and compression artefacts.

        Returns the hash integer, or None if any field overflows 16 bits.
        """
        f1_coarse = (f1_bin // freq_bin_size)
        f2_coarse = (f2_bin // freq_bin_size)

        delta_t_bin = delta_t//2

        if f1_coarse >= 65536 or f2_coarse >= 65536 or delta_t_bin >= 65536:
            return None

        FREQ_BITS = 9
        DELTA_BITS = 8

        return (f1_coarse << (FREQ_BITS + DELTA_BITS)) | (f2_coarse << DELTA_BITS ) | delta_t_bin

    def generate_hashes(
        self,
        peaks,
        fan_value=10,
        delta_t_min=1,
        delta_t_max=200,
        freq_bin_size=10
    ):
        """
        Industry-style 64-bit landmark hashing with Target Zones.
        """
        if not peaks:
            return []

        # Ensure peaks are sorted by time
        peaks_sorted = sorted(peaks, key=lambda x: x[1])
        hashes = []
        total_peaks = len(peaks_sorted)

        for i in range(total_peaks):
            f1, t1 = peaks_sorted[i]
            valid_pairs = 0  # Track actual pairs made, not just array indices

            # Scan forward through the remaining array
            for j in range(1, total_peaks - i):
                f2, t2 = peaks_sorted[i + j]
                delta_t = t2 - t1

                # 1. Skip over peaks that are too close in time
                if delta_t < delta_t_min:
                    continue
                
                # 2. TARGET ZONE LIMIT: If we scan past the max time window, stop searching for this anchor
                if delta_t > delta_t_max:
                    break

                # 3. Create the hash
                hash_value = self._make_hash(f1, f2, delta_t, freq_bin_size)
                if hash_value is not None:
                    hashes.append((hash_value, t1))
                    valid_pairs += 1  # Successfully paired with a peak in the target zone

                # 4. FAN VALUE LIMIT: Stop once we hit the limit OF VALID PAIRS
                if valid_pairs >= fan_value:
                    break

        return hashes

    @staticmethod
    def tune_parameters(
        audio_paths_and_labels,
        match_fn,
        fan_values=(5, 10, 15, 20),
        delta_t_max_values=(100, 200, 300),
        freq_bin_size_values=(5, 10),
        min_confidence_values=(0.01, 0.02, 0.05),
        sample_rate=8000,
        verbose=True,
    ):
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
            'hashes' is the list[(hash_value, offset_t)] from generate_hashes.
        fan_values, delta_t_max_values, freq_bin_size_values, min_confidence_values
            Iterables defining the grid to search.
        verbose : bool
            Print a result table while running.

        Returns
        -------
        best_params : dict
            Keys: 'fan_value', 'delta_t_max', 'freq_bin_size', 'min_confidence'
        all_results : list of dict
            Every combination tried, sorted by top1_acc descending.
        """
        import itertools

        grid = list(itertools.product(
            fan_values,
            delta_t_max_values,
            freq_bin_size_values,
            min_confidence_values,
        ))

        if verbose:
            print(f"\nParameter tuning: {len(grid)} combinations × {len(audio_paths_and_labels)} probes")
            print(f"{'fan':>5} {'dt_max':>7} {'fbsz':>6} {'min_conf':>9}  {'Top-1':>7}  {'Top-3':>7}")
            print("-" * 60)

        all_results = []
        fp = AudioFingerprinter(sample_rate=sample_rate)

        for fan_value, delta_t_max, freq_bin_size, min_conf in grid:
            top1 = top3 = total = 0

            for audio_path, expected in audio_paths_and_labels:
                try:
                    y, _   = fp.preprocess(audio_path)
                    S_db   = fp.generate_spectrogram(y)
                    peaks  = fp.find_peaks(S_db)
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
                "fan_value":      fan_value,
                "delta_t_max":    delta_t_max,
                "freq_bin_size":  freq_bin_size,
                "min_confidence": min_conf,
                "top1_acc":       top1_acc,
                "top3_acc":       top3_acc,
                "total":          total,
            }
            all_results.append(result)

            if verbose:
                print(
                    f"{fan_value:>5} {delta_t_max:>7} {freq_bin_size:>6}"
                    f" {min_conf:>9.3f}  {top1_acc:>6.1%}  {top3_acc:>6.1%}"
                )

        # Sort best first: primary = top1_acc, tiebreak = top3_acc
        all_results.sort(key=lambda r: (r["top1_acc"], r["top3_acc"]), reverse=True)
        best = all_results[0] if all_results else {}

        if verbose and best:
            print("\nBest parameters found:")
            for k in ("fan_value", "delta_t_max", "freq_bin_size", "min_confidence"):
                print(f"  {k}: {best[k]}")
            print(f"  top1_acc: {best['top1_acc']:.1%}  top3_acc: {best['top3_acc']:.1%}")

        best_params = (
            {k: best[k] for k in ("fan_value", "delta_t_max", "freq_bin_size", "min_confidence")}
            if best else {}
        )
        return best_params, all_results