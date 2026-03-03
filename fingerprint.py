import librosa
import numpy as np
from scipy.ndimage import maximum_filter


class AudioFingerprinter:

    def __init__(self, sample_rate=8000):
        self.sample_rate = sample_rate
        self.n_fft = 2048
        self.hop_length = 512
        self.fan_value = 10
        self.target_zone_time = 50

    def preprocess(self, file_path):
        """
        Load audio and standardize it.
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
    
    def find_peaks(self, S_db, amp_min=-60):
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

        # 4️ Combine masks
        peaks_mask = detected_peaks & threshold_mask

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

        return (f1_coarse << (FREQ_BITS + DELTA_BITS)) | (f2_coarse << DELTA_BITS   ) | delta_t_bin

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