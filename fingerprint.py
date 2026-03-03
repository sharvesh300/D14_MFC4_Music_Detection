import librosa
import numpy as np
from scipy.ndimage import maximum_filter


class AudioFingerprinter:

    def __init__(self, sample_rate=22050):
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
    
    def find_peaks(self, S_db, amp_min=-40):
        """
        Detect spectral peaks from log spectrogram.

        amp_min: minimum amplitude (dB) threshold
        """

        neighborhood_size = (20, 20)
        local_max = maximum_filter(S_db, size=neighborhood_size)

        # 2️ Keep points that equal local max
        detected_peaks = (S_db == local_max)

        # 3️ Apply amplitude threshold
        threshold_mask = S_db > amp_min

        # 4️ Combine masks
        peaks_mask = detected_peaks & threshold_mask

        # 5️ Extract peak indices
        freq_indices, time_indices = np.where(peaks_mask)

        peaks = list(zip(freq_indices, time_indices))

        return peaks
    
    def generate_hashes(
        self,
        peaks,
        fan_value=7,
        delta_t_min=1,
        delta_t_max=200,
        freq_bin_size=5
    ):
        """
        Industry-style 64-bit landmark hashing.

        Returns:
            List of (hash_value, anchor_time)
        """

        if not peaks:
            return []

        peaks_sorted = sorted(peaks, key=lambda x: x[1])
        hashes = []
        total_peaks = len(peaks_sorted)

        for i in range(total_peaks):
            f1, t1 = peaks_sorted[i]

            # Quantize anchor frequency
            f1_bin = f1 // freq_bin_size

            for j in range(1, fan_value + 1):

                if i + j >= total_peaks:
                    break

                f2, t2 = peaks_sorted[i + j]
                delta_t = t2 - t1

                if delta_t < delta_t_min:
                    continue

                if delta_t > delta_t_max:
                    break

                # Quantize target frequency
                f2_bin = f2 // freq_bin_size

                # Ensure values fit 16-bit range
                if f1_bin >= 65536 or f2_bin >= 65536 or delta_t >= 65536:
                    continue

                # 64-bit packed hash:
                # [f1 16][f2 16][dt 16][reserved 16]
                hash_value = (
                    (f1_bin << 48) |
                    (f2_bin << 32) |
                    (delta_t << 16)
                )

                hashes.append((hash_value, t1))

        return hashes