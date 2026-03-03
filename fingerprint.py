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