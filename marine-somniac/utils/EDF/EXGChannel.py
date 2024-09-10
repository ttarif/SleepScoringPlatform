from .Channel import Channel
from .Epoch import Epoch
import pandas as pd
import numpy as np
import mne
import yasa




class EXGChannel(Channel):
    def get_epoch(self, freq_broad:tuple[float,float]=(0.4, 30), window_sec:int=32, step_size:int=4) -> tuple:
        """
        freq_broad: broad range frequency of EEG (this is used for "absolute power" calculations, and as a divisor for calculating overall relative power)
        window_sec: size of the epoch rolling window to use in seconds
        step_size: how big of a step size to use, in seconds
        """
        return Epoch(self, freq_broad, window_sec, step_size)

    def get_epochs(self, freq_broad:tuple[float,float]=(0.4, 30), window_sec:int=32, step_size:int=4) -> tuple:
        """
        freq_broad: broad range frequency of EEG (this is used for "absolute power" calculations, and as a divisor for calculating overall relative power)
        window_sec: size of the epoch rolling window to use in seconds
        step_size: how big of a step size to use, in seconds
        """
        dt_filt = mne.filter.filter_data(
            self.signal, self.freq, 
            l_freq=freq_broad[0], h_freq=freq_broad[1], verbose=False
        )
        times, epochs = yasa.sliding_window(
            dt_filt, sf=self.freq, 
            window=window_sec, step=step_size
        )
        # add window/2 to the times to make the epochs "centered" around the times
        times = times + window_sec // 2 
        return (times, epochs)
    
    def get_antropy_features(self, epochs: np.array) -> dict:
        features = {
            'permutation_entropy': self.get_permutation_entropy(epochs),
            'higuchi_fractal_dimension': self.get_higuchi(epochs),
            'petrosian_fractal_dimension': self.get_petrosian(epochs),
            "number_zero_crossings": self.get_zero_crossings(epochs)
        }
        return features
    
    def get_epoch_stats(self, epochs: np.array) -> dict:
        stats = {
            "std": self.get_std(epochs),
            "iqr": self.get_interquartile_range(epochs),
            "skew": self.get_skew(epochs),
            "kurtosis": self.get_kurtosis(epochs)
        }
        return stats

    def get_epoch_derived_features(self, freq_broad=(0.4, 30), epoch_window_sec=32, step_size=4, welch_window_sec:int=4) -> pd.DataFrame:
        """
        Gets ECG features using similar code & syntax as YASA's feature generation, calculates deterministic features as well as spectral features
        freq_broad: broad range frequency of EEG (this is used for "absolute power" calculations, and as a divisor for calculating overall relative power)
        sfreq: sampling frequeuency, by default 500 Hz
        epoch_window_sec: size of the epoch rolling window to use
        welch_window_sec: size of the welch window for power spectral density calculations (this affects the low frequeuncy power and very low frequency power calculations, etc.)
        step_size: how big of a step size to use, in seconds
        bands: optional parameter to override the default bands used, for exmaple if you'd like more specific bands than just sdelta, fdelta, theta, alpha, etc
        """
        times, epochs = self.get_epochs(freq_broad, epoch_window_sec, step_size)

        hjorth_parameters = self.get_hjorth_params(epochs)
        antropy_features = self.get_antropy_features(epochs)
        stats = self.get_epoch_stats(epochs)
        welches, freqs, power_spectral_density = self.get_welch_of_bands(epochs, welch_window_sec)
        power_ratios = self.get_power_ratios(welches)
        absolute_power = self.get_absolute_power(freqs, freq_broad, power_spectral_density)

        welch_stds = {f"{k}_std": [np.std(array)] * len(array) for k, array in welches}
        relative_powers = {f"{k}_relative": array/absolute_power['absolute_power'] for k, array in welches}

        features = {
            'epoch': times
            **stats,
            **hjorth_parameters, 
            **antropy_features,
            **welches,
            **welch_stds,
            **relative_powers,
            **power_ratios,
            **absolute_power,
        }
        return pd.DataFrame(features)
    