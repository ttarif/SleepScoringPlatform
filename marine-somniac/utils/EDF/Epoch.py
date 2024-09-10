import numpy as np
import pandas as pd
import antropy
import mne
import yasa
from scipy.stats import iqr, skew, kurtosis
from utils.EDF.SpectralDensity import SpectralDensity
from .Base import Base
from .constants import EEG_BANDS, HR_BANDS


class Epoch(Base):
    def __init__(self, channel, freq_broad:tuple[float,float], window_sec:int, step_size:int) -> None:
        self.from_channel = channel
        self.freq_broad = freq_broad
        self.window_sec = window_sec
        self.step_size = step_size
        self.times, self.epochs = self.build_epoch()

    def build_epoch(self) -> tuple[np.array, np.array]:
        dt_filt = mne.filter.filter_data(
            self.from_channel.signal,
            self.from_channel.freq, 
            l_freq=self.freq_broad[0],
            h_freq=self.freq_broad[1],
            verbose=False
        )
        times, epochs = yasa.sliding_window(
            dt_filt, sf=self.from_channel.freq, 
            window=self.window_sec, step=self.step_size
        )
        # add window/2 to the times to make the epochs "centered" around the times
        times = times + self.window_sec // 2 
        return (times, epochs)
    
    def make_dataframe(self, feature: dict) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {
                'time': self.times,
                **feature
            }
        )

    def get_hjorth_params(self) -> dict:
        mobility, complexity = antropy.hjorth_params(self.epochs, axis=1)
        return {
            'hjorth_mobility': mobility,
            'hjorth_complexity': complexity
        }
    
    def get_permutation_entropy(self) -> dict:
        perm_ent = np.apply_along_axis(antropy.perm_entropy, axis=1, arr=self.epochs, normalize=True)
        return {'permutation_entropy': perm_ent}
    
    def get_higuchi(self) -> dict:
        higuchi = np.apply_along_axis(antropy.higuchi_fd, axis=1, arr=self.epochs)
        return {'higuchi_fractal_dimension': higuchi}
    
    def get_petrosian(self) -> dict:
        petrosian = antropy.petrosian_fd(self.epochs, axis=1)
        return {'petrosian_fractal_dimension': petrosian}

    def get_std(self) -> dict: 
        std = np.std(self.epochs, ddof=1, axis=1)
        return {'epoch_std': std}
    
    def get_interquartile_range(self) -> dict: 
        iqr_ = iqr(self.epochs, rng=(25, 75), axis=1)
        return {'epoch_iqr': iqr_}
    
    def get_skew(self) -> dict: 
        skew_ = skew(self.epochs, axis=1)
        return {'epoch_skew': skew_}
    
    def get_kurtosis(self) -> dict: 
        kurt = kurtosis(self.epochs, axis=1)
        return {'epoch_kurtosis': kurt}
    
    def get_zero_crossings(self) -> np.array: 
        nzc = antropy.num_zerocross(self.epochs, axis=1)
        return {'epoch_n_zero_crossings': nzc}
    
    def get_welch(self, window_sec:int=4, bands:list[tuple[float, float, str]]=EEG_BANDS) -> SpectralDensity:
        """
        window_sec: size of the rolling window to use in seconds
        bands: band ranges and their name from which to calculate the spectral density
        """
        return SpectralDensity(self, window_sec, bands)
    
    def get_hr_welch(self, window_sec:int=512, bands:list[tuple[float, float, str]]=HR_BANDS) -> SpectralDensity:
        """
        window_sec: size of the rolling window to use in seconds
        bands: band ranges and their name from which to calculate the spectral density
        """
        return SpectralDensity(self, window_sec, bands)

        