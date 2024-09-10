import numpy as np
import pandas as pd
import yasa
from scipy.signal import welch
from scipy.integrate import simpson
from .Base import Base


class SpectralDensity(Base):
    def __init__(self, epoch, window_sec:int, bands:list[tuple[float, float, str]]) -> None:
        self.from_epoch = epoch
        self.window_sec = window_sec
        self.bands = bands
        self.welches, self.freqs, self.power_spectral_density = self.build_welch()
        self.absolute_power = self.get_absolute_power()

    def build_welch(self) -> tuple[np.array, np.array,]:
        window_length = self.from_epoch.from_channel.freq*self.window_sec
        noverlap = window_length//2
        nperseg = window_length

        freqs, power_spectral_density = welch(
            x = self.from_epoch.epochs,
            fs = self.from_epoch.from_channel.freq,
            window ='hann',
            scaling='density',
            average='median',
            nperseg=nperseg,
            noverlap=noverlap,
        )
        bandpower = yasa.bandpower_from_psd_ndarray(power_spectral_density, freqs, bands=self.bands)
        welches = {}
        for i, (_, _, band_name) in enumerate(self.bands):
            welches[band_name] = bandpower[i]
        return (welches, freqs, power_spectral_density)
    
    def make_dataframe(self, feature: dict) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {
                'time': self.from_epoch.times,
                **feature
            }
        )

    def get_power_ratios(self) -> dict:
        power_ratios = {}
        if 'sdelta' in self.welches and 'fdelta' in self.welches:
            delta = self.welches["sdelta"] + self.welches["fdelta"]
            for wave in [w for w in self.welches.keys() if w not in ('sdelta', 'fdelta')]:
                power_ratios[f"delta/{wave}"] = delta / self.welches[wave]
        if 'alpha' in self.welches and 'theta' in self.welches:
            power_ratios["alpha/theta"] = self.welches["alpha"] / self.welches["theta"]
        return power_ratios
    
    def get_absolute_power(self) -> dict:
        idx_broad = np.logical_and(self.freqs >= self.from_epoch.freq_broad[0],
                                   self.freqs <= self.from_epoch.freq_broad[1])
        dx = self.freqs[1] - self.freqs[0]
        abs_power = simpson(self.power_spectral_density[:, idx_broad], dx=dx)
        return {"absolute_power": abs_power}

    def get_power_std(self) -> dict:
        welch_stds = {f"{k}_std": [np.std(array)] * len(array) for k, array in self.welches.items()}
        return welch_stds
    
    def get_relative_powers(self) -> dict:
        relative_powers = {f"{k}_relative": array/self.absolute_power['absolute_power'] for k, array in self.welches.items()}
        return relative_powers
    
    def get_total_power(self) -> dict:
        powers = list(self.welches.values())
        total = np.zeros(len(powers[0]))
        for power in powers:
            total += power
        return {'total': total}