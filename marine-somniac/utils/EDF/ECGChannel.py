from .EXGChannel import EXGChannel
from .Epoch import Epoch
import pandas as pd
import numpy as np
import wfdb.processing
from sleepecg import detect_heartbeats
try:
    from typing import Self
except:
    from typing_extensions import Self


class ECGChannel(EXGChannel):
    def get_heart_rate(self, search_radius:int=200, filter_threshold:int=200) -> Self:
        """
        search_radius: search radius to look for peaks (200 ~= 150 bpm upper bound)
        filter_threshold: threshold above which to throw out values (filter_threshold=200 would throw out any value above 200 bpm and impute it from its neighbors)
        step_size: adas
        """
        rpeaks = detect_heartbeats(self.signal, self.freq)  # using sleepecg
        rpeaks_corrected = wfdb.processing.correct_peaks(
            self.signal, rpeaks, search_radius=search_radius, smooth_window_size=50, peak_dir="up"
        )
        heart_rates = [60 / ((rpeaks_corrected[i+1] - rpeaks_corrected[i]) / self.freq) for i in range(len(rpeaks_corrected) - 1)]
        # Create a heart rate array matching the frequency of the ECG trace
        hr_data = np.zeros_like(self.signal)
        # Assign heart rate values to the intervals between R-peaks
        for i in range(len(rpeaks_corrected) - 1):
            start_idx = rpeaks_corrected[i]
            end_idx = rpeaks_corrected[i+1]
            hr_data[start_idx:end_idx] = heart_rates[i]

        hr_data = pd.Series(hr_data)
        hr_data[hr_data > filter_threshold] = np.nan
        hr_data = hr_data.interpolate(method='quadratic', order=5).fillna('ffill').fillna('bfill')
        hr_data = hr_data.to_numpy()
        return self._return(hr_data, step_size=self.freq)
    
    def get_hr_epoch(self, freq_broad:tuple[float,float]=(0, 1), window_sec:int=512, step_size:int=32) -> tuple:
        """
        freq_broad: broad range frequency of EEG (this is used for "absolute power" calculations, and as a divisor for calculating overall relative power)
        window_sec: size of the epoch rolling window to use in seconds
        step_size: how big of a step size to use, in seconds
        """
        return Epoch(self, freq_broad, window_sec, step_size)

    def get_hr_epoch_bundle(self, heart_rate_data, freq_broad=(0,1), sfreq=500, epoch_window_sec=512, welch_window_sec=512, step_size=32) -> pd.DataFrame:
        """
        Gets heartrate features using similar code & syntax as YASA's feature generation, calculates deterministic features as well as spectral features
        heart_rate_data: heart rate vector data (must already be processed from an ECG, this function does NOT take ECG data)
        sfreq: sampling frequeuency, by default 500 Hz
        epoch_window_sec: size of the epoch rolling window to use
        welch_window_sec: size of the welch window for power spectral density calculations (this affects the low frequeuncy power and very low frequency power calculations, etc.)
        step_size: how big of a step size to use, in seconds
        """

        times, epochs = self.get_epochs(freq_broad, epoch_window_sec, step_size)
        hjorth_parameters = self.get_hjorth_params(epochs)
        stats = self.get_epoch_stats(epochs)
        welches, freqs, power_spectral_density = self.get_welch_of_bands(epochs, welch_window_sec)

        low_high = {'lf/hf': welches['lf']/welches['hf']}
        total_power = self.get_total_power(welches)
        percent_powers = {f"{k}_percent": 100*array/total_power['total_power'] for k, array in welches}

        nu_factor = 100 / (feat['lf'] + feat['hf'])
        feat['lf_nu'] = feat['lf'] * nu_factor
        feat['hf_nu'] = feat['hf'] * nu_factor

        features = {
                'epoch': times
                **stats,
                **hjorth_parameters, 
                **welches,
                **percent_powers,
            }
        features = {f"hr_{k}":v for k,v in features.items()}
        return pd.DataFrame(features)
