# TODO 
# inspect frequency & step values - integer vs float consistency

import inspect; from typing import Self
from datetime import datetime, timedelta
import numpy as np; import pandas as pd
import mne; from sleepecg import detect_heartbeats; import wfdb.processing
from scipy.integrate import simpson; from scipy.signal import welch, hann

class Channel:
    def __init__(self, start_ts, name: str, signal: np.array, time:np.array=None, freq=None) -> None:
        self.name = name
        self.time = time
        self.signal = signal
        self.freq = freq

        # TODO end ts calculation
        self._start_ts = start_ts
        
        # TODO store reference to parent EDFutils obj and have each 
        # construction trigger storage in cache
        # self._parent = parent
        # self._parent.cache()
            
    def __getitem__(self, slice) -> Self:
        freq = self.freq
        if slice.step:
            freq = freq / slice.step
            if freq.is_integer():
                freq = int(freq)

        slice_time = self.time[slice]
        slice_signal = self.signal[slice]
        return Channel(
            name=self.name,
            signal=slice_signal,
            time=slice_time,
            freq=freq,
            start_ts=self._start_ts
        )
    
    def time_slice(self, start_time, end_time, unit='second') -> Self:
        mod = None
        match unit:
            case 'second':
                mod = 1
            case 'minute':
                mod = 60
            case 'hour':
                mod = 3600
            case _:
                raise ValueError(f'Only accepts second, minute, and hour, not {unit}')
        
        start = mod * self.freq * start_time
        end   = mod * self.freq * end_time
        return self[start:end]
    
    # TODO
    def date_slice(self, start_date, end_date) -> Self:
        self._start_ts
        pass
    
    def _return(self, new_signal, step_size) -> Self:
        # inspect.stack()[1][3] returns the name of the function
        # traced back before this function call
        new_name = f'{self.name}.{inspect.stack()[1][3]}'
        new_time = self.time[::self.freq//step_size]
        new_freq = 1/step_size
        return Channel(
            start_ts=self._start_ts,
            name=new_name,
            signal=new_signal,
            time=new_time,
            freq=int(new_freq) # should never be a float...
            # TODO support for non-integer frequencies?
        )
    
    def to_DataFrame(self) -> pd.DataFrame:
        assert len(self.signal) == len(self.time)
        return pd.DataFrame(
            data=np.array([self.time, self.signal]).T,
            columns=['time', self.name]
        )

    def get_rolling_mean(self, window_sec=30, step_size=1) -> Self:
        rolling_mean = pd.Series(self.signal).rolling(window_sec*self.freq, center=True)\
            .mean()[::self.freq].values
        return self._return(rolling_mean, step_size)

    def get_rolling_std(self, window_sec=30, step_size=1) -> Self:
        rolling_std = pd.Series(self.signal).rolling(window_sec*self.freq, center=True)\
            .std()[::self.freq].values
        return self._return(rolling_std, step_size)
    
    def _apply_rolling(self, window_sec, step_size, process) -> np.array:
        window_length = window_sec * self.freq
        step_idx = int(step_size * self.freq)

        accum = []
        for i in range(0, len(self.signal), step_idx):
            window_start = i - window_length//2
            window_end = i + window_length//2
            if window_start < 0:
                accum.append(np.nan)
            elif window_end > len(self.signal):
                accum.append(np.nan)
            else:
                accum.append(process(self.signal, window_start, window_end))
        return np.array(accum)

    def get_rolling_band_power_multitaper(self, freq_range=(0.5, 4), ref_power=1e-13,
                                           window_sec=2, step_size=1, in_dB=True) -> Self:
        """
        Gets rolling band power for specified frequency range, data frequency and window size
        freq_range: range of frequencies in form of (lower, upper) to calculate power of
        ref_power: arbitrary reference power to divide the windowed delta power by (used for scaling)
        window_sec: window size in seconds to calculate delta power (if the window is longer than the step size there will be overlap)
        step_size: step size in seconds to calculate delta power in windows (if 1, function returns an array with 1Hz power calculations)
        in_dB: boolean for whether to convert the output into decibals
        """
        def get_band_power_multitaper(a, start, end) -> np.array:
            a = a[start:end]
            # TODO: maybe edit this later so there is a buffer before and after?
            psd, freqs = mne.time_frequency.psd_array_multitaper(a, sfreq=self.freq,
                                fmin=freq_range[0], fmax=freq_range[1], adaptive=True, 
                                normalization='full', verbose=False)
            freq_res = freqs[1] - freqs[0]
            # Find the index corresponding to the delta frequency range
            delta_idx = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            # Integral approximation of the spectrum using parabola (Simpson's rule)
            delta_power = psd[delta_idx] / ref_power
            if in_dB:
                delta_power = simpson(10 * np.log10(delta_power), dx=freq_res)
            else:
                delta_power = np.mean(delta_power)
            # Sum the power within the delta frequency range
            return delta_power

        rolling_band_power = self._apply_rolling(
            window_sec=window_sec,
            step_size=step_size,
            process=get_band_power_multitaper
        )
        return self._return(rolling_band_power, step_size=step_size)

    def get_rolling_zero_crossings(self, window_sec=1, step_size=1) -> Self:
        """
        Get the zero-crossings of an array with a rolling window
        window_sec: window in seconds
        step_size: step size in seconds (step_size of 1 would mean returend data will be 1 Hz)
        """

        def get_crossing(a, start, end):
            return ((a[start:end-1] * a[start+1:end]) < 0).sum()
        
        rolling_zero_crossings = self._apply_rolling(
            window_sec=window_sec,
            step_size=step_size,
            process=get_crossing
        )
        return self._return(rolling_zero_crossings, step_size=step_size)
  
    def get_rolling_band_power_fourier_sum(self, freq_range=(0.5, 4), ref_power=0.001, window_sec=2, step_size=1) -> Self:
        """
        Gets rolling band power for specified frequency range, data frequency and window size
        freq_range: range of frequencies in form of (lower, upper) to calculate power of
        ref_power: arbitrary reference power to divide the windowed delta power by (used for scaling)
        window_sec: window size in seconds to calculate delta power (if the window is longer than the step size there will be overlap)
        step_size: step size in seconds to calculate delta power in windows (if 1, function returns an array with 1Hz power calculations)
        """
        def get_band_power_fourier_sum(a, start, end) -> np.array:
            a = a[start:end]
            """
            Helper function to get delta spectral power for one array
            """
            # Perform Fourier transform
            fft_data = np.fft.fft(a)
            # Compute the power spectrum
            power_spectrum = np.abs(fft_data)**2
            # Frequency resolution
            freq_resolution = self.freq / len(a)
            # Find the indices corresponding to the delta frequency range
            delta_freq_indices = np.where((np.fft.fftfreq(len(a), 1/self.freq) >= freq_range[0]) & 
                                        (np.fft.fftfreq(len(a), 1/self.freq) <= freq_range[1]))[0]
            # Compute the delta spectral power
            delta_power = np.sum(power_spectrum[delta_freq_indices] / ref_power) * freq_resolution

            return delta_power

        rolling_band_power = self._apply_rolling(
            window_sec=window_sec,
            step_size=step_size,
            process=get_band_power_fourier_sum
        )
        return self._return(rolling_band_power, step_size=step_size)
    
    def get_rolling_band_power_welch(self, freq_range=(0.5, 4), ref_power=0.001, window_sec=2, step_size=1) -> Self:
        """
        Gets rolling band power for specified frequency range, data frequency and window size
        freq_range: range of frequencies in form of (lower, upper) to calculate power of
        ref_power: arbitrary reference power to divide the windowed delta power by (used for scaling)
        window_sec: window size in seconds to calculate delta power (if the window is longer than the step size there will be overlap)
        step_size: step size in seconds to calculate delta power in windows (if 1, function returns an array with 1Hz power calculations)
        """
        def get_band_power_welch(a, start, end):
            lower_freq = freq_range[0]
            upper_freq = freq_range[1]
            window_length = int(window_sec * self.freq)
            # TODO: maybe edit this later so there is a buffer before and after?
            windowed_data = a[start:end] * hann(window_length)
            freqs, psd = welch(windowed_data, window='hann', fs=self.freq, nperseg=window_length, noverlap=window_length//2)
            freq_res = freqs[1] - freqs[0]
            # Find the index corresponding to the delta frequency range
            delta_idx = (freqs >= lower_freq) & (freqs <= upper_freq)
            # Integral approximation of the spectrum using parabola (Simpson's rule)
            delta_power = simpson(10 * np.log10(psd[delta_idx] / ref_power), dx=freq_res)
            # Sum the power within the delta frequency range
            return delta_power

        rolling_band_power = self._apply_rolling(
            window_sec=window_sec,
            step_size=step_size,
            process=get_band_power_welch
        )
        return self._return(rolling_band_power, step_size=step_size)
    

    def get_heart_rate(self, search_radius=200):
        """
        Gets heart rate at 1 Hz and extrapolates it to the same frequency as input data
        search_radius: search radius to look for peaks (200 ~= 150 bpm upper bound)
        """
        rpeaks = detect_heartbeats(self.signal, self.freq) # using sleepecg
        rpeaks_corrected = wfdb.processing.correct_peaks(
            self.signal, rpeaks, search_radius=search_radius, smooth_window_size=50, peak_dir="up"
        )
        # MIGHT HAVE TO UPDATE search_radius
        heart_rates = [60 / ((rpeaks_corrected[i+1] - rpeaks_corrected[i]) / self.freq) for i in range(len(rpeaks_corrected) - 1)]
        # Create a heart rate array matching the frequency of the ECG trace
        hr_data = np.zeros_like(self.signal)
        # Assign heart rate values to the intervals between R-peaks
        for i in range(len(rpeaks_corrected) - 1):
            start_idx = rpeaks_corrected[i]
            end_idx = rpeaks_corrected[i+1]
            hr_data[start_idx:end_idx] = heart_rates[i]

        return self._return(hr_data, step_size=1)
    


# config should map channels to their types: https://mne.tools/dev/generated/mne.channel_type.html
    
# Workflow
# load EDF, set global time splice range
# call processes by channel


class EDFutils:
    def __init__(self, file, eog_ch_names=None, misc_ch_names=None) -> None:
        
        # TODO consider multiple frequencies of data, should probs be specified on creation

        self.channel_types = {}
        self.eog_channels = eog_ch_names
        self.misc_channels = misc_ch_names
        self._channel_master = {}
        self._thresh = (None, None)
        raws = []
        if eog_ch_names:
            self._raw_eog = mne.io.read_raw_edf(file, include=eog_ch_names, eog=eog_ch_names, preload=False)
            raws.append(self._raw_eog)
            self.channel_types['eog'] = self._raw_eog.ch_names
        if misc_ch_names:
            self._raw_misc = mne.io.read_raw_edf(file, include=misc_ch_names, misc=misc_ch_names, preload=False)
            raws.append(self._raw_misc)
            self.channel_types['misc'] = self._raw_misc.ch_names

        self.start_ts = raws[0].info['meas_date'].replace(tzinfo=None)
        self._get_channel_type = {ch: k for k, v in self.channel_types.items() for ch in v}

        # Create a dictionary that points to functions that extract channel data 
        # _channel_master is referenced by the __getitem__ method
        for raw in raws:
            for i, ch in enumerate(raw.ch_names):
                # need to declare i and raw in scope of lambda else the func 
                # takes the value of the variable when the func is called
                self._channel_master[ch] = lambda i=i, raw=raw: raw[i]

    def __getitem__(self, item) -> Channel:
        # update value of item to cache it for subsequent calls
        value = self._channel_master[item]
        freq = self.channel_frequency(item)
        
        start_ts = self.start_ts
        start_idx, end_idx = self._thresh
        if start_idx and end_idx:
            start_ts = start_ts + timedelta(seconds=start_idx)
            start_idx = int(start_idx*freq)
            end_idx = int(end_idx*freq)

        if callable(value):
            signal, time = value()
            value = Channel(
                start_ts=start_ts,
                name=item,
                signal=signal[0],
                time=time,
                freq=freq
            )[start_idx:end_idx]
            self._channel_master[item] = value
        return value
    
    def set_date_threshold(self, start, end):
        front = (start - self.start_ts).total_seconds()
        back = (end - self.start_ts).total_seconds()
        self._thresh = (int(front), int(back))

    def channel_frequency(self, channel_type) -> int:
        # TODO should not exist, to be offloaded to configuration
        match self._get_channel_type[channel_type]:
            case 'eog':
                return 500
            case 'misc':
                return 25
            case _:
                raise Exception('Unknown channel type')

# Raw Channel Types
    # EEG, EOG, EMG
        # vlf {func: fourier, window: 60}
            # vlf.std {func: std, window: 60}
        # delta_power {func: welch, window: 2, ref_power: 1e-14, freq_range: (0.5,4)}
        # welch_power {func: welch, window: 2, ref_power: 3e-14, freq_range: (0.4, 30)}
    # ECG
        # hr {func: hr} -- build in downsampling
            # hr.mean {func: mean, window: 30}
            # hr.std {func: std, window: 30}
    # Pitch
        # mean {func: mean, window: 1}
    # Roll
    # Heading
    # GyrX, GyrY, GyrZ       
    # MagZ
    # ODBA 
    # Pressure
    
# Possible features:
    # eeg.fourier_sum (vlf)
    # eeg.welch (delta power)
    # eeg.zero_crossings
    # ecg.heart_rate
    # ..mean
    # ..std

# TODO - This entire class
class TrainingSet():
    def __init__(self) -> None:
        pass

    def compute_features(self, feature_funcs: list) -> pd.DataFrame:
        # take list of method names, return an output that can be inserted to DB?
        pass

    def subsample(self, second_interval, features='all'):
        for feature in features:
            self.get(feature)

    def bin_by_window(data: pd.DataFrame, window_length: int):
        windowed = []
        for i in range(len(data)-1):
            windowed.append(data[i:i+window_length].values.T)
        return np.array(windowed)
