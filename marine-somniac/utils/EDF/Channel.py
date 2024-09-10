import numpy as np
import pandas as pd
import inspect
from datetime import timedelta
from .Base import Base
try:
    from typing import Self
except:
    from typing_extensions import Self


class Channel(Base):
    def __init__(self, start_ts, end_ts, name: str, signal: np.array, time:np.array=None, freq=None, type_=None) -> None:
        self.name = name
        self.time = time
        self.signal = signal
        self.freq = freq
        self.type = type_
        self.start_ts = start_ts
        self.end_ts = end_ts if end_ts else self.start_ts + timedelta(seconds=time[-1])
            
    def __getitem__(self, slice) -> Self:
        """
        Enables object indexing, returns a new Channel instance with signal 
        and time attributes indexed according to the supplied slice
        """
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
            start_ts=self.start_ts,
            end_ts=self.end_ts
        )
    
    def time_slice(self, start_time, end_time, unit='second') -> Self:
        """
        Slices the data by time, returns a new Channel instance
        start_time: start time in the form of a number
        end_time: end time in the form of a number
        unit: unit of time to slice by, options are 'second', 'minute', 'hour'
        """
        mod = None
        match unit:
            case 'second':
                mod = 1
            case 'minute':
                mod = 60
            case 'hour':
                mod = 3600
            case _:
                raise ValueError(
                    f'Only accepts second, minute, and hour, not {unit}')
        
        start = mod * self.freq * start_time
        end = mod * self.freq * end_time
        return self[start:end]
    
    def date_slice(self, start_date, end_date) -> Self:
        """
        Slices the data by date, returns a new Channel instance
        start_date: start date in the form of a string or datetime object
        end_date: end date in the form of a string or datetime object
        """
        recording_start_ts = self.start_ts.timestamp()
        start_ts = pd.to_datetime(start_date).timestamp()
        end_ts = pd.to_datetime(end_date).timestamp()

        relative_start_ts = start_ts - recording_start_ts
        relative_end_ts = end_ts - recording_start_ts

        start_idx = np.searchsorted(self.time, relative_start_ts)
        end_idx = np.searchsorted(self.time, relative_end_ts, side='right')

        slice_signal = self.signal[start_idx:end_idx]
        slice_time = self.time[start_idx:end_idx]

        return self.__class__(
            name=self.name,
            signal=slice_signal,
            time=slice_time,
            freq=self.freq,
            start_ts=self.start_ts,
            end_ts=self.end_ts
        )

    
    def _return(self, new_signal, step_size) -> Self:
        """
        Used to generalize the return of window functions to minimize
        copy-pasting. Takes name of the process/method that calls it and uses 
        that to assign the name of the returned Channel object. 
        Calculates new frequency values based on input process modifications.
        new_signal: the new array to be assigned to Channel.signal
        step_size: step size of the window function used to calculate the new freq
        """
        # inspect.stack()[1][3] returns the name of the function
        # traced back before this function call
        new_name = f'{self.name}.{inspect.stack()[1][3]}'
        new_time = self.time[::self.freq//step_size]
        new_freq = step_size/1
        return self.__class__(
            start_ts=self.start_ts,
            end_ts=self.end_ts,
            name=new_name,
            signal=new_signal,
            time=new_time,
            freq=int(new_freq)  # should never be a float...
            # TODO support for non-integer frequencies?
        )
    
    def downsample(self, ds_freq:int=1):
        """
        ds_freq: frequency in Hz to which this channel will be downsampled
        """
        ss = self.freq//ds_freq
        ds_sig = self.signal[::ss]
        return self._return(ds_sig, step_size=ss)

    def get_rolling_mean(self, window_sec:int=30, step_size:int=1) -> Self:
        """
        Calculate rolling mean over Channel.signal. Returns new Channel instance
        window_sec: window size for rolling mean in seconds
        step_size: step over which to resample the output Channel
        """
        rolling_mean = pd.Series(self.signal).rolling(window_sec*self.freq, center=True)\
            .mean()[::self.freq].values
        return self._return(rolling_mean, step_size)

    def get_rolling_std(self, window_sec:int=30, step_size:int=1) -> Self:
        """
        Calculate rolling standard deviation over Channel.signal. 
        Returns new Channel instance
        window_sec: window size for rolling std in seconds
        step_size: step over which to resample the output Channel
        """
        rolling_std = pd.Series(self.signal).rolling(window_sec*self.freq, center=True)\
            .std()[::self.freq].values
        return self._return(rolling_std, step_size)
    
    def _apply_rolling(self, window_sec, step_size, process) -> np.array:
        """
        Generalized pattern to apply a transformation over a rolling window.
        window_sec: window size for applied process in seconds
        step_size: step over which to resample the signal frequency
        process: function to apply over the rolling window
        """
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
    
    def to_DataFrame(self) -> pd.DataFrame:
        """
        Returns 2-column pandas DataFrame of time and signal
        """
        assert len(self.signal) == len(self.time)
        return pd.DataFrame(
            data=np.array([self.time, self.signal]).T,
            columns=['time', self.name]
        )

    def visualize(self):
        """
        """
        pass

