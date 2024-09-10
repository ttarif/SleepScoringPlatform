from datetime import timedelta, datetime
import pandas as pd
import mne
from .Channel import Channel
from .EXGChannel import EXGChannel
from .ECGChannel import ECGChannel
try:
    from typing import Self
except:
    from typing_extensions import Self


class EDFutils:
    def __init__(self, filepath, fetch_metadata=True, config:dict=None) -> None:
        self._route_object = {
            'Other': Channel,
            'Gyroscope': Channel,
            'Pressure': Channel,
            'ODBA': Channel,
            'EEG': EXGChannel,
            'ECG': ECGChannel
        }
        
        self.filepath = filepath
        self.time_range = (None, None)
        self.channel_types = {}

        if fetch_metadata:
            with mne.io.read_raw_edf(filepath, preload=False) as raw:
                self.channels = raw.ch_names
                self.start_ts = raw.info['meas_date'].replace(tzinfo=None)
                self.end_ts = self.start_ts + timedelta(seconds=raw.times[-1])

            self.channel_freqs = {ch: self.get_channel_frequency(ch) for ch in self.channels}
        elif config is None:
            raise Exception("A configuration must be passed if `fetch_metadata` is set to False")
        else:
            self.channels = config['channels']['picked']
            self.start_ts = datetime.strptime(config['raw_time']['start'], '%Y-%m-%d %H:%M:%S.%f')
            self.end_ts = datetime.strptime(config['raw_time']['end'], '%Y-%m-%d %H:%M:%S.%f')
            start = datetime.strptime(config['time']['start'], '%Y-%m-%d %H:%M:%S.%f')
            end = datetime.strptime(config['time']['end'], '%Y-%m-%d %H:%M:%S.%f')
            self.set_date_range(start, end)

            ch_types = {}
            ch_freqs = {}
            for ch, tup in config['channels_'].items():
                ch_type, ch_freq = tup
                ch_freqs[ch] = ch_freq
                ch_types[ch] = ch_type
            self.channel_freqs = ch_freqs
            self.channel_types = ch_types

    def __getitem__(self, item) -> Channel:
        if item not in self.channels:
            raise KeyError(f"`{item}` not a channel in EDF file '{self.filepath}'")
        else:
            with mne.io.read_raw_edf(self.filepath, include=[item], preload=False) as raw:
                if all(self.time_range):
                    raw.crop(tmin=self.time_range[0], tmax=self.time_range[1])
                signal, time = raw[0]

            channel_obj = Channel
            if self.channel_types:
                ch_type = self.channel_types[item]
                channel_obj = self._route_object.get(ch_type)
                if channel_obj is None:
                    raise Exception(f"Does not accept `{ch_type}` as channel type "
                                    "only EEG, ECG, Motion, and Other")

            return channel_obj(
                start_ts=self.start_ts,
                end_ts=self.end_ts,
                name=item,
                signal=signal[0],
                time=time,
                freq=self.channel_freqs[item],
                type_=self.channel_types.get(item)
            )
        
    def get_channel_frequency(self, ch_name) -> int:
        with mne.io.read_raw_edf(self.filepath, include=[ch_name], preload=False) as raw:
            freq = len(raw.crop(tmin=0, tmax=1).pick(ch_name).get_data()[0])-1
        return freq

    # TODO
    def resample(self, sfreq, ch_names=None) -> Self:
        """
        Resamples the EDF file to a new sampling frequency and optionally picks specific channels
        sfreq: sampling frequency to resample to
        ch_names: list of channel names to pick (if None, all channels are picked)
        """
        return

    def set_date_range(self, start: datetime, end: datetime) -> None:
        """
        After setting this threshold, any Channels accessed will be spliced
        according to these supplied dates.
        start: datetime object representing start date
        end: datetime object representing end date
        """
        # TODO update using Will's date slice code to accept strings
        front = (start - self.start_ts).total_seconds()
        back = (end - self.start_ts).total_seconds()
        self.time_range = (int(front), int(back))

    # TODO
    def to_DataFrame(self, frequency:int, channels:list=None) -> pd.DataFrame:
        """
        Exports channels to a pandas DataFrame wherein each channel is a column.
        frequency: the desired output frequency to sample all data to
        channels: channels to export, default exports all
        """
        pass
