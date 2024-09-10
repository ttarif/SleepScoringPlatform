## EDFutils Helper class

`EDFutils` aims to somewhat remedy a few headaches related to the implementation of `mne.io.raw` objects.

1. EDF files with mixed channel frequencies raise exceptions when accessing channels of different frequencies from the `raw` object via indexing.
2. The return type for the data arrays is a tuple containing a nested numpy array for signals and a 1-D array for time
3. Caching has been added so that all data remains bundled with its EDF class.


## Sample Usage
```
# defining channel names belonging to different frequencies

# 500 Hz
eog = ['ECG_Raw_Ch1', 'ECG_ICA2', 'LEOG_Pruned_Ch2', 'LEMG_Pruned_Ch4', 'REEG2_Pruned_Ch7', 
       'LEEG3_Pruned_Ch8', 'REEG2_Raw_Ch7', 'LEEG3_Raw_Ch8', 'EEG_ICA5']
 
# 25 Hz
misc = ['pitch', 'roll', 'heading',
 'GyrZ', 'MagZ', 'ODBA', 'Pressure']

edf = EDFutils('data/test12_Wednesday_05_ALL_PROCESSED.edf',
     eog_ch_names=eog, misc_ch_names=misc)

edf['ECG_ICA2'].signal
>> array([-0.00014266, -0.00014266, -0.00014266, ..., -0.0156284 ,
       -0.0156284 , -0.0156284 ])


edf['pitch'].time
>> array([0.0000000e+00, 4.0000000e-02, 8.0000000e-02, ..., 3.1791488e+05,
       3.1791492e+05, 3.1791496e+05])
```

## Channel Objects

Additionally, accessing a channel (calling `edf['pitch']` for example) returns a `Channel` object with built-in processing methods for feature extraction. The following methods return `Channel` objects themselves allowing for method chaining:

```
Channel.get_rolling_mean()
Channel.get_rolling_std()
Channel.get_rolling_band_power_multitaper()
Channel.get_rolling_band_power_welch()
Channel.get_rolling_zero_crossings()
Channel.get_rolling_band_power_fourier_sum()
Channel.get_heart_rate()

# chaining example: VLF Power STD
edf['EEG'].get_rolling_band_power_fourier_sum(window_sec=60)\
       .get_rolling_mean_std(window_sec=60).to_DataFrame()
```

Channel objects store following attributes:
* `signal`: a numpy array containing sensor data
* `time`: a numpy array containing the second interval at which the sensor data is sampled
* `name`: the channel name (updated upon method chaining for traceability) 
* `freq`: the frequency in Hz of the channel **(inferred from specification of eog or misc in EDFutils definition)**
* `_start_ts`: the timestamp at which the first datapoint was collected (updates with splicing and chaining)


### Indexing & Time Slicing

Indexing a `Channel` object also returns a `Channel` object. The `time_slice` method allows for the input of specific time values so as to prevent the user from needing to calculate positional indices based on the channel frequency.

```
frequency = edf['roll'].freq
hours = 60*60*frequency

edf['roll'][0:2*60]

# OR

edf['roll'].time_slice(0, 2, 'hour')
```

The default range of a `Channel` can also be constrained by setting a bracket in the EDFutils object:

```
# get start and end timestamps from the label dataset
start, end = label_df['R.Time'].values[[0,-1]]

# convert date strings to datetime objects
start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
end = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')

# force all future channel calls to be pre-sliced according to
# these specified timestamps
edf.set_date_threshold(start, end)

edf.start_ts
>> datetime.datetime(2019, 10, 25, 8, 21, 2)

edf['roll']._start_ts
>> datetime.datetime(2019, 10, 25, 14, 45, 22)
```
