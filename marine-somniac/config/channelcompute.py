from utils.EDF.EDF import Channel, EXGChannel, ECGChannel
from utils.EDF.Epoch import Epoch
from utils.EDF.SpectralDensity import SpectralDensity

CHANNEL_TYPES = [
    "EEG",
    "ECG",
    "Pressure",
    "ODBA",
    "Gyroscope",
    "Other"
]

BASIC = {
    'Mean': Channel.get_rolling_mean,
    'Standard Deviation': Channel.get_rolling_std,
}
EXG_DERIVED = {'Epoch': EXGChannel.get_epoch}
HEARTRATE_DERIVED = {'Epoch (HR)': ECGChannel.get_hr_epoch, **BASIC}
EEG_EPOCH_DERIVED = {
    "Welch's PSD": Epoch.get_welch,
    'Hjorth Parameters': Epoch.get_hjorth_params,
    'Permutation Entropy': Epoch.get_permutation_entropy,
    'Higuchi Fractal Dimension': Epoch.get_higuchi,
    'Petrosian Fractal Dimension': Epoch.get_petrosian,
    'Zero Crossings': Epoch.get_zero_crossings,
    'Standard Deviation_': Epoch.get_std,
    'Interquartile Range': Epoch.get_interquartile_range,
    'Kurtosis': Epoch.get_kurtosis,
    'Skewness': Epoch.get_skew,
}
HR_EPOCH_DERIVED = {
    "Welch's PSD (HR)": Epoch.get_hr_welch,
    'Hjorth Parameters': Epoch.get_hjorth_params,
    'Permutation Entropy': Epoch.get_permutation_entropy,
    'Higuchi Fractal Dimension': Epoch.get_higuchi,
    'Petrosian Fractal Dimension': Epoch.get_petrosian,
    'Zero Crossings': Epoch.get_zero_crossings,
    'Standard Deviation_': Epoch.get_std,
    'Interquartile Range': Epoch.get_interquartile_range,
    'Kurtosis': Epoch.get_kurtosis,
    'Skewness': Epoch.get_skew,
}
WELCH_DERIVED = {
    'Power Ratios': SpectralDensity.get_power_ratios,
    'Absolute Power': SpectralDensity.get_absolute_power,
    'Power Standard Deviation': SpectralDensity.get_power_std,
    'Relative Powers': SpectralDensity.get_relative_powers,
    'Total Power': SpectralDensity.get_total_power
}
ECG_DERIVED = {'Heart Rate': ECGChannel.get_heart_rate}
NOT_CONFIGURABLE = [
    'Hjorth Parameters',
    'Permutation Entropy',
    'Higuchi Fractal Dimension',
    'Petrosian Fractal Dimension',
    'Zero Crossings',
    'Standard Deviation_',
    'Interquartile Range',
    'Kurtosis',
    'Skewness',
    'Power Ratios',
    'Absolute Power',
    'Power Standard Deviation',
    'Relative Powers',
    'Total Power'
]
CUSTOM_ARGSPEC = ['get_welch', 'get_hr_welch']



DERIVANDS = {
    'Pressure': {**BASIC}, 'ODBA': {**BASIC}, 'Gyroscope': {**BASIC}, 'Other': {**BASIC},
    'EEG': {**EXG_DERIVED, **BASIC},
    'ECG': {**ECG_DERIVED, **EXG_DERIVED, **BASIC},
    'Epoch': EEG_EPOCH_DERIVED,
    'Epoch (HR)': HR_EPOCH_DERIVED,
    'Heart Rate': HEARTRATE_DERIVED,
    "Welch's PSD": WELCH_DERIVED 
}
LABEL_TO_METHOD = {**BASIC, **EXG_DERIVED, **ECG_DERIVED, **HEARTRATE_DERIVED, **EEG_EPOCH_DERIVED, **HR_EPOCH_DERIVED, **WELCH_DERIVED}

FEATURE_OPTIONS = {
    'all': {**BASIC, **EXG_DERIVED, **ECG_DERIVED, **HEARTRATE_DERIVED, **EEG_EPOCH_DERIVED, **HR_EPOCH_DERIVED, **WELCH_DERIVED},
    'EEG': {**EXG_DERIVED, **BASIC},
    'ECG': {**ECG_DERIVED, **EXG_DERIVED, **BASIC},
    'Pressure': {**BASIC},
    'ODBA': {**BASIC},
    'Gyroscope': {**BASIC},
    'Other': {**BASIC}
}