#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import yasa
# import mne
import os
import sys
# import scipy
# import glob
# import six
# import wfdb
import pytz
# import sklearn
# import pomegranate
# import pyedflib
# import sleepecg
# import datetime
# import wfdb.processing
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# #import entropy as ent
# import seaborn as sns
# from matplotlib import mlab as mlab
from sleepecg import detect_heartbeats
# import matplotlib.gridspec as gs


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


# Add the src directory to the path
current_path = os.getcwd()
src_path = os.path.abspath(os.path.join(current_path, '..', 'src'))
sys.path.insert(0, src_path) 
from feature_extraction import *


# # Table of Contents
# ## [Wednesday_First_Day Feature Extraction](#first_day_wednesday)
# ### [SKlearn train test split](#sklearn_first_day)
# ### [SVM Classifier](#svm_classifier)
# ### [Random Forest](#random_forest)
# ## [Full Wednesday Testing](#full_wednesday)
# ### [Explore Errors](#model_error_exploration)
# ### [Error Inspection](#error_inspection)

# In[4]:


# Construct the relative path to the folder containing processed data
data_path = os.path.abspath(os.path.join("..", "data"))
process_data_path = os.path.abspath(os.path.join("..", "data", "01_processed-data"))
print(process_data_path)

# Read the header information to identify channels and their sampling frequencies
info = mne.io.read_raw_edf(f'{process_data_path}/test12_Wednesday_05_ALL_PROCESSED.edf',
                           preload=False).info

# Print the channel information
print(info)

# Identify channels and their corresponding sampling frequencies
channels_info = info['chs']
sampling_freq_map = {}


# In[5]:


# Load the EDF file, excluding the EOGs and EKG channels
raw = mne.io.read_raw_edf(f'{process_data_path}/test12_Wednesday_05_ALL_PROCESSED.edf', preload=True)
# raw.resample(100)                      # Downsample the data to 100 Hz
# raw.filter(0.1, 40)                    # Apply a bandpass filter from 0.1 to 40 Hz
# raw.pick_channels(['C4-A1', 'C3-A2'])  # Select a subset of EEG channels
raw # Outputs summary data about file

# Inspect Data
print(raw.info)
print('The channels are:', raw.ch_names)
print('The sampling frequency is:', raw.info['sfreq'])

# Rename channels (replace spaces if any)
channel_renaming_dict = {name: name.replace(' ', '_') for name in raw.ch_names}
raw.rename_channels(channel_renaming_dict)
print('The channels are:', raw.ch_names)

# ['ECG_Raw_Ch1', 'ECG_ICA2', 'LEOG_Pruned_Ch2', 'LEMG_Pruned_Ch4', 'REEG2_Pruned_Ch7', 'LEEG3_Pruned_Ch8', 
# 'REEG2_Raw_Ch7', 'LEEG3_Raw_Ch8', 'EEG_ICA5', 'pitch', 'roll', 'heading', 'GyrZ', 'MagZ', 'ODBA', 'Pressure']

# Assuming 'raw' is your Raw object from MNE
channel_types = {}

for ch in raw.ch_names:
    if ch.startswith('ECG'):
        channel_types[ch] = 'ecg'
    elif ch.startswith(('LEOG', 'REOG')):
        channel_types[ch] = 'eog'
    elif ch.startswith(('LEMG', 'REMG')):
        channel_types[ch] = 'emg'
    elif ch.startswith(('LEEG', 'REEG')):
        channel_types[ch] = 'eeg'
    elif ch in ['pitch', 'roll', 'heading']:
        channel_types[ch] = 'resp'
    elif ch in ['GyrZ', 'MagZ', 'ODBA']:
        channel_types[ch] = 'syst'
    elif ch in ['Pressure']:
        channel_types[ch] = 'misc'
    elif ch == 'Heart_Rate':
        channel_types[ch] = 'bio'

# Now set the channel types
raw.set_channel_types(channel_types)

# Inspect Data
print(raw.info)
print('The channels are:', raw.ch_names)
print('The sampling frequency is:', raw.info['sfreq'])

# Extract the measurement date (start time) from raw.info
start_time = raw.info['meas_date']
fs = raw.info['sfreq']

# Define the PST timezone
pst_timezone = pytz.timezone('America/Los_Angeles')

# Convert to datetime object in PST
if isinstance(start_time, datetime.datetime):
    # If it's already a datetime object, just replace the timezone
    recording_start_datetime = start_time.replace(tzinfo=pst_timezone)
elif isinstance(start_time, (int, float)):
    # Convert timestamp to datetime in PST
    recording_start_datetime = datetime.datetime.fromtimestamp(start_time, pst_timezone)
else:
    # Handle other formats if necessary
    pass

# Calculate the recording duration in seconds
recording_duration_seconds = len(raw) / fs

# Calculate the recording end datetime
recording_end_datetime = recording_start_datetime + datetime.timedelta(seconds=recording_duration_seconds)

# Calculate duration as a timedelta object
duration_timedelta = datetime.timedelta(seconds=recording_duration_seconds)

# Create a time index
#time_index = pd.date_range(recoring_start_datetime, recording_end_datetime)

# Format duration into days, hours, minutes, and seconds
days = duration_timedelta.days
hours, remainder = divmod(duration_timedelta.seconds, 3600)
minutes, seconds = divmod(remainder, 60)

print('The start time in PST (Los Angeles) is:', recording_start_datetime)
print('The end time in PST (Los Angeles) is:', recording_end_datetime)
print(f'Duration: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds')


# In[6]:


print(recording_start_datetime)
print(duration_timedelta)
print(recording_end_datetime)


# In[7]:


# Load labeled data
# Path to CSV with scored data
file_path = f'{data_path}/02_annotated-data/test12_Wednesday_06_Hypnogram_JKB_1Hz.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)
df['R.Time'] = pd.to_datetime(df['R.Time']).dt.tz_localize('America/Los_Angeles')
df['Sleep.Code'].value_counts(normalize=True)


# In[8]:


df['R.Time'].min()


# In[9]:


df['R.Time'].max()


# <a id='first_day_wednesday'></a>
# 
# # Calculate features for the Wednesday's labelled time period up to 10/26/19

# In[10]:


start_time = datetime.datetime.strptime('10/26/2019 00:00:00', '%m/%d/%Y %H:%M:%S')
# just get 10/25
subset = df[df['R.Time'].dt.tz_localize(None)  < start_time]
start_index = int((subset['R.Time'].min() - recording_start_datetime).total_seconds() * 500)
end_index = int(((subset['R.Time'].max() - recording_start_datetime).total_seconds() + 1) * 500)


# In[11]:


# EEG and ECG for just the subset: 10/25/19 2:45 pm - midnight


# In[12]:


(end_index - start_index) / 500 / 60 / 60


# In[13]:


eeg_subset = raw.copy().pick(['EEG_ICA5']).get_data()[0, start_index:end_index]
ecg_subset = raw.copy().pick(['ECG_ICA2']).get_data()[0, start_index:end_index] # TODO: Use ECG_Raw_Ch1
len(eeg_subset) / 500


# In[14]:


len(subset)


# In[15]:


# heart rate for the subset time period
hr_subset = get_heart_rate(ecg_subset)


# In[16]:


delta_power = get_rolling_band_power_welch(eeg_subset, 0, len(eeg_subset), freq_range=(0.5, 4), ref_power=1e-14,
                                           freq=500, window_sec=30, step_size=1)
print('done')
zero_crossings = get_rolling_zero_crossings(eeg_subset, 0, len(eeg_subset), window_sec=10)
print('done')
absolute_power = get_rolling_band_power_welch(eeg_subset, 0, len(eeg_subset), freq_range=(0.4, 30),
                                                 ref_power=1e-14, freq=500, window_sec=30, step_size=1)
print('done')
hr_subset = get_heart_rate(ecg_subset)
hr_mean, hr_std = get_rolling_mean_std(hr_subset, 0, len(hr_subset), window_sec=30, freq=500)
print('done')
vlf_power = get_rolling_band_power_fourier_sum(hr_subset, 0, len(hr_subset), freq_range=(0.001, 0.05),
                                               window_sec=30, freq=500, ref_power=1)
print('done')
_, vlf_power_std = get_rolling_mean_std(vlf_power, 0, len(vlf_power), freq=1, window_sec=60)


# In[17]:


sw1_filter = subset['Sleep.Num'] == 4
sw2_filter = subset['Sleep.Num'] == 5
# sw2_filter = np.array([[x] * 500 for x in sw2_filter]).flatten()
rem_filter = subset['Sleep.Num'] == 7
# rem_filter = np.array([[x] * 500 for x in rem_filter]).flatten()
drowsy_filter = subset['Sleep.Num'] == 3


# In[18]:


for myarr in [delta_power, zero_crossings, absolute_power, hr_subset, hr_mean, hr_std, vlf_power, vlf_power_std]:
    print(len(myarr))


# In[19]:


features_v1 = pd.DataFrame({
    'Delta Power': delta_power,
    'Rolling Zero Crossings': zero_crossings,
    'Rolling Absolute Power': absolute_power,
    'Heart Rate': [hr_subset[i] for i in range(0, len(hr_subset), 500)], # Downsample from 500 Hz to 1 Hz
    'Heart Rate Mean': hr_mean,
    'Heart Rate Std.Dev': hr_std,
    'Heart Rate Very Low Frequency Power': vlf_power,
    'Heart Rate VLF Power Std.Dev': vlf_power_std,
    'Sleep.Num': subset['Sleep.Num']
})


# In[20]:


features_v1.index = subset['R.Time']


# In[21]:


features_v1


# In[22]:


features_v1.to_csv('../training_sets/features_v1.csv')


# In[23]:


with open('../training_sets/features_v1_meta.txt', 'w') as f: 
    f.write(
"""
delta_power = get_rolling_band_power_welch(eeg_subset, 0, len(eeg_subset), freq_range=(0.5, 4), ref_power=1e-14,
                                           freq=500, window_sec=30, step_size=1)
zero_crossings = get_rolling_zero_crossings(eeg_subset, 0, len(eeg_subset), window_sec=10)
absolute_power = get_rolling_band_power_welch(eeg_subset, 0, len(eeg_subset), freq_range=(0.4, 30),
                                                 ref_power=1e-14, freq=500, window_sec=30, step_size=1)
hr_subset = get_heart_rate(ecg_subset)
hr_mean, hr_std = get_rolling_mean_std(hr_subset, 0, len(hr_3subset), window_sec=30, freq=500)
vlf_power = get_rolling_band_power_fourier_sum(hr_subset, 0, len(hr_subset), freq_range=(0.001, 0.05),
                                               window_sec=30, freq=500, ref_power=1)
_, vlf_power_std = get_rolling_mean_std(vlf_power, 0, len(vlf_power), freq=1, window_sec=60)

features_v1 = pd.DataFrame({
    'Delta Power': delta_power,
    'Rolling Zero Crossings': zero_crossings,
    'Rolling Absolute Power': absolute_power,
    'Heart Rate': [hr_subset[i] for i in range(0, len(hr_subset), 500)], # Downsample from 500 Hz to 1 Hz
    'Heart Rate Mean': hr_mean,
    'Heart Rate Std.Dev': hr_std,
    'Heart Rate Very Low Frequency Power': vlf_power,
    'Heart Rate VLF Power Std.Dev': vlf_power_std,
    'Sleep.Num': subset['Sleep.Num']
})
"""
)


# <a id='sklearn_first_day'></a>
# 
# # Scikit-Learn on Wednesday first day

# In[24]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV


# In[28]:


model_train_data = features_v1.dropna()
train_X, test_X, train_y, test_y = train_test_split(model_train_data.drop('Sleep.Num', axis=1),
                                                    model_train_data['Sleep.Num'], test_size=0.25, shuffle=False)


# <a id='svm_classifier'></a>
# 
# ## SVC

# In[ ]:


param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'sigmoid']} # kernel: 'poly' excluded
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(train_X,train_y)


# In[30]:


print(grid.best_params_)


# In[77]:


model_svc = SVC(C=0.1, gamma=1, kernel='rbf')


# In[78]:


def print_model_stats(model, data):
    train_X, test_X, train_y, test_y = train_test_split(data.drop('Sleep.Num', axis=1),
                                                        data['Sleep.Num'])
    from IPython.display import display, HTML
    model.fit(train_X, train_y)
    preds = model.predict(test_X)
    print('Accuracy:')
    print(np.mean(preds == test_y))
    print()
    
    print('Actual Target Dist')
    print(test_y.value_counts(normalize=True).sort_index())
    print()
    
    print('Prediction Target Dist')
    print(pd.Series(preds).value_counts(normalize=True).sort_index())
    print()
    
    display(pd.DataFrame(
        confusion_matrix(test_y, preds),
        index=[f'Actual {x}' for x in range(1, 8)],
        columns=[f'Predicted {x}' for x in range(1, 8)]
    ))


# In[79]:


print_model_stats(model_svc, model_train_data)


# In[80]:


df[['Sleep.Code', 'Sleep.Num']].drop_duplicates().set_index('Sleep.Num', drop=True).sort_index()


# In[81]:


preds = model_svc.predict(test_X)
preds_with_actual = pd.DataFrame(preds, columns=['Predicted'], index=test_y.index)
preds_with_actual['Actual'] = test_y.values
preds_with_actual = preds_with_actual.sort_index()
sns.lineplot(preds_with_actual['Actual'].iloc[:60*60], color='black', linewidth=3, label='Actual')
sns.lineplot(preds_with_actual['Predicted'].iloc[:60*60], color='gold', linewidth=1, label='Predicted')
plt.title('Predicted Sleep Num vs Actual; first hour')
plt.legend()
plt.show()


# In[82]:


df_test = model_train_data.merge(preds_with_actual, how='inner', left_index=True, right_index=True)
df_test.head(3)


# <a id='random_forest'></a>
# 
# ## Random Forest

# In[44]:


model_rf = RandomForestClassifier()


# In[ ]:


param_grid = {
    'n_estimators': [25, 50, 100, 200, 500],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 50, 25, 10, 5]
}
grid = GridSearchCV(RandomForestClassifier(),param_grid,refit=True,verbose=2,n_jobs=8)
grid.fit(train_X,train_y)


# In[47]:


grid.best_params_


# In[48]:


model_rf = RandomForestClassifier(criterion='log_loss', n_estimators=50, max_depth=50, n_jobs=8)


# In[49]:


print_model_stats(model_rf, model_train_data)


# In[ ]:


df[['Sleep.Code', 'Sleep.Num']].drop_duplicates().set_index('Sleep.Num', drop=True).sort_index()


# In[ ]:


# from sklearn.feature_selection import RFE
# rfe = RFE(RandomForestClassifier(criterion='entropy', n_estimators=500, n_jobs=8), *,
#           n_features_to_select=1, step=1, verbose=1, importance_getter='auto')


# <a id='full_wednesday'></a>
# 
# # Testing on full Wednesday file

# In[50]:


from feature_generation import generate_features


# In[51]:


path_to_edf = os.path.abspath('../data/01_processed-data/test12_Wednesday_05_ALL_PROCESSED.edf')


# In[52]:


features_all = generate_features(path_to_edf)


# In[53]:


features_all['Time'] = features_all['Time'].dt.tz_convert('America/Los_Angeles')


# In[54]:


model_rf = RandomForestClassifier(criterion='entropy', n_estimators=500, max_depth=50, n_jobs=8)


# In[55]:


start_train_dt = df['R.Time'].iloc[0]
end_train_dt = df['R.Time'].iloc[int(0.75 * len(df) - 1)]
start_test_dt = df['R.Time'].iloc[int(0.75 * len(df))]
stop_test_dt = df['R.Time'].iloc[-1]
print(start_train_dt, end_train_dt, start_test_dt, stop_test_dt, sep='\n')


# In[56]:


# Data with target
features_subset = features_all[(features_all['Time'] >= df['R.Time'].iloc[0]) & 
                               (features_all['Time'] <= df['R.Time'].iloc[-1])]
features_subset = features_subset.set_index('Time', drop=True)
features_subset['Sleep.Num'] = df.set_index('R.Time', drop=True)['Sleep.Num']


# In[57]:


# Train test split
# don't want to shuffle inside the train test split because I want the test data to be the full final 1/4
# of the data (the training set is about 3 days and the test is 1 day)
train, test = train_test_split(features_subset, train_size=0.75, shuffle=False)
train = train.sample(frac=1).dropna() # Drop any rows with NA values
test = test.sample(frac=1).dropna() # Drop any rows with NA values
X_train, y_train = train.drop('Sleep.Num', axis=1), train['Sleep.Num']
X_test, y_test = test.drop('Sleep.Num', axis=1), test['Sleep.Num']


# In[58]:


# Make predictions and evaluate whether the errors are one-offs or in blocks
model_rf.fit(X_train, y_train)
preds = model_rf.predict(X_test)
preds


# <a id='model_error_exploration'></a>
# 
# ## Explore errors - are they usually together in a row or one-offs?

# In[60]:


test_with_preds = X_test.copy()
test_with_preds['Sleep.Num'] = y_test
test_with_preds['Predicted'] = preds
test_with_preds = test_with_preds.sort_index()
test_with_preds['IsCorrect'] = test_with_preds['Sleep.Num'] == test_with_preds['Predicted']


# In[61]:


isCorr = ~test_with_preds['IsCorrect']
inarowcounts = isCorr * (isCorr.groupby((isCorr != isCorr.shift()).cumsum()).cumcount() + 1)
error_window_ends = inarowcounts[inarowcounts > inarowcounts.shift(-1)]


# In[62]:


error_window_ends_weighted = [[x] * x for x in error_window_ends.tolist()]
flattened = []
for smallarr in error_window_ends_weighted:
    flattened += smallarr
sns.kdeplot(flattened)
plt.title('Distribution of how big the error window is for each error\n\
           (if an error belongs to a consecutive window of 500 errors, its value would be 500)')
plt.show()


# In[63]:


preds


# In[64]:


def get_label_changes(arr):
    arr.index = arr.index.strftime('%m/%d/%y %H:%M:%S')
    cur_val = arr.iloc[0]
    cnt = 1
    label_changes = [(arr.index[0]), cur_val]
    for i in range(1, len(arr)):
        if arr.iloc[i] != cur_val:
            cur_val = arr.iloc[i]
            label_changes.append((arr.index[i], cur_val))
    return label_changes
            
label_dists = []
pred_dists = []
start_times = []
end_times = []
long_errors = error_window_ends[error_window_ends > 200]
for dt, val in error_window_ends[error_window_ends > 200].items():
    start_dt_window = dt - datetime.timedelta(seconds=val)
    start_times.append(start_dt_window)
    end_times.append(dt)
    filter_times = (y_test.index >= start_dt_window) & (y_test.index <= dt)
    times_window = y_test.index[filter_times]
    labels_window = get_label_changes(y_test[filter_times].sort_index())
    preds_window = get_label_changes(pd.Series(preds[filter_times], index=times_window).sort_index())
    label_dists.append(labels_window)
    pred_dists.append(preds_window)


# In[65]:


inspection = pd.DataFrame({'End_Time': end_times, 'Labels': label_dists, 'Preds': pred_dists, 'Notes': ''}, index=start_times)
inspection.index.name = 'Start_Time'


# In[66]:


df[['Sleep.Code', 'Sleep.Num']].drop_duplicates().set_index('Sleep.Num', drop=True).sort_index()


# <a id='error_inspection'></a>
# 
# ## Inspecting errors in labchart then adding some notes
# I don't think these will match up with my notes after re-running it

# In[67]:


idx = 0
print(inspection.index[idx], inspection.iloc[idx]['End_Time'], inspection.iloc[idx]['Labels'],
      inspection.iloc[idx]['Preds'], sep='\n' + '-'*50 + '\n')


# In[68]:


inspection.iloc[idx,3] = 'heart rate varying a lot? unsure about this one'


# In[69]:


idx = 1
print(inspection.index[idx], inspection.iloc[idx]['End_Time'], inspection.iloc[idx]['Labels'],
      inspection.iloc[idx]['Preds'], sep='\n' + '-'*50 + '\n')


# In[70]:


inspection.iloc[idx,3] = 'not picking up on slow wave - saying it is active/quiet waking'


# In[71]:


idx = 2
print(inspection.index[idx], inspection.iloc[idx]['End_Time'], inspection.iloc[idx]['Labels'],
      inspection.iloc[idx]['Preds'], sep='\n' + '-'*50 + '\n')


# In[72]:


inspection.iloc[idx,3] = 'not picking up on slow wave - saying it is active/quiet waking'


# In[73]:


idx = 3
print(inspection.index[idx], inspection.iloc[idx]['End_Time'], inspection.iloc[idx]['Labels'],
      inspection.iloc[idx]['Preds'], sep='\n' + '-'*50 + '\n')


# In[74]:


inspection.iloc[idx,3] = 'confusing somewhat high delta power active/quiet waking with slow wave'


# In[75]:


idx = 4
print(inspection.index[idx], inspection.iloc[idx]['End_Time'], inspection.iloc[idx]['Labels'],
      inspection.iloc[idx]['Preds'], sep='\n' + '-'*50 + '\n')


# In[76]:


inspection.iloc[idx,3] = 'labelling a full wake -> slow wave cycle as active waking delta power is not \
                          the strongest for slow wave but still pretty clear'


# In[ ]:




