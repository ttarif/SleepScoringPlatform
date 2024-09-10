## Data

This folder contains raw data to launch the analysis. The contents of this folder will not sync to GitHub. See below for a description of the content of this folder, instructions for download and data attribution, and a detailed description of data fields. 

### Folder structure for data

```
├── data
│   ├── README.md
│   ├── interim
│   │   ├── feature_discovery
│   │   │   ├── test12_Wednesday_feature_discovery_ECG.csv
│   │   │   └── test12_Wednesday_feature_discovery_EEG.csv
│   │   └── settings_accuracies.csv
│   ├── processed
│   │   └── test12_Wednesday_07_features_with_labels.csv
│   └── raw
│       ├── 01_edf_data
│       │   └── test12_Wednesday_05_ALL_PROCESSED.edf
│       └── 02_hypnogram_data
│           └── test12_Wednesday_06_Hypnogram_JKB_1Hz.csv
```

### Data attribution

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Data is made available through a CC BY 4.0 license requiring attribution. Please cite both the parent dataset and subset for data re-use.

**Citation for data subset:**

Kendall-Bar, Jessica (2024). test12_Wednesday_05_ALL_PROCESSED.edf. *figshare.* Dataset. https://doi.org/10.6084/m9.figshare.25734165.v1

**Parent dataset:**

Kendall-Bar, Jessica et al. (2023). Data for: Brain activity of diving seals reveals short sleep cycles at depth [Dataset]. *Dryad.* https://doi.org/10.7291/D1ZT2B

**Data originally published with article:** 

Kendall-Bar, JM; Williams, TM; Mukherji, R; Lozano, DA; Pitman, JK; Holser, RR; Keates, T; Beltran, RS; Robinson, PW; Crocker, DE; Adachi, T; Lyamin, OI; Vyssotski, AL; Costa, DP (2023). Brain activity of diving seals reveals short sleep cycles at depth. *Science.* https://doi.org/10.1126/science.adf0566 

### Accessing Data Subset

1. Data can be downloaded from the following figshare link: https://figshare.com/articles/dataset/test12_Wednesday_05_ALL_PROCESSED_edf/25734165

2. Data is then placed into `raw` data folder with `.edf` files in `01_edf_data` and labeled hypnogram data in `02_hypnogram_data`.

### Description of data fields

- #### *Processed EDF files:*
  Processed EEG, EMG, EOG, and motion sensor data. Motion sensor processing includes transformation of raw Accelerometer, Gyroscope, and Magnetometer data into pitch, roll, heading, and ODBA. Electrophysiological processing includes running Independent Components Analysis (ICA) to identify and remove signals contaminating brain or heart activity channels. Channels with brain only, as cleaned by ICA, are labeled `EEG_ICAX`, and channels with heart activity only, as cleaned by ICA, are labeled `ECG_ICAX`. These are sometimes but not always cleaner than the Raw EXG channels. Example filename `test12_Wednesday_05_ALL_PROCESSED.edf`.


- #### *02 Hypnogram data:* 
  This folder contains labeled data at 1s resolution 
  Processed Sleep Scoring (lab, wild, & at sea): Example filename  `test12_Wednesday_06_Hypnogram_JKB_1Hz.csv`
    <details>
    <summary> Column descriptions. </summary>
  
    - **timebins** - Time in R format for the beginning of the 30s epoch
    - **SealID** - unique identifier for each seal
    - **Recording.ID** - identifier combining the location (in the lab [CAPTIVE], in the wild [WILD], or translocated [XLOC]), age (in years [yr] or months [mo]), and age class (juvenile or weanling) of the seal
    - **ID** - in the lab [CAPTIVE], in the wild [WILD], or translocated [XLOC]
    - **Sleep.Code** - Specific sleep state designation: 
      - ***Active Waking***
      - ***Quiet Waking*** 
      - ***Drowsiness*** - Intermittent slow waves
      - ***LV Slow Wave SLeep*** - Low-voltage slow wave sleep
      - ***HV Slow Wave Sleep*** - High-voltage slow wave sleep
      - ***Certain REM Sleep*** - Rapid-Eye-Movement (REM) Sleep scored with high confidence (high degree of Heart Rate Variability [HRV])
      - ***Putative REM Sleep*** - REM Sleep scored with low confidence (low HRV)
      - ***Unscorable*** - Data not scorable due to interference, motion artifacts, or signal quality
    - **Simple.Sleep.Code** - Simplified sleep state designation: 
      - ***Active Waking***
      - ***Quiet Waking*** 
      - ***Drowsiness*** - Intermittent slow waves
      - ***SWS*** - Slow wave sleep (LV & HV combined)
      - ***REM*** - REM Sleep (certain and putative combined)
      - ***Unscorable*** - Data not scorable due to interference, motion artifacts, or signal quality
    - **Resp.Code** - Respiratory state designation:
      - ***Eupnea*** - between first breath and last breath
      - ***transition to Eupnea*** - transition to tachycardia
      - ***Apnea*** - between last breath and first breath
      - ***transition to Apnea*** - transition to bradycardia
      - ***Unscorable*** - not scorable due to noise obscuring HR detection
    - **Water.Code** - Location of animal
      - ***LAND*** - on land (in pen in the lab or on beach in the wild)
      - ***SHALLOW WATER*** - in water < 2m deep (in pool in the lab or in the lagoon at Ano Nuevo)
      - ***DEEP WATER*** - animal traversing the continental shelf (< 200 m / in water shallow enough that the animal can rest / travel along bottom)
      - ***OPEN OCEAN*** - animal in water deeper than 200 m / in water deep enough that the animal cannot rest / travel along bottom
    - **Time_s_per_day** - Time of day in seconds (out of 86400)
    - **Day** - Day of the recording
      </details>
