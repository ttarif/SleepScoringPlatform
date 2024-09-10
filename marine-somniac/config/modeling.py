BASE_FEATURE_SET = {
    "Pressure": {
        "get_rolling_mean": [{"window_sec": 30, "step_size": 1}],
        "get_rolling_std": [{"window_sec": 30, "step_size": 1}]
    },
    "ODBA": {
        "get_rolling_mean": [{"window_sec": 30, "step_size": 1}],
        "get_rolling_std": [{"window_sec": 30, "step_size": 1}]
    },
    "ODBA": {
        "get_rolling_mean": [{"window_sec": 30, "step_size": 1}],
        "get_rolling_std": [{"window_sec": 30, "step_size": 1}]
    },
    "ECG": {
        "get_heart_rate": [
            {"search_radius": 200, "filter_threshold": 200}
        ]
    },
    "EEG": {
        "get_yasa_welch": [
            {"preset_band_range": "alpha"},
            {"preset_band_range": "beta"},
            {"preset_band_range": "sigma"},
            {"preset_band_range": "theta"},
            {"preset_band_range": "sdelta"},
            {"preset_band_range": "fdelta"},
        ],
        "get_hjorth_mobility": [{}],
        "get_hjorth_complexity": [{}],

    }
}
EXTENDED_FEATURE_SET = {
    
}
REFINED_FEATURE_SET = {

}