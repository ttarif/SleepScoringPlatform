import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, datetime
import modules.instructions as instruct
from utils.SessionBase import SessionBase
from utils.EDF.EDF import EDFutils
from streamlit.runtime.uploaded_file_manager import UploadedFile
from config.channelcompute import CHANNEL_TYPES


@st.cache_data(show_spinner=False)
def load_edf_details(path):
    edf = EDFutils(path)
    details = {}
    details['start_ts'] = edf.start_ts
    details['end_ts'] = edf.end_ts
    details['freqs'] = edf.channel_freqs
    details['channels'] = edf.channels
    return details


class ConfigureEDF(SessionBase):
    def __init__(self, analysis) -> None:
        self.analysis = analysis
        self.edfpath = self.get_edf_from_analysis(analysis, path=True)
        self.channel_map = None
        self.time_range = None
        self.date_str_format = '%Y-%m-%d %H:%M:%S.%f'
        self.config_name = "EDFconfig.json"

    def upload_file(self: UploadedFile) -> None:
        file = st.file_uploader('Drop your EDF file here')
        file_validity = self.validate_file(file)
        if not file_validity[0]:
            st.error(file_validity[1])
        else:
            st.success(file_validity[1])
        if st.button('Save EDF to analysis', disabled=not file_validity[0]):
            with st.spinner('Writing file to disk, this may take a minute...'):
                self.write_edf(file, self.analysis)

        existing_edf = self.get_edf_from_analysis(self.analysis)
        if existing_edf:
            st.warning("An EDF file already exists in this analysis. "
                       "Clicking the save button will overwrite it")
        self.edfpath = self.get_edf_from_analysis(self.analysis, path=True)

    def initialize_edf_properties(self) -> None:
        with st.spinner(f'Reading metadata from EDF, please wait...'):
            self.edf = load_edf_details(self.edfpath)
        
    def channel_mapping(self) -> None:
        """
        Generate widgets for collecting EDF configuration. Prompts user to
        specify which channels will be analyzed, and to specify the type of
        each channel.
        """
        # TODO: read in existing config

        defaults = self.retrieve_defaults('channel_mapping')
        if defaults is None:
            ch_default = None
            ch_type_map = {}
        else:
            ch_default = defaults['channels']
            ch_type_map = defaults['type_map']

        picked_channels = st.multiselect(
            'What channels will you be using?',
            options=self.edf['channels'],
            default=ch_default
        )
        channel_map = pd.DataFrame(
            picked_channels,
            columns=["ch_name"]
        )
        channel_map["ch_type"] = [ch_type_map.get(ch) for ch in picked_channels]
        channel_map["ch_freq"] = [self.edf['freqs'][ch] for ch in picked_channels]

        self.channel_map = st.data_editor(
            channel_map,
            column_config={
                "ch_name": st.column_config.TextColumn(
                    "Channel Name",
                    disabled=True
                ),
                "ch_type": st.column_config.SelectboxColumn(
                    "Channel Type",
                    help=instruct.CHANNEL_TYPE_HELP,
                    options=[None]+CHANNEL_TYPES,
                    required=True
                ),
                "ch_freq": st.column_config.NumberColumn(
                    "Channel Frequency",
                    disabled=True
                )
            },
            use_container_width=True,
            hide_index=True
        )
        if not all(self.channel_map.ch_type):
            st.error("Ensure all channel types are specified")

    def set_time_range(self) -> None:
        edf_start = self.edf['start_ts']
        edf_end = self.edf['end_ts']

        defaults = self.retrieve_defaults('set_time_range')
        if defaults is None:
            start = edf_start
            end = edf_end
        else:
            start = defaults['start']
            end = defaults['end']

        with st.expander("Set time range", True):
            c = st.columns([6, 1, 6])
            c[0].markdown(f"EDF Start Timestamp: `{edf_start}`")
            c[2].write(f"EDF End Timestamp: `{edf_end}`")

            c = st.columns([2, 2, 2, 1, 2, 2, 2])
            start_date = c[0].date_input("Start Date",
                value=start.date())
            start_time = c[1].time_input("Start Time",
                value=start.time(), step=60)
            start_secs = c[2].number_input("Start Seconds",
                value=start.second + start.microsecond/10**6,
                min_value=0.0,
                max_value=60.0,
                step=1e-6,
                format="%.6f")
            
            end_date = c[4].date_input("End Date",
                value=end.date())
            end_time = c[5].time_input("End Time",
                value=end.time(), step=60)
            end_secs = c[6].number_input("End Seconds",
                value=end.second + end.microsecond/10**6,
                min_value=0.0,
                max_value=60.0,
                step=1e-6,
                format="%.6f")
            
        user_start = datetime.combine(start_date, start_time)
        user_start += timedelta(seconds=start_secs) 
        user_end = datetime.combine(end_date, end_time)
        user_end += timedelta(seconds=end_secs)
        self.time_range = (user_start, user_end)
        return None

    def retrieve_defaults(self, for_: str) -> None|dict:
        """
        Check if config file exists in the analysis directory,
        reads the json and returns it.
        """
        config_json = self.get_file_from_analysis(self.analysis, 'EDFconfig.json')
        if config_json is not None:
            config = self.read_json(config_json)
            defaults = {}
            match for_:
                case 'channel_mapping':
                    defaults['channels'] = config['channels']['picked']
                    type_map = {}
                    for ch_type, chs in config['channels']['map'].items():
                        for ch in chs:
                            type_map[ch] = ch_type
                    defaults['type_map'] = type_map
                
                case 'set_time_range':
                    defaults['start'] = datetime.strptime(
                        config['time']['start'], self.date_str_format)
                    defaults['end'] = datetime.strptime(
                        config['time']['end'], self.date_str_format)

            return defaults

        else:
            return None

    def construct_configuration(self) -> dict:
        """
        Creates a config dict based on the configurations provided via the user's
        widget selections. Stores EDF channel selection, channel type/freq mappings,
        and the time range subset of the analysis.
        """
        config = {
            'time': {
                'start': datetime.strftime(self.time_range[0], self.date_str_format),
                'end': datetime.strftime(self.time_range[1], self.date_str_format), 
                'tz': None
            },
            'raw_time': {
                'start': datetime.strftime(self.edf['start_ts'], self.date_str_format),
                'end': datetime.strftime(self.edf['end_ts'], self.date_str_format)
            },
            'channels': {
                'picked': [],
                'map': {ch: [] for ch in CHANNEL_TYPES+['ignore']},
                'freq': {},
            },
            'channels_': {}
        }
        channel_map = config['channels']['map']
        channel_freqs = config['channels']['freq']
        for _, row in self.channel_map.iterrows():
            ch_type = row['ch_type']
            ch_name = row['ch_name']
            ch_freq = row['ch_freq']

            config['channels']['picked'].append(ch_name)
            channel_map[ch_type].append(ch_name)
            if ch_freq not in channel_freqs:
                channel_freqs[ch_freq] = []
            channel_freqs[ch_freq].append(ch_name)

            config['channels_'][ch_name] = (ch_type, ch_freq)

        for ch in self.edf['channels']:
            if ch not in config['channels']['picked']:
                config['channels']['map']['ignore'].append(ch)
        
        return config

    def validate_configuration(self) -> tuple:
        """
        Logical checks on user-specified configurations used to constraint
        the ability to write the config file.
        """
        if self.channel_map is None:
            return (False, "`self.channel_map` not found")
        if not all(self.channel_map.ch_type):
            return (False, "Ensure all channel types are specified"
                    'in the "Map Channels" menu')
        if self.time_range[0] > self.time_range[1]:
            return (False, f"Specified start time `{self.time_range[0]}` "
                    f"occurs after end time `{self.time_range[1]}`")
        if self.time_range[0] < self.edf['start_ts']:
            return (False, f"Specified start time `{self.time_range[0]}` " 
                    f"occurs before EDF start time `{self.edf['start_ts']}`")
        if self.time_range[1] > self.edf['end_ts']:
            return (False, f"Specified end time `{self.time_range[1]}` "
                    f"occurs after EDF end time `{self.edf['end_ts']}`")
        else:
            return (True, "Configuration valid, please confirm & save (will overwrite previous)")
        
    def save_configuration(self):
        """
        Writes config file to `ANALYSIS_STORE`/`ANALYSIS`/EDFconfiog.json.
        """
        self.write_configuration(
            config=self.construct_configuration(),
            analysis=self.analysis,
            name=self.config_name
        )
        st.toast(f"Configuration saved.")

    @staticmethod
    def validate_file(file: UploadedFile) -> tuple:
        """
        Checks for user inputs to the st.file_uploader for EDF submission.
        """
        if file is None:
            return (False, "No file detected (may take a moment even when loading bar is full)")
        ext = file.name.split('.')[-1].lower()
        if not ext == 'edf':
            return (False, f'Only accepts .edf file extension, not "{ext}"')
        else:
            return (True, "Valid file")