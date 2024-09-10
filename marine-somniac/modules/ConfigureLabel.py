import streamlit as st
import pandas as pd
import json
from modules.ConfigureSession import SessionConfig
from streamlit.runtime.uploaded_file_manager import UploadedFile
# lots of ConfigureEDF methods pasted and modified, 
# should create a base class for both to inherit

# TODO - 
# * validate_configuration
# * time specification options (just datetime rn)


class ConfigureLabel(SessionConfig):
    def __init__(self, analysis) -> None:
        self.analysis = analysis
        self.labelpath = self.get_label_from_analysis(analysis, path=True)
        self.col_map = {}
        self.config_name = "labelconfig.json"

    def upload_file(self) -> None:
        file = st.file_uploader('Drop your lable CSV file here')
        file_validity = self.validate_file(file)
        if not file_validity[0]:
            st.error(file_validity[1])
        else:
            st.success(file_validity[1])
        if st.button('Save labels to analysis', disabled=file is None):
            with st.spinner('Writing file to disk...'):
                self.write_labels(file)
    
        existing_label = self.get_label_from_analysis(self.analysis)
        if existing_label:
            st.warning("A label file already exists in this analysis. "
                       "Clicking the save button will overwrite it")
        self.labelpath = self.get_label_from_analysis(self.analysis, path=True)

    def column_mapping(self):
        df = pd.read_csv(self.labelpath)
        time_col = st.selectbox(
            'Which column contains timestamps?',
            options=['']+list(df.columns)
        ) 
        if time_col:
            time_valid = self.validate_date_column(df[time_col])
            if not time_valid[0]:
                st.error(time_valid[1])
            else:
                st.markdown("**Time Range**")
                times = list(df[time_col].sort_values().values)
                start = times[0]
                end = times[-1]
                st.text(f"{start} to {end}")

        label_col = st.selectbox(
            'Which column contains labels?',
            options=['']+list(df.columns)
        )
        if label_col:
            st.markdown("**Label Values**")
            st.text(pd.unique(df[label_col]))

        self.col_map['time'] = time_col
        self.col_map['label'] = label_col

    def save_configuration(self) -> None:
        csvpath = f"{self.get_analysis_path()}/labels.csv"
        rename = {
            self.col_map['time']: 'datetime',
            self.col_map['label']: 'label'
        }
        self.write_configuration(
            config=self.construct_configuration(),
            analysis=self.analysis,
            name=self.config_name
        )
        pd.read_csv(self.labelpath)\
            .rename(columns=rename)\
            [list(rename.values())]\
            .to_csv(csvpath, index=False)
        st.toast("File saving complete")

    def construct_configuration(self) -> dict:
        config = {
            'time': self.col_map['time'],
            'time_type': 'datetime64[ns]',
            'time_format': '%Y-%m-%d 14:45:22',
            'label': self.col_map['label']
        }
        return config

    @staticmethod
    def validate_date_column(column: pd.Series):
        try: 
            column.astype("datetime64[ns]")
        except:
            return (False, "Could not coerce column to datetime")
        return (True, "Pass")

    def validate_configuration(self) -> tuple:
        config = self.construct_configuration()
        if not all([bool(i) for i in config.values()]):
            return (False, "Ensure all inputs have been submitted.")
        else:
            return (True, "Valid configuration")

    def write_labels(self, file: UploadedFile) -> None:
        parent = f"{self.get_analysis_path()}"
        pd.read_csv(file).to_csv(
            f"{parent}/{file.name}"
        )

    @staticmethod
    def validate_file(file: UploadedFile) -> tuple:
        """
        Checks for user inputs to the st.file_uploader for CSV submission.
        """
        if file is None:
            return (False, "No file detected (may take a moment even when loading bar is full)")
        ext = file.name.split('.')[-1].lower()
        if not ext == 'csv':
            return (False, f'Only accepts .csv file extension, not "{ext}"')
        else:
            return (True, "Valid file")
        