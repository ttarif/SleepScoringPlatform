import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import pandas as pd
import os
import json
from utils.StringUtils import StringUtils
from utils.GeneralUtils import GeneralUtils
from config.meta import ANALYSIS_STORE


class SessionBase(StringUtils, GeneralUtils):
    @staticmethod
    def initialize_session() -> None:
        """
        Initializes session_state variables. Creates analysis store directory.
        """
        SESSION_VARS = (
            'picked_analysis',
        )
        for session_var in SESSION_VARS:
            if session_var not in st.session_state:
                st.session_state[session_var] = None

        if ANALYSIS_STORE not in os.listdir(os.getcwd()):
            os.mkdir(ANALYSIS_STORE)

    @staticmethod
    def validate_analysis_name(analysis: str) -> tuple[bool, str]:
        """
        Confirm that analysis string is a valid directory name and
        that it does not already exist.
        """
        if analysis.strip() == '':
            return (False, "Enter an analysis name to get started.")
        elif analysis in SessionBase.get_existing_analyses():
            return (False, "Analysis name already in use.")
        else:
            return (True, "Pass")

    @staticmethod
    def get_analysis_files(analysis: str) -> str:
        return os.listdir(f"{ANALYSIS_STORE}/{analysis}")

    @staticmethod
    def get_analysis_path(analysis: str) -> str:
        return f"{ANALYSIS_STORE}/{analysis}"

    @staticmethod
    def get_existing_analyses() -> list:
        return os.listdir('filestore')
    
    @staticmethod
    def get_file_from_analysis(analysis, file) -> str | None:
        parent_path = SessionBase.get_analysis_path(analysis)
        if file in os.listdir(parent_path):
            return f"{parent_path}/{file}"
        return None
    
    @staticmethod
    def get_labels(analysis) -> pd.DataFrame:
        path = SessionBase.get_file_from_analysis(analysis, 'labels.csv')
        return pd.read_csv(path)
    
    @staticmethod
    def get_edfconfig(analysis) -> dict:
        path = SessionBase.get_file_from_analysis(analysis, 'EDFconfig.json')
        return SessionBase.read_json(path)
    
    @staticmethod
    def get_edf_from_analysis(analysis: str, path=False) -> str | None:
        """
        Search for and retrieve filepath of the EDF file for a given analysis.
        """
        if analysis in SessionBase.get_existing_analyses():
            for file in os.listdir(f'{ANALYSIS_STORE}/{analysis}'):
                ext = file.split('.')[-1].lower()
                if ext == 'edf' and not path:
                    return file
                elif ext == 'edf' and path:
                    return f"{ANALYSIS_STORE}/{analysis}/{file}"
        return None

    @staticmethod
    def get_label_from_analysis(analysis: str, path=False) -> str | None:
        """
        Search for and retrieve filepath of the label file for a given analysis.
        """
        if analysis in SessionBase.get_existing_analyses():
            for file in os.listdir(f'{ANALYSIS_STORE}/{analysis}'):
                if file != 'labels.csv':
                    ext = file.split('.')[-1].lower()
                    if ext == 'csv' and not path:
                        return file
                    elif ext == 'csv' and path:
                        return f"{ANALYSIS_STORE}/{analysis}/{file}"
        return None

    @staticmethod
    def write_edf(file: UploadedFile, parent_dir) -> None:
        """
        Take the EDF file in the form streamlit's UploadedFile object (return type of
        st.file_uploader), read it into bytes, then write to disk under the configurable
        `ANALYSIS_STORE`/`ANALYSIS` path.
        """
        session_dir = f'{ANALYSIS_STORE}/{parent_dir}'
        if parent_dir not in os.listdir(ANALYSIS_STORE):
            os.mkdir(session_dir)

        existing_file = SessionBase.get_edf_from_analysis(parent_dir)
        if existing_file is not None:
            os.remove(f"{ANALYSIS_STORE}/{parent_dir}/{existing_file}")

        file_bytes = file.read()
        file_write_path = f'{session_dir}/{file.name}'
        with open(file_write_path, 'wb') as f:
            f.write(file_bytes)

    @staticmethod
    def write_configuration(config: dict, analysis, name) -> None:
        """
        Write a dictionary to a json file under the specfied analysis directory.
        """
        path = f"{SessionBase.get_analysis_path(analysis)}/{name}"
        with open(path, "w") as f:
            json.dump(config, f, default=str, indent=4)