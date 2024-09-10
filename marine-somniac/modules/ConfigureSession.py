import streamlit as st
from streamlit_theme import st_theme
import modules.instructions as instruct
from utils.SessionBase import SessionBase
from config.meta import GET_STARTED, ANALYSIS_STORE


class SessionConfig(SessionBase):
    def __init__(self, sidebar_widget=True) -> None:
        self.initialize_session()
        if sidebar_widget:
            with st.sidebar:
                opts = ['']+self.get_existing_analyses()
                default = st.session_state['picked_analysis']
                if default is None:
                    default = 0
                self.analysis = st.selectbox(
                    "Pick your analysis",
                    options=opts,
                    index=default,
                    help=instruct.PICK_ANALYSIS_HELP
                )
                analysis_index = opts.index(self.analysis)
                st.session_state['picked_analysis'] = analysis_index
            self.insert_logo()

    def get_analysis_files(self) -> str:
        return SessionBase.get_analysis_files(self.analysis)

    def get_edfconfig(self) -> dict:
        return SessionBase.get_edfconfig(self.analysis)
    
    def get_labels(self):
        return SessionBase.get_labels(self.analysis)

    def get_file_from_analysis(self, file) -> str:
        return SessionBase.get_file_from_analysis(self.analysis, file)

    def get_analysis_path(self) -> str:
        return f"{ANALYSIS_STORE}/{self.analysis}"
    
    def validate_analysis(self, modes: list) -> tuple[bool, str]:
        if 'edfconfig' in modes:
            if not self.analysis:
                return (False, "Select your analysis to get started.")
            cfg_path = self.get_file_from_analysis('EDFconfig.json')
            if cfg_path is None:
                return (False, f"No specified configuration found for {self.get_edf_from_analysis()}. "
                            f'Please create one in "{GET_STARTED}"')
            
        if 'labelconfig' in modes:
            pass

        return (True, "Pass")

    @staticmethod
    def insert_logo(sidebar=True) -> None:
        try:
            theme = st_theme()['base']
            if sidebar:
                st.sidebar.image(f'assets/sidebar_logo_{theme}.jpeg')
            else:
                st.image(f'assets/logo_{theme}.jpeg')
        except:
            pass

    # TODO
    def check_model_eligibility():
        pass

    @staticmethod
    def construction_message():
        st.title("🔨Under Construction🔨")