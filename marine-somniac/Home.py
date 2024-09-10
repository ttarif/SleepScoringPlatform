# TODO - global
# Create and Edit
    # - Add visuals for time alignment + validation
    # - Refactor code in ConfigureLabel and ConfigureEDF (highly similar, redundancy)
# ConfigureLabel - exception handling and better validation
# ConfigureFeatures 
    # - preload with saved config
    # - preset options
    # - exception handling when empty config
# BuildFeatures 
    # - don't recompute features if config is the same
    # - join all features together for a final training set
    # - plot code is a mess

# Train Model - IMPLEMENT
# Evaluate Model - IMPLEMENT
# Download data - support for non-json files
import streamlit as st
from modules.ConfigureSession import SessionConfig
import modules.instructions as instruct
from config.meta import APP_NAME

st.set_page_config(
    page_title=APP_NAME,
    initial_sidebar_state='expanded',
    layout='centered',
    page_icon='assets/logo_dark.jpeg'
)
SessionConfig.initialize_session()

title, image = st.columns([4,2])
with title:
    st.title(APP_NAME)
    st.subheader('Sleep scoring for our aquatic pals')
with image:
    SessionConfig.insert_logo(sidebar=False)


st.markdown(
    f"Welcome to {APP_NAME}! This is a tool for partially automating sleep stage scoring. "
    "While this app was built with Northern elephant seals in mind, many utilities are "
    "generalizeable to other organisms, namely the computation of aggregate/windowed features "
    "from electrophysiological data (EEG, ECG) such as frequency power or heart rate."
)
st.markdown("""
    This tool will save your uploaded data to a remote server on which all computations will be 
    performed.
""")
instruct.get_started()
instruct.compute_features()

