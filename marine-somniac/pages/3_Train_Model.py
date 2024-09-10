import streamlit as st
from modules.ConfigureSession import SessionConfig
from config.meta import APP_NAME

st.set_page_config(
    page_title=APP_NAME,
    initial_sidebar_state='expanded',
    layout='wide'
)
SessionConfig()
SessionConfig.construction_message()