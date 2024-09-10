import streamlit as st
from modules.ConfigureFeatures import ConfigureFeatures
from modules.BuildFeatures import BuildFeatures
from modules.ConfigureSession import SessionConfig
import modules.instructions as instruct
from config.meta import APP_NAME

st.set_page_config(
    page_title=f"{APP_NAME}: Feature Builder",
    initial_sidebar_state='expanded',
    layout='wide'
)
session = SessionConfig()


@st.experimental_dialog(f'Feature Configuration for "{session.analysis}"', width='large')
def show_config(config):
    st.write(config)


st.title('Compute Features')
instruct.feature_generation()

validity = session.validate_analysis(modes=['edfconfig'])
if not validity[0]:
    st.error(validity[1])
else:
    page = ConfigureFeatures(session.analysis)
    # starting_point = st.radio(
    #     "Choose a starting point",
    #     options=["Saved configuration", "From scratch", "All possible", 'Base Model', 'Extended Model', 'Refined Model'],
    #     horizontal=True,
    #     disabled=True,
    #     help="🔨Under Construction🔨"
    # )
    view_config = st.container()
    conf, build = st.tabs(['Configure', 'Build & Explore Features'])

    with conf:
        page.specify_methods_per_channel()

        valid = page.validate_all_configurations()
        if not valid[0]:
            st.error(valid[1])
        else:
            st.success(valid[1])

        if st.button("Save Configuration", disabled=not valid[0], use_container_width=True):
            page.save_configuration()

    with build:
        bld = BuildFeatures(
            analysis=page.analysis,
            build_config=page.retrieve_configuration()
        )
        with view_config:
            if st.button("View Saved Configuration", use_container_width=True):
                if bld.build_config:
                    cfg = bld.flatten_configuration()
                else:
                    cfg= {}
                show_config(cfg)

        if st.button("Start Feature Calculations", use_container_width=True):
            bld.compile_commands()
            bld.execute_all_commands()

        # TODO check if feature store exists
        bld.visualize_features()