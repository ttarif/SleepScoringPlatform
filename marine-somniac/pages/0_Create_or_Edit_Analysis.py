"""
This module will be the entrypoint to any user journey. It will prompt the user
to supply the necessary files (EDF and label CSV), then write these to disk.
The user will then need to specify configurations/mappings for both of these files which will
be used for the other modules in the app.
"""
import streamlit as st
import modules.instructions as instruct
from modules.ConfigureSession import SessionConfig
from modules.ConfigureEDF import ConfigureEDF
from modules.ConfigureLabel import ConfigureLabel
from config.meta import APP_NAME

st.set_page_config(
    page_title=f"{APP_NAME}: Configure",
    initial_sidebar_state='expanded',
    layout='wide'
)
session = SessionConfig(sidebar_widget=False)
session.insert_logo()

with st.expander("Read this if you plan to use a pre-trained model"):
    instruct.model_constraints()

mode = st.radio(
    'Create or Edit',
    options=['Start New Analysis', 'Edit Existing'],
    horizontal=True,
    label_visibility='collapsed'
)
if mode == 'Start New Analysis':
    analysis_name = st.text_input(
        'Name your analysis',
        max_chars=60,
        help=instruct.ANALYSIS_NAME
    )
    session_validity = session.validate_analysis_name(analysis_name)
    st.subheader("Upload a file to save your analysis")
    st.write('After uploading your file, switch to the "Edit Existing" workflow')

elif mode == 'Edit Existing':
    existing = session.get_existing_analyses()
    if not any(existing):
        st.error('No existing analyses found')
        analysis_name = None
        analysis_description = None
    else:
        analysis_name = st.selectbox(
            'Pick analysis',
            options=['']+existing
        )
    session_validity = (analysis_name != '', "Select a analysis")
    st.subheader("Add and configure your EDF and labels")

edf_pane, label_pane = st.tabs(['EDF', 'Labels'])

if not session_validity[0]:
    st.error(session_validity[1])

else:
    with edf_pane:
        edfWidgets = ConfigureEDF(analysis_name)
        with st.expander("Upload/Overwrite File", True):
            edfWidgets.upload_file()

        if not ConfigureEDF.get_edf_from_analysis(analysis_name):
            st.error("No EDF file associated with this analysis. Please upload one.")

        elif mode == 'Edit Existing':
            edfWidgets.initialize_edf_properties()
            with st.expander("Map Channels", True):
                edfWidgets.channel_mapping()
            edfWidgets.set_time_range()

            edf_valid = edfWidgets.validate_configuration()
            if not edf_valid[0]:
                st.error(edf_valid[1])
            else:
                st.success(edf_valid[1])
            
            if st.button("Save Configuration", disabled=not edf_valid[0], use_container_width=True):
                edfWidgets.save_configuration()

    with label_pane:
        lblWidgets = ConfigureLabel(analysis_name)
        with st.expander("Upload/Overwrite File", True):
            lblWidgets.upload_file()
    
        if not ConfigureLabel.get_label_from_analysis(analysis_name):
            st.error("No label file associated with this analysis. Please upload one.")

        elif mode == 'Edit Existing':
            with st.expander("Map Columns", True):
                lblWidgets.column_mapping()

            label_valid = lblWidgets.validate_configuration()
            if not label_valid[0]:
                st.error(label_valid[1])
            else:
                st.success(label_valid[1])
            
            if st.button("Save Configuration", disabled=not label_valid[0], key='L', use_container_width=True):
                lblWidgets.save_configuration()
        
    # if edfWidgets.get_edf_from_analysis(analysis_name) and lblWidgets.get_label_from_analysis(analysis_name):
    #     st.divider()
    #     st.subheader("**Label & EDF Compatibility Checks**")
    #     with st.spinner("Checking specified timestamps are compatible...")
    #         session.analysis = analysis_name
    #         edfconfig = session.get_edfconfig()
    #         labels = session.get_labels()


    #     st.success("Success!")

