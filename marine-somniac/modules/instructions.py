import streamlit as st
from config.meta import APP_NAME, GET_STARTED, __PAPER_LINK

PICK_ANALYSIS_HELP = 'You can create an analysis in the "Start New Analysis" page.'
CHANNEL_TYPE_HELP = 'The channel type tells us what features should be built off a given channel.'
FEATURE_FREQUENCY_HELP = "All features genereated herein must be of the same output frequency. " \
                         "This widget is where that output frequency is specified."
N_COMPS_HELP = "You may want to calculate multiple instances of a feature to experiment with different " \
               "parameters like window sizes, etc."

ANALYSIS_NAME = 'This will be a directory name, special characters may be rejected'

def feature_generation():
    st.markdown("""
        By default, all recommended base feature computations are specified. Edit the chosen features or 
        their specific settings below. You can also add multiple instances of any features so long as they 
        have unique settings.
    """)

def model_constraints():
    st.markdown('''
        To use any of the existing models, you will need:
        * 1x Gyroscopic Channel (Z)
        * 1x Pressure Channel
        * 1x ODBA Channel
        * 1x ECG Channel
        * 1x EEG Channel
    ''')

def get_started():
    st.subheader('Getting Started')
    st.markdown(
        "Please note, this tool only functions on data of the .edf format. "
        "We have found that most feature computations are more effective on electrophysiological data "
        f"that has already undergone [some degree of processing (ICA)]({__PAPER_LINK}). "
    )
    st.markdown("**Mapping your files**")
    st.markdown(f"""
        This application needs to know a few details about your data before we can get started
        with your analysis. In the ***{GET_STARTED}*** page, you can specify things like which channels you'll
        be exploring and letting the application know what they are. Configurations need to be 
        specified for both your EDF data as well as any label data (if you will be training your
        own models).
    """)

    with st.expander("Configuration Summary & File Constraints"):
        st.markdown("""
            **Signal Data (must be EDF)**  
            * Config will specify time bracket of your analysis (needed to line up with your labels)
            * Config will specify what your custom channel names represent (are they ECG, motion data?)
                    
            **Label Data (must be CSV)**
            * All rows should be equally spaced time intervals (ex: each row labels 1 second)
            * Config will specify which column refers to your time interval and your label(s)
        """)

    st.markdown("""        
        You only have to do this once per analysis. If you ever need to edit your configuration, you can 
        simply return to that page, make your edits, and save them.
    """)

def compute_features():
    st.subheader("Computing Features")
    st.markdown("""
        To calculate inputs to the pre-trained models or train your own, you'll first need to compute features 
        that capture patterns in the data that allows for the differentiation of sleep states. You'll have the 
        option to simply choose suggested feature sets instead of specifying your own from scratch. Please note 
        that computation can take an especially long time if you're scoring a large time range of data. It is 
        recommended to limit your scoring to 5 hours and do it in batches, but the app will still function on 
        larger datasets, but it will be taxing on the server.
                
        ***Once you've specified your features...***
                
        Simply navigate to the "Build & Explore Features" tab and click the "Build Features" button. Each feature 
        will be stored as its own csv so that if issues occur during computation, progress can be resumed from the 
        last uncomputed feature.
                
        You can now visually assess your features and compare them to your labels!
    """)