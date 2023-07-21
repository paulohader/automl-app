import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling


with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoNickML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling","Download"])
    st.info("This appplication allows you to build an automated ML pipeline for Rainfall Thresholds Triggering Landslides using Streamlit, Pandas Profiling and Pycaret. ")
    
st.write("Hello World")
