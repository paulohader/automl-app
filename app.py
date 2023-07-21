import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import os


with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling","Download"])
    st.info("This appplication allows you to build an automated ML pipeline for Rainfall Thresholds Triggering Landslides using Streamlit, Pandas Profiling and Pycaret. ")
    
st.write("Welcome to the Rainfall Thresholds Probabilistic App!")


if choice == "Upload":
    st.title("Upload your data for modelling")
    file = st.file_uploader("Upload your dataset here")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target, silent=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")