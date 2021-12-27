

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
from PIL import Image

import streamlit as st


# Sidebar option tuple
sid_opt_tuple = ('COVID Cases', 'Covid Deaths', 'Vaccines')

# Option tuple
opt_tuple = ('Tendencia de la infección por Covid-19 en un País.',
             'Predicción de Infertados en un País.')

# Main
st.write("""
    # PROYECTO 2
    *Juan Antonio Solares Samayoa* - 201800496
""")

# add sidebar
app_sidebar = st.sidebar.title("MENU")

st.sidebar.header("Load a file: ")

# file uploader
upload_file = st.sidebar.file_uploader("Choose a .csv, .xls or .json file")

# add selectbox to the sidebar
sidebar_selectbox = st.sidebar.selectbox('Select Type...', sid_opt_tuple)


# read csv file

if upload_file is not None:
    data = pd.read_csv(upload_file)

    # st.write(data.columns)
    st.success('your file has been successfully uploaded')

    # Reports
    reports = st.selectbox('Select report', opt_tuple)

    # Multiselect
    options = st.multiselect(
        'Select fields',
        data.columns,
        [])
    
    st.write('You selected:', options)

    ##st.subheader("Deaths by state")

    ##
    date = options[0]
    deaths = options[1]

    st.set_option('deprecation.showPyplotGlobalUse', False)
    data[date] = pd.to_datetime(data.date, format="%m/%d/%Y")
    data.index = data[date]
    plt.plot(data[deaths], label="Covid deaths history")
    st.pyplot()
    
else:
    st.markdown("the file is empty or invalid, please upload a valid file")


# ## Validate sidebar option
