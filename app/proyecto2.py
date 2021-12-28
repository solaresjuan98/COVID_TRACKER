
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from covidcases import *

# Sidebar option tuple
sid_opt_tuple = ('COVID Cases', 'COVID Deaths', 'Vaccines')

#  **** OPTION TUPLES ****
# Covid deaths
covid_deaths_tuple = (
    'Análisis del número de muertes por coronavirus en un País.', 'xd')
# Covid Cases
covid_cases_tuple = ('Tendencia de la infección por Covid-19 en un País.',
                     'Predicción de Infertados en un País.', 'Ánalisis Comparativo entres 2 o más paises o continentes.')

# Covid Vaccines
covid_vaccines_tuple = ()


# Main
st.sidebar.write("""
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

    # Validate area of analysis
    if sidebar_selectbox == 'COVID Cases':
        select_report = st.selectbox('Select report', covid_cases_tuple)

        st.write(data)
        # Validate option
        if select_report == 'Tendencia de la infección por Covid-19 en un País.':
            covidInfectionTendence(data)

    elif sidebar_selectbox == 'COVID Deaths':

        select_report = st.selectbox('Select report', covid_deaths_tuple)

        st.write(data)
        if select_report == 'Análisis del número de muertes por coronavirus en un País.':
            pass

    elif sidebar_selectbox == 'Vaccines':
        pass


else:
    st.markdown("the file is empty or invalid, please upload a valid file")


# ## Validate sidebar option
