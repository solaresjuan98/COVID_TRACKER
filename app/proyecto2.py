import datetime as dt
from os import write

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import streamlit as st
from pandas._config.config import options
from pandas.core.frame import DataFrame
from PIL import Image

# from covidcases import covidInfectionTendence
# from coviddeaths import covidDeathsByCountry,covidDeathsPredictionByDep


# METHODS
def covidDeathsByCountry(data: DataFrame):

    data_options = st.multiselect('Select fields: ', data.columns)

    try:
        df = data[data_options[0]]
        country_options = st.multiselect('Select country', df)

        country = data.loc[data[data_options[0]] == country_options[0]]

        size = country.columns.__len__()

        # Generate Graph
        x = country.columns[4:size-1]
        x = pd.to_datetime(x, format='%m/%d/%y')

        deaths = country.loc[:, country.columns[4]:country.columns[size-1]]
        data_index = deaths.index[0]

        d = {
            'columns': country.columns[4:size-1],
            'data': [deaths.loc[data_index]],
            'index': [1]
        }

        # plot graph
        try:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            df = pd.DataFrame(
                d['data'], columns=d['columns'], index=d['index'])
            df.columns.names = ['Covid deaths']
            row = df.iloc[0]
            row.plot(kind="line")
            plt.show()
            st.pyplot()

        except:
            st.warning('The graph could not be generated')

    except:
        st.warning("Please select a field")


# Tendencia de la infección por Covid-19 en un País.
def covidInfectionTendence(data: DataFrame):

    data_options = st.multiselect('Select fields', data.columns)
    # st.write(data_options)
    try:
        df = data[data_options[0]]
        country_options = st.multiselect('Select country', df)
        # st.write(country_options)
        # st.write(df.head(2))

        country = data.loc[data[data_options[0]] == country_options[0]]

        # st.write(country.columns.__len__())
        size = country.columns.__len__()

        #st.write(country.columns[4:size - 1])
        # st.write(country[country.columns[4]])

        # Generate graph
        x = country.columns[4:size-1]  # -> date
        x = pd.to_datetime(x, format='%m/%d/%y')
        # data.index = x
        positive_cases = country.loc[:, country.columns[4]                                     :country.columns[size - 1]]  # -> positive_cases

        data_index = positive_cases.index[0]
        st.write(positive_cases.loc[data_index])

        d = {'columns': country.columns[4:size - 1],
             'data': [positive_cases.loc[data_index]], 'index': [1]}

        # st.write(xd)

        # Plot graph
        try:

            df = pd.DataFrame(
                d['data'], columns=d['columns'], index=d['index'])
            df.columns.names = ['COVID INFECTIONS']
            row = df.iloc[0]
            row.plot(kind="line")
            plt.show()
            st.pyplot()

        except:
            st.warning('The graph could not be generated')

    except:
        st.warning("Please select a field")



# ===================== END METHODS =====================

# Sidebar option tuple
sid_opt_tuple = ('COVID Cases', 'COVID Deaths', 'Vaccines')

#  **** OPTION TUPLES ****
# Covid deaths
covid_deaths_tuple = (
    'Análisis del número de muertes por coronavirus en un País.', 'Predicción de mortalidad por COVID en un Departamento.')
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
            covidDeathsByCountry(data)
        elif select_report == 'Predicción de mortalidad por COVID en un Departamento.':
            covidDeathsPredictionByDep(data)

    elif sidebar_selectbox == 'Vaccines':
        pass


else:
    st.markdown("the file is empty or invalid, please upload a valid file")


# ## Validate sidebar option
