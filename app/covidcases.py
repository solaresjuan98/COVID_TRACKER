
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
        positive_cases = country.loc[:, country.columns[4]
            :country.columns[size - 1]]  # -> positive_cases

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
