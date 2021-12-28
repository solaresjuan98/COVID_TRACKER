
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


# Predicción de mortalidad por COVID en un Departamento.
def covidDeathsPredictionByDep(data: DataFrame):


    data_options = st.multiselect('Select fields: ', data.columns)

    try:

        pass

    except:
        st.warning('Please select a field')

    
