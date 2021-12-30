import datetime as dt
from matplotlib import colors

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
from sklearn import linear_model
import streamlit as st
from pandas._config.config import options
from pandas.core.algorithms import mode
from pandas.core.frame import DataFrame
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# from covidcases import covidInfectionTendence
# from coviddeaths import covidDeathsByCountry,covidDeathsPredictionByDep


def generatePredictionGraph(y: DataFrame, grade, days):

    X = []
    Y = y
    print(y)

    size = y.__len__()
    for i in range(0, size):
        X.append(i)

    #st.write('XD')
    #st.write(Y)
    X = np.asarray(X)
    Y = np.asarray(Y)

    X = X[:, np.newaxis]
    Y = Y[:, np.newaxis]

    st.subheader("Plot graph")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.scatter(X, Y)
    plt.show()
    st.pyplot()

    st.write(Y)
    # Step 2: Data preparation
    nb_degree = grade

    polynomial_features = PolynomialFeatures(degree=nb_degree)
    X_TRANSF = polynomial_features.fit_transform(X)

    # ## print(Y)
    # Step 3: define and train a model
    model = LinearRegression()
    model.fit(X_TRANSF, Y)

    # Step 4: calculate bias and variance
    Y_NEW = model.predict(X_TRANSF)

    rmse = np.sqrt(mean_squared_error(Y, Y_NEW))
    r2 = r2_score(Y, Y_NEW)

    print('RMSE: ', rmse)
    print('R2: ', r2)

    # Step 5: predicition
    x_new_min = 0.0
    x_new_max = float(days)  ## days to predict

    X_NEW = np.linspace(x_new_min, x_new_max, 50)
    X_NEW = X_NEW[:, np.newaxis]

    X_NEW_TRANSF = polynomial_features.fit_transform(X_NEW)

    Y_NEW = model.predict(X_NEW_TRANSF)

    plt.plot(X_NEW, Y_NEW, color='coral', linewidth=3)

    plt.grid()
    plt.xlim(x_new_min, x_new_max)  ## X axis
    #plt.ylim(Y_NEW[int(x_new_min)], Y_NEW[int(x_new_max)])
    plt.ylim(0, 100)
    #title = 'Degree={ }; RMSE={ }; R2={ }'.format(nb_degree, round(rmse, 2), round(r2, 2))
    plt.title('Covid prediction')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.savefig('pol_reg.jpg', bbox_inches='tight')
    plt.show()
    st.pyplot()
    print(Y_NEW)
    pass


# ===================== METHODS =====================

# Tendencia de la infección por Covid-19 en un País.


# Predicción de Infectados en un País.
def covidInfectedPredictionByCountry(data: DataFrame):

    option = st.multiselect('Select date, country and numeric variable: ',
                            data.columns)

    try:

        df = data[[option[0], option[1], option[2]]]
        #st.write(df)
        country = st.selectbox('Select country: ',
                               df[[option[1]]].drop_duplicates())

        # Filter data by country
        c = [country]

        data[data[option[1]].isin(c)]

        y = data[option[2]]

        days = st.slider('Select a number of days to predict', 5, 100)
        print(y)
        generatePredictionGraph(y, 1, days)

    except Exception as e:
        st.write(e)
        st.warning('Please select three fields')

    # infected = st.multiselect('Select numeric variable: ', data.columns)

    pass


# Indice de Progresión de la pandemia.

# Predicción de mortalidad por COVID en un Departamento.


# Ánalisis Comparativo entres 2 o más paises o continentes.
def covidComparative(data: DataFrame):

    option = st.selectbox('Select type of comparation: ',
                          ('Countries', 'Continents'))

    pass


# Tendencia del número de infectados por día de un País
def covidInfectedByDay(data: DataFrame):

    select_col = st.multiselect(
        'Select a columns to parameterize (date, country and numeric variable): ',
        data.columns)

    #st.write(select_col)

    try:
        # select_date = st.selectbox('Select a date: ',
        #                            data[select_col[0]].drop_duplicates())

        select_country = st.selectbox('Select a country: ',
                                      data[select_col[1]].drop_duplicates())

        ## Filter data
        country = [select_country]

        country_infections = data[data[select_col[1]].isin(country)]  ##

        ## sort
        country_infections[select_col[0]] = pd.to_datetime(
            country_infections[
                select_col[0]])  #country_infections.sort_values(by='Date')
        st.write(country_infections)

        # group by and sum
        #st.write(country_infections.groupby([select_col[0], select_col[1]]).sum().sort_values(by='date'))

        xd = country_infections.groupby(
            [select_col[0],
             select_col[1]]).sum().sort_values(by='date').head()

        st.write(xd.head())

        generatePredictionGraph(country_infections[select_col[2]], 2, 10)

    except Exception as e:
        st.write(e)
        st.warning('Select the fields ')

    pass


# Predicción de mortalidad por COVID en un País
def covidDeathPredictionByCountry(data: DataFrame):

    data_options = st.multiselect('Select fields', data.columns)

    try:
        df = data[data_options[0]]
        country_options = st.multiselect('Select country: ', df)
        country = data.loc[data[data_options[0]] == country_options[0]]
        size = country.columns.__len__()

        #
        start = st.slider("Select a start day: ", 0, size)
        end = st.slider("Select an end day: ", 0, size)

        if end > start:
            # x axis
            st.write("Range between ", country.columns[start], " and ",
                     country.columns[end])
            # st.write(country[start])
            x = country.columns[4:size - 1]
            x = pd.to_datetime(x, format='%m/%d/%y')
            # y axis
            deaths = country.loc[:,
                                 country.columns[4]:country.columns[size - 1]]
            data_index = deaths.index[0]

            st.write(deaths.loc[data_index][end])  # No borrar XD
            X = []
            Y = deaths.loc[data_index][start:end]
            # st.write(deaths.loc[data_index][start:end])
            for i in range(start, end):

                X.append(i)

            X = np.asarray(X)
            Y = np.asarray(Y)

            X = X[:, np.newaxis]
            Y = Y[:, np.newaxis]

            plt.scatter(X, Y)
            plt.show()
            st.pyplot()

            # Step 2:
            nb_degree = 3

            polynomial_features = PolynomialFeatures(degree=nb_degree)
            X_TRANSF = polynomial_features.fit_transform(X)

            # Step 3:
            model = LinearRegression()
            model.fit(X_TRANSF, Y)

            # Step 4:
            Y_NEW = model.predict(X_TRANSF)

            rmse = np.sqrt(mean_squared_error(Y, Y_NEW))
            r2 = r2_score(Y, Y_NEW)

            st.write('RMSE: ', rmse)
            st.write('R2', r2)

            # Step 5:
            n_days = st.slider("Days to predict ", 0, 100)
            x_new_min = start
            x_new_max = start + n_days

            X_NEW = np.linspace(x_new_min, x_new_max)
            X_NEW = X_NEW[:, np.newaxis]

            X_NEW_TRANSF = polynomial_features.fit_transform(X_NEW)
            Y_NEW = model.predict(X_NEW_TRANSF)

            plt.plot(X_NEW, Y_NEW, color='coral', linewidth=3)

            plt.grid()
            plt.xlim(x_new_min, x_new_max)
            # st.write(Y[x_new_max])
            plt.ylim(0, deaths.loc[data_index][end] + 50)

            st.write("## Covid deaths prediction for ",
                     country.columns[start + n_days])

            plt.title('Covid prediction')
            plt.xlabel('x')
            plt.ylabel('y')

            plt.show()
            st.pyplot()

            # image = plt.savefig('pol_reg.jpg', bbox_inches='tight')
            # with open(image, "rb") as file:
            #     btn = st.download_button(
            #             label="Download image",
            #             data=file,
            #             file_name="image.png",
            #             mime="image/jpg"
            #         )

        try:
            st.write("")

        except:
            st.warning('The graph could not be generated')

    except:

        st.warning('Please select a field')


# Análisis del número de muertes por coronavirus en un País.
def covidDeathsByCountry(data: DataFrame):

    data_options = st.multiselect('Select fields: ', data.columns)

    try:
        df = data[data_options[0]]
        country_options = st.multiselect('Select country', df)

        country = data.loc[data[data_options[0]] == country_options[0]]

        size = country.columns.__len__()

        # Generate Graph
        x = country.columns[4:size - 1]
        x = pd.to_datetime(x, format='%m/%d/%y')

        deaths = country.loc[:, country.columns[4]:country.columns[size - 1]]
        data_index = deaths.index[0]

        d = {
            'columns': country.columns[4:size - 1],
            'data': [deaths.loc[data_index]],
            'index': [1]
        }

        # plot graph
        try:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            df = pd.DataFrame(d['data'],
                              columns=d['columns'],
                              index=d['index'])
            df.columns.names = ['Covid deaths']
            row = df.iloc[0]
            row.plot(kind="line")
            plt.show()
            st.pyplot()

        except:
            st.warning('The graph could not be generated')

    except:
        st.warning("Please select a field")


# Tendencia de casos confirmados de Coronavirus en un departamento de un País
def covidCasesByDep(data: DataFrame):

    try:
        # date, state, cases
        data_options = st.multiselect('Select fields [date, region, cases]: ',
                                      data.columns)

        st.write(data_options)

        country_option = st.multiselect('Select country', data[data_options[1]].drop_duplicates())

        ## select states
        c = [country_option[0]]
        cs =  data[data[data_options[1]].isin(c)]

        st.write(cs)
       

        state_option = st.multiselect('Select state/province/department', cs.drop_duplicates())

        # states = data[data_options[1]]
        # dep_options = st.multiselect('Select Department/State/Province',
        #                              states)

        # state = [dep_options[0]]
        # flt = data[data.state.isin(state)]
        # flt = flt[[data_options[0], data_options[2]]]

        # st.write(flt)

        # make graphic
        # st.set_option('deprecation.showPyplotGlobalUse', False)
        # st.write("# Covid cases in ", state[0])
        # flt[data_options[0]] = pd.to_datetime(flt.date, format="%m/%d/%Y")
        # flt.index = flt[data_options[0]]
        # plt.plot(flt[data_options[2]], label="Covid cases history")
        # st.pyplot()

        # cormat = flt.corr()
        # f, ax = plt.subplots(figsize=(12, 9))
        # sns.heatmap(cormat, vmax=0.8, square=True)
        # st.pyplot()
        # Linear regression
        # f, ax = plt.subplots(figsize=(10, 8))
        # sns.regplot(x='date', y='positive', data=flt, ax=ax)
        # st.pyplot()

        # st.write(flt)
    except Exception as e:
        st.write(e)
        st.write('Select a field')


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
        x = country.columns[4:size - 1]  # -> date
        x = pd.to_datetime(x, format='%m/%d/%y')
        # data.index = x
        positive_cases = country.loc[:, country.columns[4]:country.columns[
            size - 1]]  # -> positive_cases

        data_index = positive_cases.index[0]
        st.write(positive_cases.loc[data_index])

        d = {
            'columns': country.columns[4:size - 1],
            'data': [positive_cases.loc[data_index]],
            'index': [1]
        }

        # st.write(xd)

        # Plot graph
        try:

            df = pd.DataFrame(d['data'],
                              columns=d['columns'],
                              index=d['index'])
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
    'Análisis del número de muertes por coronavirus en un País.',
    'Predicción de mortalidad por COVID en un Departamento.',
    'Predicción de mortalidad por COVID en un País')
# Covid Cases
covid_cases_tuple = (
    'Tendencia de la infección por Covid-19 en un País.',
    'Predicción de Infectados en un País.',
    'Ánalisis Comparativo entres 2 o más paises o continentes.',
    'Tendencia del número de infectados por día de un País',
    'Tendencia de casos confirmados de Coronavirus en un departamento de un País'
)

# Covid Vaccines
covid_vaccines_tuple = ()

# Main
st.sidebar.write("""
    # PROYECTO 2
    *Juan Antonio Solares Samayoa* - 201800496
""")

# add sidebar
# insert image
image = st.sidebar.image(
    'https://peterborough.ac.uk/wp-content/uploads/2020/03/NHS-covid-banner-1.png'
)
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

        elif select_report == 'Tendencia de casos confirmados de Coronavirus en un departamento de un País':
            covidCasesByDep(data)

        elif select_report == 'Predicción de Infectados en un País.':
            covidInfectedPredictionByCountry(data)

        elif select_report == 'Tendencia del número de infectados por día de un País':
            #select_date = st.selectbox('Select data to parameterize', data.)
            covidInfectedByDay(data)
        elif select_report == 'Ánalisis Comparativo entres 2 o más paises o continentes.':

            covidComparative(data)
            pass

    elif sidebar_selectbox == 'COVID Deaths':

        select_report = st.selectbox('Select report', covid_deaths_tuple)

        # st.write(data)
        if select_report == 'Análisis del número de muertes por coronavirus en un País.':
            covidDeathsByCountry(data)
        elif select_report == 'Predicción de mortalidad por COVID en un Departamento.':
            # covidDeathsPredictionByDep(data)
            pass
        elif select_report == 'Predicción de mortalidad por COVID en un País':
            covidDeathPredictionByCountry(data)

    elif sidebar_selectbox == 'Vaccines':
        pass

else:

    st.warning("the file is empty or invalid, please upload a valid file")

# ## Validate sidebar option
