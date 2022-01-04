import base64
import datetime as dt
import io
import time
from math import e

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import streamlit as st
from attr import field
from fpdf import FPDF
from matplotlib import colors
from pandas._config.config import options
from pandas.core import groupby
from pandas.core.algorithms import mode
from pandas.core.frame import DataFrame
from pandas.core.reshape.pivot import pivot_table
from PIL import Image
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from streamlit.elements.arrow import Data


def generatePredictionGraph(y: DataFrame, grade, days, max_val):

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

    # GRAPH 1
    #st.subheader("Plot graph")
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    #plt.scatter(X, Y)
    #plt.show()
    #st.pyplot()

    # st.write(Y)
    #  2:
    nb_degree = grade

    polynomial_features = PolynomialFeatures(degree=nb_degree)
    X_TRANSF = polynomial_features.fit_transform(X)

    # ## print(Y)
    #  3:
    model = LinearRegression()
    model.fit(X_TRANSF, Y)

    #  4
    Y_NEW = model.predict(X_TRANSF)

    rmse = np.sqrt(mean_squared_error(Y, Y_NEW))
    r2 = r2_score(Y, Y_NEW)

    print('RMSE: ', rmse)
    print('R2: ', r2)

    #  5:
    x_new_min = 0.0
    x_new_max = float(days)  ## days to predict

    X_NEW = np.linspace(x_new_min, x_new_max, 50)
    X_NEW = X_NEW[:, np.newaxis]

    X_NEW_TRANSF = polynomial_features.fit_transform(X_NEW)

    Y_NEW = model.predict(X_NEW_TRANSF)

    st.subheader("Plot graph")
    plt.scatter(X, Y)
    plt.plot(X_NEW, Y_NEW, color='green', linewidth=3)
    #plt.scatter(X_NEW, Y_NEW, color='blue', linewidth=3)
    plt.grid()
    plt.xlim(x_new_min, x_new_max)  ## X axis

    plt.ylim(0, Y_NEW[int(Y_NEW.size - 1)])
    title = 'Degree={}; RMSE={}; R2={}'.format(nb_degree, round(rmse, 2),
                                               round(r2, 2))
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.savefig('D:\\prediction.png')
    #plt.savefig('pol_reg.jpg', )
    plt.show()
    st.pyplot()
    st.caption('Prediction graph')
    st.write("La predicción será de ", Y_NEW[int(Y_NEW.size - 1)][0])

    #
    #fig = ff.create_distplot(Y_NEW, ['test'], bin_size=[0] )

    pass


def generateTendencyGraph(y, header, maxY):
    x = []
    #y = country_infections[select_col[2]]

    for i in range(0, y.__len__()):
        x.append(i)

    X = np.asarray(x).reshape(-1, 1)
    reg = linear_model.LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)

    plt.scatter(X, y, color='black')
    plt.plot(X, y_pred, color='blue', linewidth=3)
    plt.savefig('D:\\trend.jpg')
    #print(reg.predict([[10]]))
    max_val = maxY

    # Header
    st.subheader(header)
    plt.title(header)
    plt.ylim(-10, max_val + 10)
    plt.show()
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.caption('COVID-19 tendence graph')

    # st.info("""Para poder comprender de mejor forma esta grafica,
    # es importante tomar en cuenta la pendiente generada, por ejemplo si la pendiente es creciente (positiva)
    # la tendencia de contagios en los dias siguientes al analisis será a la alta, pero si de lo contrario
    # la pendiente de la grafica es decreciente (negativa), la tendencia numero de contagios es a la baja.
    # """)


def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'


# write pdf report
def write_pdf(title, content, imagepath):

    pdf = FPDF()
    pdf.add_page()
    #pdf.set_xy(0, 0)
    pdf.set_font('Times', 'B', 12)
    pdf.cell(200, 10, title, 0, 2, 'C')
    pdf.set_font('Times', '', 12)
    pdf.multi_cell(200, 10, txt=content, align="J")

    #pdf.cell(90, 10, " ", 0, 2, 'C')
    #pdf.cell(-40)
    # pdf.cell(50, 10, 'X', 1, 0, 'C')
    # pdf.cell(40, 10, variable[0], 1, 0, 'C')
    # pdf.cell(40, 10, '', 1, 2, 'C')
    # pdf.cell(-90)
    # pdf.set_font('arial', '', 12)
    # for i in range(0, len(flt)):
    #     pdf.cell(50, 10, '%s' % (flt[variable[1]].iloc[i]),
    #              1, 0, 'C')
    #     pdf.cell(40, 10,
    #              '%s' % (str(flt[variable[0]].iloc[i])), 1,
    #              0, 'C')
    #     pdf.cell(40, 10,
    #              '%s' % (str(flt.positive.iloc[i])), 1, 2,
    #              'C')
    #     pdf.cell(-90)
    # pdf.cell(90, 10, " ", 0, 2, 'C')

    pdf.image(imagepath)
    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")

    st.markdown(html, unsafe_allow_html=True)


# ===================== REPORTS =====================


# 1. Tendencia de la infección por Covid-19 en un País.
def covidInfectionTendence(data: DataFrame):

    data_options = st.multiselect('Select filter [country]: ', data.columns)
    # st.write(data_options)
    try:

        if data_options.__len__() == 0:
            st.warning('Please select a column')
        else:

            country_options = st.selectbox(
                'Select country', data[data_options[0]].drop_duplicates())

            country = [country_options]

            flt = data[data[data_options[0]].isin(country)]

            have_date = st.checkbox('This file has "date" field')

            #st.write(have_date)

            if have_date:
                variable = st.multiselect(
                    'Select variables to analize [date, numeric]: ',
                    data.columns)

                if variable.__len__() < 2:
                    st.warning('Please select variables to analize')

                else:

                    st.write(variable)
                    # data

                    flt[variable[0]] = pd.to_datetime(flt[variable[0]])
                    flt[variable[0]] = flt[variable[0]]

                    flt = flt[[variable[0],
                               variable[1]]].sort_values(by=[variable[0]])

                    #sum
                    st.write(
                        flt.groupby([variable[0],
                                     variable[1]]).sum().reset_index())

                    # Tendency
                    generateTendencyGraph(flt[variable[1]],
                                          "Tendency COVID spread by country",
                                          flt[variable[1]].max())

                    st.subheader('Intepretación: ')
                    st.info(
                        """Esta grafica muestra la tendencia de indice de contagios que tiene {} en un periodo de {} días. 
                        Una pendiente positiva indica que la cantidad de casos irán aumentando con el paso del tiempo, y una pendiente 
                        negativa indica que los casos irán decreciendo con el paso del tiempo. """
                        .format(country[0], flt.__len__()))

                    ### PDF
                    pdf_title = 'Tendencia de la infección por Covid-19 en un País. \n\n'
                    content = """Esta grafica muestra la tendencia de indice de contagios que tiene {} en un periodo de {} días. 
                        Una pendiente positiva indica que la cantidad de casos irán aumentando con el paso del tiempo, y una pendiente negativa indica que los casos irán decreciendo con el paso del tiempo. """.format(
                        country[0], flt.__len__())

                    export_as_pdf = st.button("Export Report")

                    if export_as_pdf:
                        write_pdf(pdf_title, content, 'D:\\trend.jpg')

            else:
                variable = st.selectbox('Select variable to analyze: ',
                                        data.columns)

                st.write(flt[variable])

                generateTendencyGraph(flt[variable],
                                      "Tendency COVID spread by country",
                                      flt[variable].max())

            #generatePredictionGraph(flt[variable], 3, 40, flt[variable].max())

    except Exception as e:
        st.write(e)
        st.warning("Please select a field")


# 2.  Predicción de Infectados en un País.
def covidInfectedPredictionByCountry(data: DataFrame):

    has_date = st.checkbox('This file has "date" field')

    try:
        if has_date:
            option = st.multiselect(
                'Select date, country and numeric variable: ', data.columns)

            if option.__len__() < 3:
                st.warning('Please select more fields')
            else:

                df = data[[option[0], option[1], option[2]]]
                #st.write(df)
                country = st.selectbox('Select country: ',
                                       df[[option[1]]].drop_duplicates())

                # Filter data by country
                c = [country]

                data[data[option[1]].isin(c)]

                y = data[option[2]]

                days = st.slider('Select a number of days to predict', 5, 1000)

                grade = st.slider('Select a polynomial grade prediction: ', 1,
                                  5)
                # print(y)
                generatePredictionGraph(y, grade, days, y.max())

                interpretacion = """
                La gráfica muestra una predicción de personas infectadas en {} en un periodo de {} días, y la predicción está siendo representada por medio de un polinomio
                de grado {}. (Es importante tomar en cuenta que entre mayor sea el grado de un polinomio mucho más precisa será la predicción obtenida).
                """.format(c, days, grade)

                st.info(interpretacion)

                export_as_pdf = st.button("Export Report")
                pdf_title = '2.  Predicción de Infectados en un País.'
                #content = ""

                if export_as_pdf:
                    write_pdf(pdf_title, interpretacion, 'D:\\prediction.png')
                pass

        else:

            option = st.multiselect(
                'Select a field, and numeric variable to filter [ex. country, infections]: ',
                data.columns)

            country = st.selectbox('Select country',
                                   data[option[0]].drop_duplicates())

            data = data[data[option[0]].isin([country])]

            data[[option[0], option[1]]]

            days = st.slider('Select a number of days to predict', 5, 1000)

            grade = st.slider('Select a polynomial grade prediction: ', 1, 5)

            y = data[option[1]]
            generatePredictionGraph(y, grade, days, y.max())
            #st.write(data)
            export_as_pdf = st.button("Export Report")
            pdf_title = '2.  Predicción de Infectados en un País.'
            content = ""

            if export_as_pdf:
                write_pdf(pdf_title, content, 'D:\\prediction.png')
            pass

    except Exception as e:
        st.write(e)
        st.warning('Please select three fields')

    # infected = st.multiselect('Select numeric variable: ', data.columns)

    pass


# 3. Indice de Progresión de la pandemia.
def pandemicProgression(data: DataFrame):

    options = st.multiselect('Select variable and order by [cases, date]: ',
                             data.columns)

    if options.__len__() < 2:

        st.warning('Please select a numeric variable and date')

    else:

        cases = options[0]
        date_ = options[1]

        data = data[[date_, cases]]

        # data['fecha'] = pd.to_datetime(data['fecha'])
        data[date_] = pd.to_datetime(data[date_])
        # data = data.sort_values(by='fecha')
        data = data.sort_values(by=date_)
        data = data.groupby(date_)[cases].sum()
        st.write(data)

        st.subheader('Global pandemic progression')

        st.line_chart(data)
        st.caption('Pandemic progression')
        st.info("""
        Esta grafica representa el progreso de infecciones que ha habido desde el inicio de la pandemia en todo el mundo
        """)

        generateTendencyGraph(data, 'Pandemic progression regression',
                              data.max())

        interpretacion = """
        La grafica de tendencia representa el comportamiento que tendrá la variable estudiada "{}" a lo largo del tiempo, y la pendiente indica 
        si la tendencia irá aumentando o disminuyendo con el tiempo.
        """.format(cases)

        st.info(interpretacion)

        export_as_pdf = st.button("Export Report")
        pdf_title = '3. Indice de Progresión de la pandemia.'
        #content = ""

        if export_as_pdf:
            write_pdf(pdf_title, interpretacion, 'D:\\trend.jpg')


# 4. Predicción de mortalidad por COVID en un Departamento.
def covidDeathsPredictionByDeparment(data: DataFrame):

    try:
        # date, state, cases
        data_options = st.multiselect(
            'Select fields [date, region, cases, filter]: ', data.columns)

        if data_options.__len__() < 4:
            st.warning('Select more fields to analyze')

        else:

            date_ = data_options[0]
            region = data_options[1]
            cases = data_options[2]
            flter = data_options[3]

            st.write(data_options)

            country_option = st.selectbox('Select country',
                                          data[region].drop_duplicates())

            ## select states
            c = [country_option]
            cs = data[data[region].isin(c)]

            province = st.selectbox('Select state/province/department',
                                    cs[flter].drop_duplicates())

            # Filter by deparment/state
            dep = cs[data[flter].isin([province])]

            ## convert dates
            dep[date_] = pd.to_datetime(dep[date_])
            dep = dep.sort_values(by=date_)
            st.subheader('Deaths in {} by day'.format(province))
            st.write(dep[[date_, cases]])

            y = dep[cases]

            # slider
            n_days = st.slider('Select number of days to predict: ', 5, 100)
            grade = st.slider('Select grade of the regression: ', 1, 3)
            max_val = dep[cases].max()

            generatePredictionGraph(y, grade, n_days, max_val)

            interpretacion = """
            La gráfica de predicción nos indica el posible valor de la variable "{}" en un periodo de {} días, 
            representado con un polinomio de grado {}.
            """.format(cases, n_days, grade)

            st.info(interpretacion)

            export_as_pdf = st.button("Export Report")
            pdf_title = '4. Predicción de mortalidad por COVID en un Departamento.'
            #content = ""

            if export_as_pdf:
                write_pdf(pdf_title, interpretacion, 'D:\\prediction.png')

    except Exception as e:
        st.write(e)
        st.warning('Select a field')


# 5. Predicción de mortalidad por COVID en un País (ARRRRRREGLAR)
def covidDeathPredictionByCountry(data: DataFrame):

    try:

        options_selected = st.multiselect(
            'Select fields to filter [country, deaths]: ', data.columns)

        if options_selected.__len__() < 2:
            st.warning('Plase select fields to filter')
        else:
            fltr = options_selected[1]
            country = st.selectbox('Please select a country: ',
                                   data[options_selected[0]].drop_duplicates())

            data = data[data[options_selected[0]].isin([country])]
            has_date = st.checkbox('The file has "date" field')
            data

            if has_date:

                date_ = st.selectbox('Select a type "date" field ',
                                     data.columns)

                data[date_] = pd.to_datetime(data[date_])

                # flt = data[[date_, fltr]]
                # st.write(flt.reset_index())
                # flt = flt.sort_values(by=date_)
                # st.write(flt)
                data = data.sort_values(by=date_).reset_index()

                data

                #data[['dateRep', 'deaths']]
                data[fltr]

                n_days = st.slider('Select a number of days to predict: ', 100,
                                   1000)
                grade = st.slider('Select grade of the polynomial grade: ', 1,
                                  10)

                generatePredictionGraph(data[fltr], grade, n_days,
                                        data[fltr].max())

                # export_as_pdf = st.button("Export Report")
                # pdf_title = '4. Predicción de mortalidad por COVID en un Departamento.'

                # if export_as_pdf:
                #     write_pdf(pdf_title, interpretacion, 'D:\\prediction.png')
            else:
                pass

    except Exception as e:
        st.write(e)
        st.warning('An error has occurred')

        #

        #flt[[]]

        #st.write(options_selected)


# 6. Análisis del número de muertes por coronavirus en un País.
def covidDeathsByCountry(data: DataFrame):

    data_options = st.multiselect(
        'Select field, and variables to analize [ex. country, deaths, date]: ',
        data.columns)

    try:

        if data_options.__len__() < 3:
            st.write('Please select more fields')
        else:

            # IMPORTANTE REVISAR AQUI XD
            country = st.selectbox('Select country',
                                   data[data_options[0]].drop_duplicates())

            data = data[data[data_options[0]].isin([country])]

            data[data_options[2]] = pd.to_datetime(data[data_options[2]])

            st.subheader('Deaths analysis in {}'.format(country))
            st.write(data[[data_options[1], data_options[2]]])

            y = data[data_options[1]]

            generateTendencyGraph(y, 'Deaths in {}'.format(country), y.max())
            intepretacion = """
            Esta grafica analiza el comportamiento del numero de muertes en {} desde que empezó la pandemia, así como la pendiente que indicará si el numero
            de fallecidos a causa del COVID-19 va a aumentar o disminuir con el paso del tiempo. Si la pendiente es creciente, indica que la cantidad de muertes
            tiende a crecer, pero si la pendiente es decreciente el numero de muertes tiende a disminuir con el tiempo.
            """.format(country)
            st.info(intepretacion)

            export_as_pdf = st.button("Export Report")
            pdf_title = '6. Análisis del número de muertes por coronavirus en un País.'
            #content = ""

            if export_as_pdf:
                write_pdf(pdf_title, intepretacion, 'D:\\trend.jpg')

    except Exception as e:
        st.write(e)
        st.warning("Please select a field")


# 7. Tendencia del número de infectados por día de un País
def covidInfectedByDay(data: DataFrame):

    select_col = st.multiselect(
        'Select a columns to parameterize (date, country and numeric variable): ',
        data.columns)

    #st.write(select_col)

    try:
        # select_date = st.selectbox('Select a date: ',
        #                            data[select_col[0]].drop_duplicates())

        if select_col.__len__() < 3:
            st.warning('Select more fields')
        else:
            select_country = st.selectbox(
                'Select a country: ', data[select_col[1]].drop_duplicates())

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

            fig = country_infections.groupby(
                [select_col[0],
                 select_col[1]]).sum().sort_values(by=select_col[0]).head()

            st.write(fig)

            # x = []
            y = country_infections[select_col[2]]
            hd = "COVID-19 Spread tendency in " + country[0]
            max_val = country_infections[select_col[2]].max()

            generateTendencyGraph(y, hd, max_val)
            interpretacion = """
            La gráfica muestra la tendencia del número diario de infectados en {}, al observar detenidamente la grafica, si la pendiente es creciente
            esto significa que el numero de contagios en {} estará aumentando con el paso del tiempo, pero si la pendiente es decreciente significa que el numero
            de contagios diarios irá decreciendo con el paso del tiempo, o por lo menos hasta la llegada de una nueva ola de contagios.
            """.format(select_country, select_country)

            st.info(interpretacion)

            export_as_pdf = st.button("Export Report")
            pdf_title = '7. Tendencia del número de infectados por día de un País'
            #content = ""

            if export_as_pdf:
                write_pdf(pdf_title, interpretacion, 'D:\\trend.jpg')

    except Exception as e:
        st.write(e)
        st.warning('Select the fields ')

    pass


# 8. Predicción de casos de un país para un año.
def casesPredictionOneYear(data: DataFrame):

    select_option = st.selectbox('Select a field to filter [country]: ',
                                 data.columns)

    country = st.selectbox('Select a country: ',
                           data[select_option].drop_duplicates())

    data = data[data[select_option].isin([country])]

    var = st.selectbox('Select a variable to predict', data.columns)

    has_date = st.checkbox('This file has "date" field')

    if has_date:

        date_ = st.selectbox('Order by: ', data.columns)

        data[date_] = pd.to_datetime(data[date_])

        flt = data[[date_, var]]
        st.write(flt)
        flt = data.sort_values(by=date_)

        st.write(flt[[date_, var]])

        grade = st.slider('Select polynomial grade: ', 1, 10)
        generatePredictionGraph(flt[var], grade, 365, flt[var].max())
        interpretacion = """
        La gráfica nos indica la predicción de casos que tendrá {} despues de un año del primer contagio por COVID-19, es importante tomar en cuenta
        que es un valor estimado, y que el grado del polinomio usado tambien indica la precisión de los valores obtenidos.
        """.format(country)
        st.write()

        export_as_pdf = st.button("Export Report")
        pdf_title = '8. Predicción de casos de un país para un año.'
        #content = ""

        if export_as_pdf:
            write_pdf(pdf_title, interpretacion, 'D:\\prediction.png')

    else:

        y = data[var]

        days = st.slider('Select number of days to predict: ', 10, 1000)
        grade = st.slider('Select polynomial grade: ', 1, 10)

        st.write(y)
        generatePredictionGraph(y, grade, days, y.max())
        interpretacion = """
        La gráfica nos indica la predicción de casos que tendrá {} despues de un año del primer contagio por COVID-19, es importante tomar en cuenta
        que es un valor estimado, y que el grado del polinomio usado tambien indica la precisión de los valores obtenidos.
        """.format(country)
        st.write(interpretacion)

        export_as_pdf = st.button("Export Report")
        pdf_title = '8. Predicción de casos de un país para un año.'
        #content = ""

        if export_as_pdf:
            write_pdf(pdf_title, interpretacion, 'D:\\prediction.png')

    #generatePredictionGraph(data[var],6, 365, 500000)


# 9. Tendencia de la vacunación de en un País.
def vaccinationTendencyByCountry(data: DataFrame):

    data_options = st.multiselect(
        'Select [date, country/region, and numeric variable]: ', data.columns)

    try:

        if data_options.__len__() < 3:
            st.warning('Please select fields to filter')
        else:
            region = data_options[1]

            country = st.selectbox('Select country: ',
                                   data[region].drop_duplicates())

            # pais = 'Guatemala'
            # data = data[data[region].isin(['Guatemala'])]

            data = data[data[region].isin([country])]

            st.write(data)

            vac = data[data_options[2]]

            st.write(vac)

            generateTendencyGraph(vac, "Vaccines trend graph", vac.max() + 500)

            interpretacion = """Para poder comprender de mejor forma esta grafica,
            es importante tomar en cuenta la pendiente generada, por ejemplo si la pendiente es creciente (positiva)
            la tendencia de vacunacion COVID-19 en los dias siguientes al analisis será a la alta, pero si de lo contrario
            la pendiente de la grafica es decreciente (negativa), la tendencia numero de vacunaciones por COVID-19 es a la baja.
            """

            st.info(interpretacion)

            export_as_pdf = st.button("Export Report")
            pdf_title = '9. Tendencia de la vacunación de en un País.'
            #content = ""

            if export_as_pdf:
                write_pdf(pdf_title, interpretacion, 'D:\\trend.jpg')

    except Exception as e:
        st.write(e)
        st.warning(":c")


# 10. Ánalisis Comparativo de Vacunación entre 2 paises.
def vaccinationComparationByCountries(data: DataFrame):

    try:

        option = st.multiselect(
            'Select field and variable of comparation [place, variable, group by] : ',
            data.columns)

        if option.__len__() < 3:
            st.warning('Select more fields')

        else:

            place = option[0]
            variable = option[1]
            col1, col2 = st.columns(2)

            with col1:

                st.subheader('Select Place 1:')
                country1 = st.selectbox('Select country 1: ',
                                        data[place].drop_duplicates())
                p1 = data[data[place].isin([country1])]

                #st.write(p1[variable])
                t1 = p1[variable].sum()
                generateTendencyGraph(p1[variable],
                                      'Vaccinations in {}'.format(country1),
                                      p1[variable].max())

            with col2:

                st.subheader('Select Place 2:')
                country2 = st.selectbox('Select country 2: ',
                                        data[place].drop_duplicates())
                p2 = data[data[place].isin([country2])]

                #st.write(p2[variable])
                t2 = p2[variable].sum()
                generateTendencyGraph(p2[variable],
                                      'Vaccinations in {}'.format(country2),
                                      p2[variable].max())

            ## graph
            plotdata = pd.DataFrame({str(variable): [t1, t2]},
                                    index=[country1, country2])

            st.bar_chart(plotdata)
            st.caption('Vaccination comparative graph')

            ## Interpretación
            interpretacion = ""
            if t1 > t2:
                st.info(
                    'De acuerdo a la grafica, {} presenta una mejor tasa de vacunacion que {}'
                    .format(country1, country2))
                interpretacion += 'De acuerdo a la grafica, {} presenta una mejor tasa de vacunacion que {}'.format(
                    country1, country2)

            else:

                st.info(
                    'De acuerdo a la grafica, {} presenta una mejor tasa de vacunacion que {}'
                    .format(country1, country2))
                interpretacion += 'De acuerdo a la grafica, {} presenta una mejor tasa de vacunacion que {}'.format(
                    country1, country2)

            st.info(
                """Es importante analizar los datos de las vacunaciones, ya que en este aspecto podemos analizar las diferentes situaciones
            en los que puede encontrar cada pais, los países más desarrollados tienen acceso más rapido a las vacunas que los paises menos desarrollados
            debido a factores como el economico, y politico (como la corrupción) o incluso la cultura de los países, ya que existen grupos
            de personas que debido a creencias culturales, religiosas, etc. se rehusan a vacunarse."""
            )

            interpretacion += """Es importante analizar los datos de las vacunaciones, ya que en este aspecto podemos analizar las diferentes situaciones
            en los que puede encontrar cada pais, los países más desarrollados tienen acceso más rapido a las vacunas que los paises menos desarrollados
            debido a factores como el economico, y politico (como la corrupción) o incluso la cultura de los países, ya que existen grupos
            de personas que debido a creencias culturales, religiosas, etc. se rehusan a vacunarse."""

            export_as_pdf = st.button("Export Report")
            pdf_title = '10. Ánalisis Comparativo de Vacunación entre 2 paises.'
            #content = ""

            if export_as_pdf:
                write_pdf(pdf_title, interpretacion, 'D:\\trend.jpg')

            # st.set_option('deprecation.showPyplotGlobalUse', False)
            # plotdata.plot(kind="bar", color="green")
            # plt.title("Comparative")
            # st.pyplot()

    except Exception as e:

        st.write(e)
        st.warning('Error :(')


# 11. Porcentaje de hombres infectados por covid-19 en un País desde el primer caso activo
def menPercentageInfected(data: DataFrame):

    field = st.selectbox('Select field to filter: ', data.columns)

    try:

        country = st.selectbox('Select country: ',
                               data[field].drop_duplicates())

        data = data[data[field].isin([country])]

        gender = st.multiselect('Select variables to analize [men, women]: ',
                                data.columns)

        if gender.__len__() < 2:
            st.warning('Select fields to compare. ')

        else:

            gender1 = gender[0]
            gender2 = gender[1]

            #data[[gender1, gender2]]
            t1 = data[gender1].sum()
            t2 = data[gender2].sum()

            data

            st.write("--" * 100)

            st.subheader('Total infected: {}'.format(t1 + t2))

            col1, col2 = st.columns(2)

            with col1:
                per1 = (t1 / (t1 + t2)) * 100
                st.subheader('Total of {} infected: {} '.format(gender1, t1))
                st.write('* {}%'.format(per1.__round__(3)))

            with col2:
                per2 = (t2 / (t1 + t2)) * 100
                st.subheader('Total of {} infected: {} '.format(gender2, t2))
                st.write('* {}%'.format(per2.__round__(3)))

            # data
            st.spinner()
            with st.spinner(text='Loading charts'):
                time.sleep(3)
                df = {
                    'Data': [per1.__round__(3),
                             per2.__round__(3)],
                    'index': ['Men', 'Women']
                }
                st.subheader('Men % infected since the first day ')
                # fig = px.pie(df, values='Data', names='Data')
                # st.plotly_chart(fig, use_container_width=True)
                df = DataFrame(df)

                df.plot(kind="bar", color="blue")
                plt.title("Comparative Men % infected since the first day ")
                plt.savefig('D:\\comparative.png')
                st.pyplot()

                st.caption(
                    'Comparative of Men % infected since the first day ')

                interpretacion = "Esta gráfica muestra el porcentaje de hombres infectados frente al total de personas infectadas desde el primer caso de COVID 19 detectado, actualizado a los datos del ultimo dia que contiene el archivo."
                st.info(interpretacion)

                export_as_pdf = st.button("Export Report")
                pdf_title = '11. Porcentaje de hombres infectados por covid-19 en un País desde el primer caso activo'
                #content = ""

                if export_as_pdf:
                    write_pdf(pdf_title, interpretacion, 'D:\\comparative.png')
                #st.line_chart(data[[gender1, gender2]])
                #st.success('Done')
    except Exception as e:
        st.write(e)
        st.warning('Error :c')
    pass


# 12. Ánalisis Comparativo entres 2 o más paises o continentes.
def covidComparative(data: DataFrame):

    try:

        option = st.multiselect(
            'Select field and variable of comparation [place, variable, group by] : ',
            data.columns)

        if option.__len__() < 3:
            st.warning('Select more fields')
        else:

            place = option[0]
            variable = option[1]

            countries = st.multiselect('Select list of countries: ',
                                       data[place].drop_duplicates())
            st.write(countries)

            elements = []
            #size =
            if countries.__len__() >= 2:

                for i in range(0, countries.__len__()):
                    country = countries[i]
                    temp = data[data[place].isin([country])]
                    val = temp[variable].sum()
                    elements.append(val)

                #generateTendencyGraph(elements, '--', 1000000)

                # Graph
                plotdata = pd.DataFrame({
                    str(variable): elements,
                },
                                        index=[countries])

                #st.bar_chart(plotdata,)
                st.spinner()
                with st.spinner(text="loading..."):
                    time.sleep(3)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    plotdata.plot(kind="bar", color="blue")
                    plt.title(
                        "Comparative between two or more countries or continents"
                    )
                    plt.savefig("D:\\compcountries.png")
                    st.pyplot()

                    interpretacion = """
                    Esta grafica representa la COMPARACIÓN de "{}" entre los paises seleccionados, para poder determinar qué país cuenta con los numeros más altos
                    para poder sacar conclusiones y poder predecir sobre lo que ocurrirá en el futuro con la pandemia.
                    """.format(variable)
                    st.info(interpretacion)

                    export_as_pdf = st.button("Export Report")
                    pdf_title = '12. Ánalisis Comparativo entres 2 o más paises o continentes.'
                    #content = ""

                    if export_as_pdf:
                        write_pdf(pdf_title, interpretacion,
                                  'D:\\compcountries.png')
            else:
                st.warning('Please, select more countries ')

    except Exception as e:

        st.write(e)
        st.warning('Error :(')

    # if option == 'Countries':

    # elif option == 'Continents':

    #     col1, col2 = st.columns(2)

    #     with col1:
    #         st.subheader('Select Continent 1:')
    #         continent1 = st.selectbox('Select continent 1: ', data.columns)

    #     with col2:
    #         st.subheader('Select Continent 2:')
    #         continent2 = st.selectbox('Select continent 2: ', data.columns)


# 13. Muertes promedio por casos confirmados y edad de covid 19 en un País.


# 14. Muertes según regiones de un país - Covid 19.
def deathsByCountryRegions(data: DataFrame):

    option_selected = st.multiselect('Select field to filter [country]: ',
                                     data.columns)

    try:

        if option_selected.__len__() == 0:
            st.warning('Please select a filter: ')

        else:
            flter = option_selected[0]

            country = st.selectbox('Select country: ',
                                   data[flter].drop_duplicates())

            data = data[data[flter].isin([country])]

            filters = st.multiselect(
                'Select variables to filter ex: [ state/department/region, deaths]',
                data.columns)

            if filters.__len__() < 2:
                st.warning('Select more filters')

            else:

                data = data.groupby(filters[0])[filters[1]].sum()

                st.subheader("{} by {} in {}".format(filters[1], filters[0],
                                                     country).capitalize())

                st.spinner()
                with st.spinner(text="loading data..."):
                    time.sleep(3)

                    data.plot(kind="bar", color="blue")
                    plt.title(
                        "Comparative between two or more countries or continents"
                    )
                    plt.savefig("D:\\deathsregion.png")
                    st.pyplot()
                    data

                    st.info("""
                    Esta tabla representa el numero de "{}" por cada "{}" en {}
                    """.format(filters[1], filters[0], country))

                interpretacion = """
                    Esta grafica representa el numero de "{}" por cada "{}" en {}
                    """.format(filters[1], filters[0], country)

                export_as_pdf = st.button("Export Report")
                pdf_title = "14. Muertes según regiones de un país - Covid 19."
                if export_as_pdf:
                    write_pdf(pdf_title, interpretacion,
                              'D:\\deathsregion.png')

    except Exception as e:
        st.write(e)
        st.warning('Error :c')

    pass


# 15. Tendencia de casos confirmados de Coronavirus en un departamento de un País
def covidCasesByDep(data: DataFrame):

    try:
        # date, state, cases
        data_options = st.multiselect(
            'Select fields [date, region/country, cases, state/province/department]: ',
            data.columns)

        if data_options.__len__() < 4:
            st.warning('Please select more fields. ')
        else:
            st.write(data_options)

            country_option = st.selectbox(
                'Select country', data[data_options[1]].drop_duplicates())

            ## select states
            c = [country_option]
            cs = data[data[data_options[1]].isin(c)]

            # Select department
            province = st.selectbox('Select state/province/department',
                                    cs[data_options[3]].drop_duplicates())

            #st.write(cs)
            # Filter by deparment/state
            dep = cs[data[data_options[3]].isin([province])]

            #st.write(dep)
            #st.write(dep[[data_options[0], data_options[2]]])

            ## convert dates
            dep[data_options[0]] = pd.to_datetime(dep[data_options[0]])
            dep = dep.sort_values(by=data_options[0])

            st.subheader('"{}" data table in {}'.format(
                data_options[2], country_option).capitalize())
            st.write(dep[[data_options[0], data_options[2]]])

            y = dep[data_options[2]]
            hd = "COVID-19 Spread tendency in " + province
            max_val = dep[data_options[2]].max()
            #st.write(max_val)
            generateTendencyGraph(y, hd, max_val)

            interpretacion = """ 
            La gráfica de tendencia de casos confirmados nos muestra el comportamiento que 
            tendrá la variable "{}" para el rango de días que contiene el dataset. Es importante resaltar
            la importancia que tiene la pendiente, ya que si la pendiente es positiva, la tendencia será de aumento
            de casos confirmados, por lo cual deben las autoridades tomar medidas para frentar ese incrementos, pero
            si la pendiente es negativa, esto es un indicativo que la tendencia en los proximos días será de que habrá
            una disminución notable de los casos.
            """.format(data_options[0])
            st.info(interpretacion)

            export_as_pdf = st.button("Export Report")
            pdf_title = '15. Tendencia de casos confirmados de Coronavirus en un departamento de un País'
            #content = ""

            if export_as_pdf:
                write_pdf(pdf_title, interpretacion, 'D:\\trend.jpg')

    except Exception as e:
        st.write(e)
        st.warning('Select a field')


# 16. Porcentaje de muertes frente al total de casos en un país, región o continente
def percentageDeathsCases(data: DataFrame):

    try:

        options = st.multiselect(
            'Select filter [country, region or continent] and [death, cases]',
            data.columns)

        if options.__len__() >= 3:

            interpretacion = ""
            st.write(options)
            reg = options[0]
            var1 = options[1]
            var2 = options[2]

            place = st.selectbox('Select place: ', data[reg].drop_duplicates())

            data = data[data[reg].isin([place])]

            # total deaths
            t1 = data[var1].sum()
            t2 = data[var2].sum()
            st.write('* total deaths in {}: {}'.format(place, t1).capitalize())
            interpretacion += '* total deaths in {}: {} \n'.format(
                place, t1).capitalize()
            st.write('* total cases in {}: {}'.format(place, t2).capitalize())
            interpretacion += '* total cases in {}: {} \n'.format(
                place, t2).capitalize()
            perc1 = (t1 / t2) * 100
            perc2 = 100 - perc1
            # st.subheader('Percentege {}, {}'.format(perc1.__round__(2),
            #                                         perc2.__round__(2)))

            interpretacion += '* total percentage deaths in {}: {}% \n'.format(
                place, perc1.__round__(3)).capitalize()
            interpretacion += '* total percentage cases in {}: {}% \n'.format(
                place, perc2.__round__(3)).capitalize()

            df = DataFrame({'Total': [perc1.__round__(3),
                                      perc2.__round__(3)]},
                           index=[var1, var2])

            st.spinner()

            with st.spinner(text='Loading data...'):
                time.sleep(5)
                st.subheader('Death percentages in {}'.format(place))
                df

                st.subheader('Deaths percentage vs Positive cases')

                fig = px.pie(df, values='Total', names='Total')
                st.plotly_chart(fig, use_container_width=True)

                st.info(
                    "Esta grafica muestra el porcentaje de muertes que hay entre casos activos en {}"
                    .format(place))

                st.line_chart(data[[var1, var2]])
                st.info("""
                Para poder entender la grafica anterior con mas claridad, la grafica lineal muestra 
                el comportamiento de muertes que hay entre casos activos en {}
                """.format(place))

                export_as_pdf = st.button("Export Report")
                pdf_title = '16. Porcentaje de muertes frente al total de casos en un país, región o continente'
                #content = ""

                if export_as_pdf:
                    write_pdf(
                        pdf_title, interpretacion,
                        'https://htmlcolorcodes.com/assets/images/colors/white-color-solid-background-1920x1080.png'
                    )
        else:
            st.warning('Select more variables')

    except Exception as e:
        st.write(e)
        st.write('Error :C')


# 17. Tasa de comportamiento de casos activos en relación al número de muertes en un continente
def performoranceRateCasesDeaths(data: DataFrame):

    options = st.multiselect('Select filters: [place, date ,cases, deaths]',
                             data.columns)
    st.write(options)

    place = options[0]
    date_ = options[1]
    cases = options[2]
    deaths = options[3]

    try:

        if options.__len__() < 4:
            st.warning('Please select more fields')

        else:

            continent = st.selectbox('Select continent: ',
                                     data[place].drop_duplicates())

            flt = data[data[place].isin([continent])]

            flt[date_] = pd.to_datetime(flt[date_])
            #flt[[date_, cases, deaths]]

            flt[[date_, deaths]]
            #st.line_chart(flt[[date_, deaths]])
            # st.caption('Cases / Deaths performance during the pandemic. ')
            # st.info(
            #     'Esta grafica muestra el comportamiento de las muertes y los casos confirmados a lo largo de la pandemia en {}'
            #     .format(place))

        pass
    except Exception as e:

        st.write(e)
        st.warning('Error :(')

    pass


# 18. Comportamiento y clasificación de personas infectadas por COVID-19 por municipio en un País.
def classificationInfectedPeopleByState(data: DataFrame):

    select_field = st.selectbox('Select a field to filter: ', data.columns)

    try:

        country = st.selectbox('Select country',
                               data[select_field].drop_duplicates())

        #data = data[data[select_field].isin([country])]

        country_filter = st.selectbox(
            'Filter country by region/state/province/deparment', data.columns)

        data = data[data[select_field].isin([country])]

        state = st.selectbox('Select state/province/department: ',
                             data[country_filter].drop_duplicates())

        data = data[data[country_filter].isin([state])]

        #data
        filters = st.multiselect('Select variables to filter: ', data.columns)

        if filters.__len__() < 2:
            st.warning('Please select more variables to filter')

        else:
            var1 = filters[0]
            var2 = filters[1]

            has_date = st.checkbox('This file has "date" column ')

            if has_date:

                date_field = st.selectbox('Select date type field',
                                          data.columns)

                data[date_field] = pd.to_datetime(data[date_field])
                data = data.sort_values(by=date_field).reset_index()

                st.subheader('Data classifications')
                st.dataframe(data[[date_field, var1, var2]])

                ## AQUI ME QUEDE xd
                ## columns
                col1, col2 = st.columns(2)

                st.spinner()
                with st.spinner(text="Loading..."):
                    time.sleep(3)

                with col1:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    generateTendencyGraph(
                        data[var1],
                        '{} trend graphic in {}'.format(var1, state),
                        data[var1].max())

                    st.info("""
                    Esta grafica contiene la tendencia que tiene la variable "{}"
                    con el paso del tiempo, es importante analizar la pendiente que se genera
                    para saber si irá incrementando o decrementanto el valor de la variable estudiada
                    """.format(var1))

                with col2:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    generateTendencyGraph(
                        data[var2],
                        '{} trend graphic in {}'.format(var2, state),
                        data[var2].max())
                    st.info("""
                    Esta grafica contiene la tendencia que tiene la variable "{}"
                    con el paso del tiempo, es importante analizar la pendiente que se genera
                    para saber si irá incrementando o decrementanto el valor de la variable estudiada
                    """.format(var2))
            else:
                pass

        #data[[var1, var2]]

    except Exception as e:
        st.write(e)
        st.warning('Error :c')

    pass


# 19. Predicción de muertes en el último día del primer año de infecciones en un país.


# 20. Tasa de crecimiento de casos de COVID-19 en relación con nuevos casos diarios y tasa de muerte por COVID-19 **
def growthRateCasesAndDeathRate(data: DataFrame):

    try:

        selected_options = st.multiselect(
            'Select variables to analize: [positives, deaths]', data.columns)

        if selected_options.__len__() < 2:
            st.warning('Please select more fields')
        else:
            var1 = selected_options[0]
            var2 = selected_options[1]

            has_date = st.checkbox('This file has Date type field ')

            if has_date:
                date_field = st.selectbox('Select date type field',
                                          data.columns)

                data[date_field] = pd.to_datetime(data[date_field])
                data = data.sort_values(by=date_field).reset_index()
                st.dataframe(data[[date_field, var1, var2]])

                # Tasa de crecimiento de casos de COVID-19 en relación con nuevos casos diarios
                # tasa de muerte por COVID-19
            else:
                pass

    except Exception as e:
        st.write(e)
        st.warning('Error :(')


# 21. Predicciones de casos y muertes en todo el mundo - Neural Network MLPRegressor


# 22. Tasa de mortalidad por coronavirus (COVID-19) en un país. ***
def deathsRateByCountry(data: DataFrame):

    option = st.selectbox('Select a field to filter [ex: country]',
                          data.columns)

    country = st.selectbox('Select country: ', data[option].drop_duplicates())

    numeric_vars = st.multiselect(
        'Select numeric varabales to analyze [ex: deaths, positives]: ',
        data.columns)

    if numeric_vars.__len__() == 2:

        deaths = numeric_vars[0]
        positives = numeric_vars[1]

        data = data[data[option].isin([country])]

        flt = data[[deaths, positives]].reset_index()

        st.subheader("Deaths / Positive people in {} ".format(country))
        st.write(flt)

        deathrate = []

        for i in range(0, flt.__len__()):
            # death_rate = (n1/n2) * 100
            n1 = flt[deaths][i]  # deaths
            n2 = flt[positives][i]  # positives

            if n1 == 0 or n2 == 0:
                deathrate.append(0)
            else:
                # st.write('xd')
                death_rate = (n1 / n2) * 100
                deathrate.append(death_rate)

        #xd = pd.DataFrame(deathrate)
        #st.head
        generateTendencyGraph(deathrate, "Death rate by country", 20)

        st.info("""
        La grafica de arriba representa el porcentaje de personas fallecidas que hay por el numero de personas 
        que contraen COVID-19 en {}, una pendiente decreeciente indica que la tasa de mortalidad tiende a bajar con el paso del tiempo
        , pero una pendiente positiva indica que la tasa de moratalidad tiene una tendencia a ser más alta en el futuro. Lo cual son
        datos muy importantes para las autoridades, quienes tienen que tomar acciones para evitar 
        que la tasa de mortalidad suba.
        """.format(country))

        st.write('--' * 100)

        df = {'Death Rate': deathrate}

        st.subheader(
            'Grafica lineal de la tasa de mortalidad en {}'.format(country))
        st.line_chart(df)
        st.caption('Death Linear rate graph')
        #de(xd, 3, 20, 1000)

        #st.write(deathrate)

        pass

    pass


# 23. Factores de muerte por COVID-19 en un país.


# 24. Comparación entre el número de casos detectados y el número de pruebas de un país.
def covidCasesTestComparation(data: DataFrame):

    try:
        options = st.multiselect(
            'Select fields to filter [Country, cases, tests]', data.columns)

        if options.__len__() < 3:
            st.warning('Select more fields.')

        else:
            place = options[0]
            var1 = options[1]
            var2 = options[2]

            country = st.selectbox('Select country',
                                   data[place].drop_duplicates())

            flt = data[data[place].isin([country])]

            val1 = flt[var1].sum()
            val2 = flt[var2].sum()

            #st.write(val1.sum())
            #st.write(val2.sum())
            st.subheader(
                "Comparación entre el numero de casos y numero de pruebas")
            plotdata = pd.DataFrame({'Data': [val1, val2]}, index=[var1, var2])
            plotdata.plot(kind="bar", color="blue")
            plt.title(
                "Comparación entre el numero de casos y numero de pruebas")
            plt.savefig("D:\\compcasestests.png")
            st.pyplot()
            #st.bar_chart(plotdata)

            export_as_pdf = st.button("Export Report")
            pdf_title = ' 24. Comparación entre el número de casos detectados y el número de pruebas de un país.'
            #content = ""

            if export_as_pdf:
                write_pdf(pdf_title, " ", 'D:\\compcasestests.png')

    except Exception as e:

        st.write(e)
        st.warning('Error :(')


# 25. Predicción de casos confirmados por día
def covidCasesPredictionByDay(data: DataFrame):

    try:

        field = st.multiselect('Select filter, and order by: ', data.columns)

        if field.__len__() < 2:
            st.warning('Please select more fields')
        else:

            flter = field[0]
            order_by = field[1]

            st.write('Filtering by: ', field)

            data[order_by] = pd.to_datetime(data[order_by])

            st.write(data)

            data = data.groupby(order_by)[flter].sum()

            st.subheader('{} field analysis'.format(flter).capitalize())
            st.write(data)
            n_days = st.slider('Select number of days to predict ', 10, 1000)
            grade = st.slider('Select polynomial grade ', 1, 10)
            #y = data[flter]

            st.spinner()
            with st.spinner(text="Generating prediction..."):
                time.sleep(3)
                generatePredictionGraph(data, grade, n_days, data.max())

            prediccion = """
            La predicción realizada para un periodo de {} días con un polinomio de grado {} brindará un valor proximado en cuanto al numero de "{}".
            Un polomio de grado mayor nos brinda mayor precision que una grafica lineal.
            """.format(n_days, grade, flter)

            st.info(prediccion)

            export_as_pdf = st.button("Export Report")
            pdf_title = '25. Predicción de casos confirmados por día'
            #content = ""

            if export_as_pdf:
                write_pdf(pdf_title, prediccion, 'D:\\prediction.png')

    except Exception as e:

        st.write(e)
        st.warning('Error :(')


# ===================== END METHODS =====================

# Sidebar option tuple
sid_opt_tuple = ('COVID Cases', 'COVID Deaths', 'Vaccines')

#  **** OPTION TUPLES ****
# Covid deaths
covid_deaths_tuple = (
    'Análisis del número de muertes por coronavirus en un País.',
    'Predicción de mortalidad por COVID en un Departamento.',
    'Predicción de mortalidad por COVID en un País',
    'Porcentaje de muertes frente al total de casos en un país, región o continente',
    'Tasa de mortalidad por coronavirus (COVID-19) en un país.',
    'Muertes según regiones de un país - Covid 19.')
# Covid Cases
covid_cases_tuple = (
    'Tendencia de la infección por Covid-19 en un País.',
    'Indice de Progresión de la pandemia.',
    'Predicción de Infectados en un País.',
    'Ánalisis Comparativo entres 2 o más paises o continentes.',
    'Tendencia del número de infectados por día de un País',
    'Tendencia de casos confirmados de Coronavirus en un departamento de un País',
    'Comparación entre el número de casos detectados y el número de pruebas de un país.',
    'Predicción de casos confirmados por día',
    'Tasa de comportamiento de casos activos en relación al número de muertes en un continente',
    'Predicción de casos de un país para un año.',
    'Porcentaje de hombres infectados por covid-19 en un País desde el primer caso activo',
    'Tasa de crecimiento de casos de COVID-19 en relación con nuevos casos diarios y tasa de muerte por COVID-19',
    'Comportamiento y clasificación de personas infectadas por COVID-19 por municipio en un País.'
)

# Covid Vaccines
covid_vaccines_tuple = ('Tendencia de la vacunación de en un País.',
                        'Ánalisis Comparativo de Vacunación entre 2 paises.')

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
        st.header('COVID Spread/Cases Reports ')

        select_report = st.selectbox('Select report', covid_cases_tuple)

        st.write(data)
        # Validate option
        if select_report == 'Tendencia de la infección por Covid-19 en un País.':
            covidInfectionTendence(data)

        elif select_report == 'Tendencia de casos confirmados de Coronavirus en un departamento de un País':
            covidCasesByDep(data)

        elif select_report == 'Indice de Progresión de la pandemia.':
            pandemicProgression(data)

        elif select_report == 'Predicción de Infectados en un País.':
            covidInfectedPredictionByCountry(data)

        elif select_report == 'Tendencia del número de infectados por día de un País':
            #select_date = st.selectbox('Select data to parameterize', data.)
            covidInfectedByDay(data)

        elif select_report == 'Ánalisis Comparativo entres 2 o más paises o continentes.':

            covidComparative(data)

        elif select_report == 'Comparación entre el número de casos detectados y el número de pruebas de un país.':

            covidCasesTestComparation(data)

        elif select_report == 'Predicción de casos confirmados por día':

            covidCasesPredictionByDay(data)

        elif select_report == 'Tasa de comportamiento de casos activos en relación al número de muertes en un continente':

            performoranceRateCasesDeaths(data)

        elif select_report == 'Predicción de casos de un país para un año.':

            casesPredictionOneYear(data)

        elif select_report == 'Porcentaje de hombres infectados por covid-19 en un País desde el primer caso activo':

            menPercentageInfected(data)

        elif select_report == 'Tasa de crecimiento de casos de COVID-19 en relación con nuevos casos diarios y tasa de muerte por COVID-19':

            growthRateCasesAndDeathRate(data)

        elif select_report == 'Comportamiento y clasificación de personas infectadas por COVID-19 por municipio en un País.':
            classificationInfectedPeopleByState(data)

        #report_text2 = st.text_input("Content")
        # export_as_pdf = st.button("Export Report")

        # if export_as_pdf:
        #     pdf = FPDF()
        #     pdf.add_page()
        #     pdf.set_font('Arial', 'B', 16)
        #     pdf.cell(40, 10, report_text)
        #     pdf.cell(40, 10, '\n\n\n')

        #     html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")

        #     st.markdown(html, unsafe_allow_html=True)

    elif sidebar_selectbox == 'COVID Deaths':

        st.header("COVID deaths Reports")
        select_report = st.selectbox('Select report', covid_deaths_tuple)

        st.write(data)
        if select_report == 'Análisis del número de muertes por coronavirus en un País.':
            covidDeathsByCountry(data)

        elif select_report == 'Predicción de mortalidad por COVID en un Departamento.':
            covidDeathsPredictionByDeparment(data)

        elif select_report == 'Predicción de mortalidad por COVID en un País':
            covidDeathPredictionByCountry(data)

        elif select_report == 'Porcentaje de muertes frente al total de casos en un país, región o continente':
            percentageDeathsCases(data)

        elif select_report == 'Tasa de mortalidad por coronavirus (COVID-19) en un país.':
            deathsRateByCountry(data)

        elif select_report == 'Muertes según regiones de un país - Covid 19.':
            deathsByCountryRegions(data)

    elif sidebar_selectbox == 'Vaccines':
        st.header('COVID Vaccines Reports')
        select_report = st.selectbox('Select report', covid_vaccines_tuple)

        st.write(data)
        if select_report == 'Tendencia de la vacunación de en un País.':
            vaccinationTendencyByCountry(data)

        elif select_report == 'Ánalisis Comparativo de Vacunación entre 2 paises.':
            vaccinationComparationByCountries(data)

else:

    st.warning("The file is empty or invalid, please upload a valid file")

# ## Validate sidebar option
