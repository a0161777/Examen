
import numpy as np
import streamlit as st
import pandas as pd


st.write("# Predicción de temperatura")
st.image("Temperatura.jpg",
         caption="Predicción de la temperatura de una ciudad de México en cierto año y mes.")

st.header("Descripción de la predicción")

def user_input_features():

    City = st.number_input(
        'Ciudad (1- Acapulco, 2- Acuña o 3- Aguascalientes):',
        min_value=1,
        max_value=3,
        value=1,
        step=1
    )


    Month = st.number_input(
        'Mes (de 1 a 12 según sea el mes):',
        min_value=1,
        max_value=12,
        value=1,
        step=1
    )


    Year = st.number_input(
        'Año (desde 1800 a 2013):',
        min_value=1800,
        max_value=2013,
        value=2000,
        step=1
    )

   
    user_input_data = {
        "Cuidad ( 1- Acapulco, 2- Acuña y 3 - Aguascalientes)": City,
        "Mes(del 1 al 12 según sea el mes)": Month,
        "Año (desde 1800 a 2013)": Year
    }

  
    features = pd.DataFrame(user_input_data, index=[0])
    return features



df = user_input_features()


datos = pd.read_csv("Temperature.csv", encoding="latin-1")


X = datos.drop(columns="AverageTemperature")
y = datos["AverageTemperature"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613777)

LR = LinearRegression()
LR.fit(X_train, y_train)


b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['Cuidad( 1- Acapulco, 2- Acuña y 3 - Aguascalientes)'] + b1[1]*df['Mes(del 1 al 12 según sea el mes)'] + b1[2]*df["Año (desde 1800 hasta 2013)"]

st.subheader('Predicción de temperatura')
st.write('La temperatura será de: ', prediccion)
