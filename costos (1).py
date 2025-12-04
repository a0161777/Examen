import numpy as np
import streamlit as st
import pandas as pd


# Título de la app
st.write("# Predicción de temperatura")
st.image("Temperatura.jpg",
         caption="Predicción de la temperatura de una ciudad de México en cierto año y mes.")

st.header("Descripción de la predicción")


def user_input_features():

    # Entrada de ciudad
    City = st.number_input(
        'Ciudad (1- Acapulco, 2- Acuña o 3- Aguascalientes):',
        min_value=1,
        max_value=3,
        value=1,
        step=1
    )

    # Entrada de mes
    Month = st.number_input(
        'Mes (de 1 a 12 según sea el mes):',
        min_value=1,
        max_value=12,
        value=1,
        step=1
    )

    # Entrada de año
    Year = st.number_input(
        'Año (desde 1800 a 2013):',
        min_value=1800,
        max_value=2013,
        value=2000,
        step=1
    )

    # Diccionario con las características
    user_input_data = {
        "Cuidad": City,
        "Mes(del 1 al 12 según sea el mes)": Month,
        "Año (desde 1800 a 2013)": Year
    }

    # Convertir a DataFrame
    features = pd.DataFrame(user_input_data, index=[0])
    return features


# Obtener datos del usuario
df = user_input_features()

# Cargar base de datos
datos = pd.read_csv("Temperature.csv", encoding="latin-1")

# Separar variables
X = datos.drop(columns="AverageTemperature")
y = datos["AverageTemperature"]

# Entrenar el modelo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1613777
)

LR = LinearRegression()
LR.fit(X_train, y_train)

# Predecir
prediccion = LR.predict(df)[0]

# Mostrar predicción
st.subheader("Predicción de temperatura")
st.write(f"La temperatura será de: **{prediccion:.2f} °C**")
