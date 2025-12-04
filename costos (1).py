import streamlit as st 
import pandas as pd


# Configuración básica
st.set_page_config(page_title="Predicción de temperatura", layout="centered")

st.write("# Predicción de temperatura")
st.image("Temperatura.jpg",
         caption="Predicción de la temperatura de una ciudad de México en cierto año y mes.")

# 1. Cargar datos
datos = pd.read_csv("Temperature.csv", encoding="latin-1")

# (Opcional) ver columnas:
# st.write(datos.head())

# 2. Definir features (X) y target (y)
# En tu CSV las columnas son:
# ['AverageTemperature', 'City', 'Country', 'Month', 'Year']
# Usamos City, Month y Year como entradas del modelo
X = datos[["City", "Month", "Year"]]
y = datos["AverageTemperature"]

# 3. Modelo: train/test split y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1613777
)

LR = LinearRegression()
LR.fit(X_train, y_train)

st.header("Datos para la predicción")

# 4. Entradas del usuario (mismo estilo que tu otro código)
def user_input_features():
    City = st.number_input(
        "Ciudad (1- Acapulco, 2- Acuña o 3- Aguascalientes):",
        min_value=1,
        max_value=3,
        value=1,
        step=1,
    )

    Month = st.number_input(
        "Mes (de 1 a 12 según sea el mes):",
        min_value=1,
        max_value=12,
        value=1,
        step=1,
    )

    Year = st.number_input(
        "Año (desde 1800 a 2013):",
        min_value=1800,
        max_value=2013,
        value=2000,
        step=1,
    )

    user_input_data = {
        "City": City,
        "Month": Month,
        "Year": Year,
    }

    return pd.DataFrame(user_input_data, index=[0])

# Obtener inputs del usuario
df = user_input_features()

# 5. Predicción
prediccion = LR.predict(df)[0]

# 6. Mostrar predicción
st.subheader("Predicción de temperatura")
st.write(f"La temperatura será de: **{prediccion:.2f} °C**")
