import streamlit as st
import joblib
import pandas as pd

# Load Model

model = joblib.load("model/model.pkl")

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

st.title("💻 Laptop Price Predictor")
st.write("Enter laptop specifications to predict its price.")

# Inputs

brand = st.selectbox(
    "Brand",
    ["Acer", "Asus", "Dell", "HP", "Lenovo"]
)

processor_speed = st.number_input(
    "Processor Speed (GHz)",
    min_value=1.0,
    max_value=5.0,
    value=2.5,
    step=0.1
)

ram = st.selectbox(
    "RAM (GB)",
    [4, 8, 16, 32]
)

storage = st.selectbox(
    "Storage Capacity (GB)",
    [256, 512, 1000]
)

screen = st.number_input(
    "Screen Size (inches)",
    min_value=11.0,
    max_value=18.0,
    value=15.6,
    step=0.1
)

weight = st.number_input(
    "Weight (kg)",
    min_value=2.0,
    max_value=5.0,
    value=2.5,
    step=0.1
)


# Predict


if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "Brand": [brand],
        "Processor_Speed": [processor_speed],
        "RAM_Size": [ram],
        "Storage_Capacity": [storage],
        "Screen_Size": [screen],
        "Weight": [weight]
    })

    prediction = model.predict(input_df)[0]

    st.success(f"💰 Predicted Price: ₹ {prediction:,.2f}")