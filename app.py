import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Company
company = st.selectbox("Brand", df["Company"].unique())

# Type
type_name = st.selectbox("Type", df["TypeName"].unique())

# RAM
ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input("Weight of Laptop (kg)", min_value=0.5, max_value=5.0, value=1.5)

# Touchscreen
touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])

# IPS
ips = st.selectbox("IPS Display", ["No", "Yes"])

# Screen Size
screen_size = st.slider("Screen Size (inches)", 10.0, 18.0, 13.3)

# Resolution
resolution = st.selectbox(
    "Screen Resolution",
    [
        "1920x1080",
        "1366x768",
        "1600x900",
        "3840x2160",
        "3200x1800",
        "2880x1800",
        "2560x1600",
        "2560x1440",
        "2304x1440"
    ]
)

# CPU
cpu = st.selectbox("CPU", df["Cpu brand"].unique())

# HDD
hdd = st.selectbox("HDD (GB)", [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox("SSD (GB)", [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox("GPU", df["Gpu brand"].unique())

# Operating System
os = st.selectbox("Operating System", df["os"].unique())


if st.button("Predict Price"):

    touchscreen = 1 if touchscreen == "Yes" else 0
    ips = 1 if ips == "Yes" else 0

    X_res = int(resolution.split("x")[0])
    Y_res = int(resolution.split("x")[1])

    ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size

    query = pd.DataFrame({
        "Company": [company],
        "TypeName": [type_name],
        "Ram": [ram],
        "Weight": [weight],
        "Touchscreen": [touchscreen],
        "Ips": [ips],
        "ppi": [ppi],
        "Cpu brand": [cpu],
        "HDD": [hdd],
        "SSD": [ssd],
        "Gpu brand": [gpu],
        "os": [os]
    })

    prediction = pipe.predict(query)[0]

    st.success(f"Predicted Price: ₹ {int(np.exp(prediction))}")