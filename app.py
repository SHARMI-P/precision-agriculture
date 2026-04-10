# app.py

import streamlit as st
import pickle, numpy as np, pandas as pd
import shap, matplotlib.pyplot as plt

st.set_page_config(page_title="Precision Agriculture Dashboard", layout="wide")
st.title("Precision Agriculture — Yield Prediction & Fertilizer Optimizer")

model      = pickle.load(open('models/best_model.pkl', 'rb'))
scaler     = pickle.load(open('models/scaler.pkl', 'rb'))
feat_cols  = pickle.load(open('models/feature_cols.pkl', 'rb'))
le_area    = pickle.load(open('models/le_area.pkl', 'rb'))
le_item    = pickle.load(open('models/le_item.pkl', 'rb'))

tab1, tab2, tab3 = st.tabs(["Yield Prediction", "Fertilizer Optimizer", "SHAP Insights"])

with tab1:
    st.header("Predict Crop Yield")
    col1, col2 = st.columns(2)

    with col1:
        crop    = st.selectbox("Crop", le_item.classes_)
        area    = st.selectbox("Region", le_area.classes_)
        year    = st.slider("Year", 1990, 2030, 2023)
        rainfall = st.number_input("Rainfall (mm/year)", 0, 3000, 1000)
        temp    = st.number_input("Avg Temperature (°C)", 0, 50, 25)

    with col2:
        N  = st.number_input("Nitrogen (N)", 0, 140, 50)
        P  = st.number_input("Phosphorus (P)", 0, 145, 50)
        K  = st.number_input("Potassium (K)", 0, 205, 50)
        ph = st.slider("Soil pH", 3.0, 9.0, 6.5)
        humidity = st.slider("Soil Humidity (%)", 0, 100, 70)
        pesticides = st.number_input("Pesticides (tonnes)", 0.0, 500.0, 10.0)

    if st.button("Predict Yield"):
        npk_ratio      = N / (P + K + 1)
        soil_fertility = 0.4*N + 0.3*P + 0.3*K
        climate_index  = rainfall*0.5 + (30 - abs(temp-25))*0.3 + humidity*0.2
        ph_dev         = abs(ph - 7.0)
        decade         = (year // 10) * 10
        area_enc       = le_area.transform([area])[0]
        item_enc       = le_item.transform([crop])[0]

        row = [[year, rainfall, pesticides, temp,
                N, P, K, ph, humidity,
                npk_ratio, soil_fertility,
                climate_index, ph_dev, decade,
                area_enc, item_enc]]

        prediction = model.predict(scaler.transform(row))[0]
        st.success(f"Predicted Yield: **{prediction:,.0f} hg/ha**")
        st.info(f"= approximately {prediction/10000:.2f} tonnes/hectare")

with tab2:
    st.header("Fertilizer Optimizer")
    st.write("Enter current soil conditions. The optimizer will find the N, P, K doses that maximize yield.")

    from src.optimizer import recommend_fertilizer
    crop2   = st.selectbox("Crop", le_item.classes_, key='opt_crop')
    area2   = st.selectbox("Region", le_area.classes_, key='opt_area')
    n2 = st.number_input("Current N", 0, 140, 40, key='cn')
    p2 = st.number_input("Current P", 0, 145, 40, key='cp')
    k2 = st.number_input("Current K", 0, 205, 40, key='ck')

    if st.button("Optimize Fertilizer"):
        result = recommend_fertilizer(
            model, scaler, feat_cols, crop2, area2,
            n2, p2, k2, 1000, 25, 10.0, 6.5, 70
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("Optimal N", f"{result['optimal_N']}", f"{result['delta_N']:+.1f}")
        col2.metric("Optimal P", f"{result['optimal_P']}", f"{result['delta_P']:+.1f}")
        col3.metric("Optimal K", f"{result['optimal_K']}", f"{result['delta_K']:+.1f}")
        st.success(f"Predicted Max Yield: {result['predicted_yield']:,.0f} hg/ha")

with tab3:
    st.header("SHAP Feature Importance")
    st.image("plots/shap_bar.png", caption="Global Feature Importance")
    st.image("plots/shap_beeswarm.png", caption="Feature Impact Direction")