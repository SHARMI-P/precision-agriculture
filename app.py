# app.py
import streamlit as st
import pickle, numpy as np, pandas as pd
import shap, matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="Precision Agriculture Dashboard", layout="wide")
st.title("Precision Agriculture — Yield Prediction & Fertilizer Optimizer")

# ── Load yield models ─────────────────────────────────────────────────────────
model      = pickle.load(open('models/best_model.pkl', 'rb'))
scaler     = pickle.load(open('models/scaler.pkl', 'rb'))
feat_cols  = pickle.load(open('models/feature_cols.pkl', 'rb'))
le_area    = pickle.load(open('models/le_area.pkl', 'rb'))
le_item    = pickle.load(open('models/le_item.pkl', 'rb'))

# ── Load CNN model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_cnn():
    cnn = tf.keras.models.load_model("models/cnn_disease_model.h5")
    with open("models/class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    return cnn, class_names

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Disease Detection",
    "🌾 Yield Prediction",
    "🧪 Fertilizer Optimizer",
    "📊 SHAP Insights"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CNN Disease Detection
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Plant Disease Detection")
    st.write("Upload a leaf image and the CNN model will identify the disease.")

    cnn_model, class_names = load_cnn()
    st.success("✅ CNN Model loaded — 93.55% accuracy | 72 disease classes")

    uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

        with col2:
            with st.spinner("Analyzing..."):
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                preds = cnn_model.predict(img_array)
                top3_idx = np.argsort(preds[0])[::-1][:3]
                top_class = class_names[top3_idx[0]]
                top_conf  = preds[0][top3_idx[0]] * 100

                parts   = top_class.split("___")
                plant   = parts[0].replace("_", " ")
                disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

            st.markdown("### Result")
            if "healthy" in disease.lower():
                st.success(f"✅ **{plant}** — Healthy")
            else:
                st.error(f"⚠️ **{plant}** — {disease}")

            st.metric("Confidence", f"{top_conf:.2f}%")

            st.markdown("### Top 3 Predictions")
            for i, idx in enumerate(top3_idx):
                name = class_names[idx].replace("___", " → ").replace("_", " ")
                conf = preds[0][idx] * 100
                st.progress(int(conf), text=f"{i+1}. {name}: {conf:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Yield Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Predict Crop Yield")
    col1, col2 = st.columns(2)
    with col1:
        crop     = st.selectbox("Crop", le_item.classes_)
        area     = st.selectbox("Region", le_area.classes_)
        year     = st.slider("Year", 1990, 2030, 2023)
        rainfall = st.number_input("Rainfall (mm/year)", 0, 3000, 1000)
        temp     = st.number_input("Avg Temperature (°C)", 0, 50, 25)
    with col2:
        N          = st.number_input("Nitrogen (N)", 0, 140, 50)
        P          = st.number_input("Phosphorus (P)", 0, 145, 50)
        K          = st.number_input("Potassium (K)", 0, 205, 50)
        ph         = st.slider("Soil pH", 3.0, 9.0, 6.5)
        humidity   = st.slider("Soil Humidity (%)", 0, 100, 70)
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
        st.info(f"= approximately {prediction/10000:.4f} tonnes/hectare")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Fertilizer Optimizer
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Fertilizer Optimizer")
    st.write("Enter current soil conditions. The optimizer will find the N, P, K doses that maximize yield.")
    from src.optimizer import recommend_fertilizer
    crop2  = st.selectbox("Crop", le_item.classes_, key='opt_crop')
    area2  = st.selectbox("Region", le_area.classes_, key='opt_area')
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SHAP Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("SHAP Feature Importance")
    if os.path.exists("plots/shap_bar.png"):
        st.image("plots/shap_bar.png", caption="Global Feature Importance")
    else:
        st.info("SHAP plots not yet generated. Run `python main.py` to generate them.")
    if os.path.exists("plots/shap_beeswarm.png"):
        st.image("plots/shap_beeswarm.png", caption="Feature Impact Direction")