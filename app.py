import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="AUTO.RIA Price Prediction", page_icon="🚗")

st.title("🚗 Used Car Price Prediction")
st.markdown("Predict used car price on the Ukrainian market based on auto.ria.com data.")

# ── load model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('models/lightgbm.pkl')

model = load_model()

# ── inputs ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    release_year = st.slider("Release year", 2000, 2024, 2018)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000,
                               value=100000, step=5000)
    engine_volume = st.number_input("Engine volume (L)", min_value=0.5,
                                     max_value=6.0, value=2.0, step=0.1)

with col2:
    fuel_type = st.selectbox("Fuel type",
                              ["Бензин", "Дизель", "Електро", "Гібрид", "Газ/Бензин"])
    transmission = st.selectbox("Transmission",
                                  ["Автомат", "Механічна", "Варіатор", "Робот"])
    road_accident = st.selectbox("Accident history",
                                   ["No accident", "Was in accident"])

# ── predict ───────────────────────────────────────────────────────────
if st.button("Predict price 💰", type="primary"):
    # будуємо рядок фіч так само як під час тренування
    input_df = pd.DataFrame([{
        'release_year': release_year,
        'mileage': mileage,
        'engine_volume': engine_volume,
        'road_accident': 1 if road_accident == "Was in accident" else 0,
        'is_electric': 1 if fuel_type == "Електро" else 0,
    }])

    # OHE фічі — заповнюємо нулями, виставляємо потрібні
    ohe_cols = [c for c in model.feature_name_ if c not in input_df.columns]
    for col in ohe_cols:
        input_df[col] = 0.0

    fuel_col = f'fuel_type_{fuel_type}'
    trans_col = f'transmission_{transmission}'
    if fuel_col in input_df.columns:
        input_df[fuel_col] = 1.0
    if trans_col in input_df.columns:
        input_df[trans_col] = 1.0

    input_df = input_df[model.feature_name_]

    pred_log = model.predict(input_df)[0]
    pred_price = np.expm1(pred_log)

    st.success(f"### Estimated price: **${pred_price:,.0f}**")
    st.caption("Model: LightGBM | R² = 0.860 | MAE = $4,694")