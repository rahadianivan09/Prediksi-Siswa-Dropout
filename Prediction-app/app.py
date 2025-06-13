
import streamlit as st
import joblib
import numpy as np

# Load pipeline
pipeline = joblib.load("dropout_prediction_pipeline.pkl")
model = pipeline['model']
scaler = pipeline['scaler']
features = pipeline['features']

st.title("Prediksi Dropout Siswa")

input_data = []

st.header("Masukkan Informasi Siswa")
for feat in features:
    if isinstance(feat, str) and ('_' in feat or feat.endswith('yes') or feat.endswith('no')):
        val = st.selectbox(f"{feat}", ["Tidak", "Ya"])
        input_data.append(1 if val == "Ya" else 0)
    else:
        val = st.number_input(f"{feat}", step=1.0)
        input_data.append(val)

if st.button("Prediksi Dropout"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("❌ Siswa Berpotensi Dropout")
    else:
        st.success("✅ Siswa Tidak Dropout")
