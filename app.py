import streamlit as st
import joblib
import numpy as np

# Load pipeline
pipeline = joblib.load("dropout_prediction_pipeline.pkl")
model = pipeline['model']
scaler = pipeline['scaler']
features = pipeline['features']

# Pastikan avg_grade tidak diminta dari user karena sudah dihitung otomatis sebelumnya
features = [f for f in features if f != 'avg_grade']

st.title("📘 Prediksi Dropout Siswa")
st.subheader("Masukkan Informasi Siswa")

# Fitur numerik berdasarkan statistik deskriptif
numerical_inputs = {
    "age": (15, 22, 17),
    "Medu": (0, 4, 2),
    "Fedu": (0, 4, 2),
    "traveltime": (1, 4, 1),
    "studytime": (1, 4, 2),
    "failures": (0, 3, 0),
    "famrel": (1, 5, 4),
    "freetime": (1, 5, 3),
    "goout": (1, 5, 3),
    "Dalc": (1, 5, 1),
    "Walc": (1, 5, 2),
    "health": (1, 5, 4),
    "absences": (0, 32, 3),
    "G1": (0, 19, 11),
    "G2": (0, 19, 11)
}

input_data = []

with st.expander("🔢 Fitur Numerik"):
    for feat in numerical_inputs:
        if feat in features:
            min_val, max_val, default = numerical_inputs[feat]
            val = st.slider(f"{feat}", min_val, max_val, default)
            input_data.append(val)

with st.expander("✅ Fitur Kategorikal"):
    for feat in features:
        if feat not in numerical_inputs:
            val = st.checkbox(f"{feat.replace('_', ' ')}", value=False)
            input_data.append(1 if val else 0)

# Validasi jumlah fitur
st.write(f"Jumlah input: {len(input_data)} / {len(features)} fitur yang diminta")

# Tombol prediksi
if st.button("🔮 Prediksi Dropout"):
    input_array = np.array(input_data).reshape(1, -1)

    # Transformasi fitur numerik
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.error("❌ Prediksi: Siswa **berpotensi Dropout**.")
    else:
        st.success("✅ Prediksi: Siswa **kemungkinan LULUS**.")
