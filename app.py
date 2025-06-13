import streamlit as st
import joblib
import numpy as np

# Load pipeline
pipeline = joblib.load("dropout_prediction_pipeline.pkl")
model = pipeline['model']
scaler = pipeline['scaler']
features = pipeline['features']

st.title("📘 Prediksi Dropout Siswa")
st.subheader("Masukkan Informasi Siswa")

# Mapping fitur numerik dan default nilai berdasarkan statistik deskriptif
numerical_defaults = {
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
    "G2": (0, 19, 11),
    "avg_grade": (0, 19, 11)
}

input_data = []

# Input fitur numerik
with st.expander("🔢 Fitur Numerik"):
    for feat in features:
        if feat in numerical_defaults:
            min_val, max_val, default = numerical_defaults[feat]
            val = st.slider(f"{feat}", min_val, max_val, default)
            input_data.append(val)

# Input fitur kategorikal/dummy
with st.expander("✅ Fitur Kategorikal"):
    for feat in features:
        if feat not in numerical_defaults:
            val = st.checkbox(f"{feat.replace('_', ' ').title()}", value=False)
            input_data.append(1 if val else 0)

# Tombol prediksi
if st.button("🔮 Prediksi Dropout"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("❌ Prediksi: Siswa **berpotensi Dropout**.")
    else:
        st.success("✅ Prediksi: Siswa **kemungkinan LULUS**.")

# Validasi jumlah input
st.caption(f"Jumlah fitur input: {len(input_data)} dari {len(features)}")
