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

with st.form("student_input_form"):
    st.markdown("### 🔢 Input Data Siswa")

    for feat in features:
        if feat in numerical_inputs:
            min_val, max_val, default = numerical_inputs[feat]
            val = st.slider(f"{feat}", min_value=min_val, max_value=max_val, value=default)
            input_data.append(val)
        else:
            val = st.checkbox(f"{feat.replace('_', ' ').capitalize()}", value=False)
            input_data.append(1 if val else 0)

    submitted = st.form_submit_button("🔮 Prediksi Dropout")

# Prediksi setelah submit
if submitted:
    try:
        input_array = np.array(input_data).reshape(1, -1)
        
        # Debug info (opsional)
        st.caption(f"🔎 Jumlah fitur input: {len(input_data)} dari {len(features)}")
        st.caption(f"Fitur input: {features}")

        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.error("❌ Prediksi: Siswa **berpotensi Dropout**.")
        else:
            st.success("✅ Prediksi: Siswa **kemungkinan LULUS**.")
    
    except Exception as e:
        st.error(f"🚨 Terjadi kesalahan saat prediksi:\n\n{e}")
