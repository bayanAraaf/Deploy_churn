# ===========================
# streamlit_app.py
# ===========================
import streamlit as st
import pandas as pd
import joblib

# ===========================
# 1. Judul & Deskripsi
# ===========================
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction App")
st.markdown("""
Aplikasi ini memprediksi apakah seorang pelanggan **akan churn atau tidak** berdasarkan fitur tertentu.  
Model terbaik yang digunakan adalah **Random Forest** dengan performa tinggi pada data pelatihan.  
Upload data pelanggan baru atau masukkan data secara manual untuk melihat hasil prediksi.
""")

# ===========================
# 2. Load Model
# ===========================
# ===========================
# 2. Load Model (versi aman untuk deploy)
# ===========================
import os

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "churn_model.pkl")

    if not os.path.exists(model_path):
        st.error("⚠️ Model belum ditemukan. Pastikan file `churn_model.pkl` ada di folder yang sama dengan app.py.")
        st.stop()

    model = joblib.load(model_path)
    return model

model = load_model()
st.success("✅ Model berhasil dimuat.")

# ===========================
# 3. Input Data
# ===========================
st.sidebar.header("🧩 Input Data Pelanggan")

input_mode = st.sidebar.radio("Pilih metode input:", ["Upload CSV", "Input Manual"])

if input_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("📁 Data yang Diupload")
        st.dataframe(data.head())
    else:
        st.warning("Upload file CSV untuk melanjutkan.")
        st.stop()

else:
    st.subheader("✏️ Masukkan Data Secara Manual")

    # TODO: Ganti daftar fitur di bawah sesuai dataset kamu
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.number_input("Tenure (lama berlangganan)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    # Buat dataframe dari input manual
    data = pd.DataFrame({
        "gender": [gender],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Contract": [contract]
    })

# ===========================
# 4. Prediksi
# ===========================
if st.button("🔍 Prediksi Churn"):
    # TODO: Pastikan urutan dan preprocessing fitur sama dengan yang kamu pakai di training
    try:
        prediction = model.predict(data)
        proba = model.predict_proba(data)[:, 1]

        result = "Churn ❌" if prediction[0] == 0 else "Tidak Churn ✅"
        st.success(f"**Hasil Prediksi:** {result}")
        st.metric(label="Probabilitas Churn", value=f"{proba[0]*100:.2f}%")

    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")

# ===========================
# 5. Footer
# ===========================
st.markdown("---")
st.caption("Dibuat untuk Ujian BNSP Associate Data Science — oleh [Nama Kamu]")

