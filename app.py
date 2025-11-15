# ===========================
# streamlit_app.py
# ===========================
import streamlit as st
import pandas as pd
import joblib
from transformers import TelcoCleaner  # pastikan file transformers.py ada di folder yang sama
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
st.markdown('Untuk Yang Mau Upload CSV Wajib Memliki Columns Yang Sama Seperti Input Manual')

# ===========================
# 2. Load Model (versi aman untuk deploy)
# ===========================
import os

@st.cache_resource
def load_models():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "churn_model.pkl")

    if not os.path.exists(model_path):
        st.error("⚠️ Model belum ditemukan. Pastikan file `churn_model.pkl` ada di folder yang sama dengan app.py.")
        st.stop()

    models = joblib.load(model_path)
    return models

models = load_models()
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

if input_mode == "Upload CSV" and data.isnull().any().any():
    st.warning("⚠️ Data mengandung nilai kosong. Pastikan data sudah bersih atau gunakan template yang sesuai.")

    # Semua fitur yang dibutuhkan model
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    tenure = st.number_input("Tenure (lama berlangganan)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)

    # Bentuk dataframe sesuai fitur training
    data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

# ===========================
# 4. Prediksi
# ===========================
st.sidebar.header("Pilih Model")
model_name = st.sidebar.selectbox("Pilih model:", ["Random Forest (rm)", "Logistic Regression (logreg)"])

if model_name.startswith("Random"):
    model = models["rm"]
else:
    model = models["logreg"]

if st.button("🔍 Prediksi Churn"):
    try:
        prediction = model.predict(data)
        proba = model.predict_proba(data)[:, 1]

        result = "Churn ❌" if prediction[0] == 1 else "Tidak Churn ✅"
        st.success(f"**Hasil Prediksi:** {result}")
        st.metric(label="Probabilitas Churn", value=f"{proba[0]*100:.2f}%")

    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")


# ===========================
# 5. Footer
# ===========================
st.markdown("---")
st.caption("Dibuat untuk Ujian BNSP Associate Data Science — oleh Bayan Araaf")

