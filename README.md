# 📊 Customer Churn Prediction App

This Streamlit web app predicts whether a customer will **churn (leave)** or **stay subscribed** based on their profile and service usage data.  
The model uses a **Random Forest pipeline** trained on historical customer churn data, complete with preprocessing for both categorical and numerical features.

---

## 🚀 Live Demo
👉 [Streamlit App](https://deploychurn-ha4t6ppxzct3ujvz45w8t5.streamlit.app/)  


---

## 📦 Features
- Predicts customer churn in real-time from user input or uploaded CSV  
- Handles categorical and numerical features automatically via preprocessing pipeline  
- Clean, professional Streamlit interface  
- Displays prediction result and churn probability  
- Model and preprocessing pipeline loaded from `.pkl` file  

---

## 🛠️ Tech Stack
- Python  
- Streamlit  
- scikit-learn  
- joblib  
- pandas  
- numpy  

---

## 📁 Project Structure

📦 customer-churn-app/
├── streamlit_app.py
├── model_rf.pkl
├── requirements.txt
└── README.md


## ▶️ How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/bayanAraaf/Deploy_churn
   cd customer-churn-app

2. Install dependencies:
   pip install -r requirements.txt

3. Run the app:
   streamlit run app.py


📚 Model Info

The model was trained using:
Random Forest Classifier — selected for best accuracy on test data
Pipeline Preprocessing — includes scaling, encoding, and feature selection

Feature Engineering Includes:
Splitting categorical and numerical columns
Applying one-hot encoding for categorical data
Standard scaling for numerical features
Combining all steps using ColumnTransformer and Pipeline

📊 Dataset Source
This model was trained on a Customer Churn Dataset containing customer subscription records (e.g., demographics, contract type, and monthly charges).
The dataset aims to predict whether a customer will discontinue the service (Churn = Yes) or remain loyal (Churn = No).

⚠️ Notes
Ensure model_rf.pkl (the saved pipeline) is in the same folder as streamlit_app.py
Use full pipeline export (joblib.dump(pipeline, "model_rf.pkl")) to maintain consistent preprocessing
The app supports both manual input and CSV upload for prediction

✍️ Author Bayan
📄 BNSP Associate Data Scientist Candidate
🔗 LinkedIn http://bit.ly/4pt1XFe
💡 Feel free to fork, star, or contribute!
