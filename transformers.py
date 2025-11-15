from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class TelcoCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, expected_features):
        self.expected_features = expected_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Hapus kolom yang tidak relevan
        if 'customerID' in X.columns:
            X = X.drop(columns=['customerID'])

        # Konversi TotalCharges ke numerik dan isi NaN
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
            X['TotalCharges'] = X['TotalCharges'].fillna(X['TotalCharges'].median())

        # Ubah Churn ke angka (jika ada)
        if 'Churn' in X.columns:
            X['Churn'] = X['Churn'].map({'Yes': 1, 'No': 0})

        # Tambahkan kolom yang hilang dengan nilai default
        for col in self.expected_features:
            if col not in X.columns:
                X[col] = pd.NA

        return X[self.expected_features]