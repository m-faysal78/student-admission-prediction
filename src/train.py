import os
import joblib
from sklearn.linear_model import LinearRegression
from preprocess import load_data, preprocess_data

DATA_PATH = "../data/admission_data.csv"

df = load_data(DATA_PATH)

X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

model = LinearRegression()
model.fit(X_train, y_train)

os.makedirs("../model", exist_ok=True)

joblib.dump(model, "../model/admission_model.pkl")
joblib.dump(scaler, "../model/scaler.pkl")

print("Training completed successfully")
