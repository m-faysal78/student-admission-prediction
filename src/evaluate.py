import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocess import load_data, preprocess_data

df = load_data("../data/admission_data.csv")

X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

model = joblib.load("../model/admission_model.pkl")

preds = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, preds))
print("MAE:", mean_absolute_error(y_test, preds))
print("R2:", r2_score(y_test, preds))
