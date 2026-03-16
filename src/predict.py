import joblib
import numpy as np

model = joblib.load("../model/admission_model.pkl")
scaler = joblib.load("../model/scaler.pkl")

sample = np.array([[320, 110, 4.0, 4.5, 4.0, 1]])

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

print("Predicted admission probability:", prediction[0])
