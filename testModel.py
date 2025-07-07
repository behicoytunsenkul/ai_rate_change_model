import numpy as np
import joblib

bestModel = joblib.load('bestModel.pkl')
scaler = joblib.load('scaler.pkl')

def predict(oldValue,newValue):
    residual_value = newValue - oldValue
    rate = newValue / oldValue
    data = np.array([[oldValue,newValue,residual_value,rate]])
    scaled_input = scaler.transform(data)
    return bestModel.predict(scaled_input)[0]

sample_predict = predict(0.023432,0.232442)
print(f"Predicted rate is: {sample_predict:.6f}%")
