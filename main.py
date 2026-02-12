import joblib
import numpy as np

model = joblib.load("models/fraud_model.pkl")

def predict_transaction(transaction_data, threshold=0.7):
    prob = model.predict_proba(transaction_data)[0][1]
    if prob >= threshold:
        return "Fraud 🚨", prob
    else:
        return "Legitimate ✅", prob

# Example usage
sample_txn = np.random.rand(1, 31)
result, probability = predict_transaction(sample_txn)

print("Prediction:", result)
print("Fraud Probability:", probability)
