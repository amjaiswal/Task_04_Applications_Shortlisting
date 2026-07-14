import joblib
import pandas as pd

# Load model
model = joblib.load("models/model.pkl")

# Example match score
sample = pd.DataFrame({
    "match_score": [82.5]
})

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0]

print("=" * 50)
print("Prediction Result")
print("=" * 50)

print(f"Match Score : {sample.iloc[0]['match_score']}")

if prediction == 1:
    print("Prediction : Good Match")
else:
    print("Prediction : Poor Match")

print(f"Confidence : {max(probability)*100:.2f}%")