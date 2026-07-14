from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("models/model.pkl")


@app.route("/")
def home():
    return jsonify({
        "message": "Task 4 - AI/ML Job Matching API is Running"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "Healthy",
        "model_loaded": True
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data received"
            }), 400

        if "match_score" not in data:
            return jsonify({
                "error": "match_score field is required"
            }), 400

        score = float(data["match_score"])

        sample = pd.DataFrame({
            "match_score": [score]
        })

        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0]

        result = "Good Match" if prediction == 1 else "Poor Match"

        return jsonify({
            "match_score": score,
            "prediction": result,
            "confidence": round(max(probability) * 100, 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)