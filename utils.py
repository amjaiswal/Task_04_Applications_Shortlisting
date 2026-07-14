import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
)

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Load data
predictions = pd.read_csv("outputs/predictions.csv")
model = joblib.load("models/model.pkl")

# Ground truth and predictions
y_true = (predictions["match_score"] >= 60).astype(int)
y_pred = model.predict(predictions[["match_score"]])

# --------------------------
# Confusion Matrix
# --------------------------
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Poor Match", "Good Match"]
)

disp.plot()

plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# --------------------------
# Precision Recall Curve
# --------------------------
PrecisionRecallDisplay.from_predictions(
    y_true,
    model.predict_proba(predictions[["match_score"]])[:, 1]
)

plt.title("Precision Recall Curve")
plt.savefig("outputs/precision_recall_curve.png")
plt.close()

# --------------------------
# Explanation JSON
# --------------------------
examples = predictions.head(10).to_dict(orient="records")

with open(
    "outputs/explanation_examples.json",
    "w"
) as f:
    json.dump(examples, f, indent=4)

print("Outputs generated successfully.")