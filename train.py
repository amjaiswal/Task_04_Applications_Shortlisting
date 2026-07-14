import os
import joblib
import pandas as pd

from explain import generate_explanation

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# ---------------------------------------------------
# Create folders
# ---------------------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------
# Load datasets
# ---------------------------------------------------
students = pd.read_csv("data/students.csv")
jobs = pd.read_csv("data/jobs.csv")

# ---------------------------------------------------
# TF-IDF Vectorizer
# ---------------------------------------------------
vectorizer = TfidfVectorizer()

all_text = pd.concat(
    [students["skills"], jobs["required_skills"]],
    ignore_index=True
)

vectorizer.fit(all_text)

student_vectors = vectorizer.transform(students["skills"])
job_vectors = vectorizer.transform(jobs["required_skills"])

predictions = []

# ---------------------------------------------------
# Student-Job Matching
# ---------------------------------------------------
for i, student in students.iterrows():

    similarities = cosine_similarity(
        student_vectors[i],
        job_vectors
    )[0]

    best_index = similarities.argmax()

    best_job = jobs.iloc[best_index]

    score = round(similarities[best_index] * 100, 2)

    explanation = generate_explanation(
        student_name=student["name"],
        student_skills=student["skills"],
        job_title=best_job["job_title"],
        required_skills=best_job["required_skills"],
        score=score
    )

    predictions.append({
        "student_id": student["student_id"],
        "student_name": student["name"],
        "job_id": best_job["job_id"],
        "job_title": best_job["job_title"],
        "match_score": score,
        "explanation": explanation["reason"]
    })

# ---------------------------------------------------
# Save Predictions
# ---------------------------------------------------
pred_df = pd.DataFrame(predictions)

pred_df.to_csv(
    "outputs/predictions.csv",
    index=False
)

# ---------------------------------------------------
# Prepare Training Data
# ---------------------------------------------------
X = pred_df[["match_score"]]
y = (pred_df["match_score"] >= 60).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# Train Model
# ---------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------
accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, zero_division=0)
recall = recall_score(y_test, pred, zero_division=0)

cm = confusion_matrix(
    y_test,
    pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) != 0
    else 0
)

# ---------------------------------------------------
# Save Metrics
# ---------------------------------------------------
with open("outputs/metrics.txt", "w") as file:
    file.write("Model Evaluation Metrics\n")
    file.write("========================\n\n")
    file.write(f"Accuracy            : {accuracy:.4f}\n")
    file.write(f"Precision           : {precision:.4f}\n")
    file.write(f"Recall              : {recall:.4f}\n")
    file.write(f"False Positive Rate : {false_positive_rate:.4f}\n")

# ---------------------------------------------------
# Save Model
# ---------------------------------------------------
joblib.dump(
    model,
    "models/model.pkl"
)

joblib.dump(
    vectorizer,
    "models/vectorizer.pkl"
)

# ---------------------------------------------------
# Summary
# ---------------------------------------------------
print("\n===================================")
print(" Training Completed Successfully")
print("===================================")

print(f"Students Processed : {len(students)}")
print(f"Jobs Available     : {len(jobs)}")
print(f"Predictions Saved  : outputs/predictions.csv")
print(f"Metrics Saved      : outputs/metrics.txt")
print(f"Model Saved        : models/model.pkl")
print(f"Vectorizer Saved   : models/vectorizer.pkl")

print("\nTop 10 Matches")
print(pred_df.head(10))