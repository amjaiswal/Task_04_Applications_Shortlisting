# Task 04 - Applications & Shortlisting (Explainability)

## Overview

This project implements an AI-powered Job Matching System that matches students with suitable jobs using Machine Learning and TF-IDF similarity. It also provides explainable predictions, evaluation metrics, and a REST API for live inference.

---

## Features

- Student-Job Matching
- Explainable AI Recommendations
- Match Score Generation
- TF-IDF Vectorization
- Cosine Similarity Matching
- Random Forest Classification
- Precision, Recall, Accuracy Evaluation
- False Positive Rate Calculation
- Confusion Matrix
- Precision-Recall Curve
- Flask REST API
- Error Handling
- Model Serialization using Joblib

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Matplotlib
- Joblib

---

## Project Structure

```
Task_04_Applications_Shortlisting/
│
├── data/
├── models/
├── outputs/
├── app.py
├── train.py
├── predict.py
├── explain.py
├── utils.py
├── Task_04.ipynb
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Train Model

```bash
python train.py
```

---

## Predict

```bash
python predict.py
```

---

## Run API

```bash
python app.py
```

---

## API Endpoints

### Health Check

```
GET /health
```

### Prediction

```
POST /predict
```

Example JSON

```json
{
    "match_score": 85
}
```

Response

```json
{
    "match_score": 85,
    "prediction": "Good Match",
    "confidence": 100.0
}
```

---

## Outputs

- predictions.csv
- metrics.txt
- confusion_matrix.png
- precision_recall_curve.png
- explanation_examples.json

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- False Positive Rate

---

## Author

Amar Jaiswal