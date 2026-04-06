# 🛡️ ScholarShield — Fake Scholarship Detection System

**Student:** Divya | **USN:** U18IW23S0016  
**Project:** Fake Scholarship Detection using Machine Learning

---

## Overview

ScholarShield is a complete web application that uses **4 Machine Learning models** running simultaneously to detect fraudulent scholarship announcements. A majority-vote ensemble produces the final verdict.

### ML Models Used
| Model | Algorithm | Library |
|-------|-----------|---------|
| Naive Bayes | MultinomialNB (α=0.3) | scikit-learn |
| Logistic Regression | LogisticRegression (C=1.5) | scikit-learn |
| Decision Tree | DecisionTreeClassifier (depth=15) | scikit-learn |
| SVM | LinearSVC (C=1.2) | scikit-learn |

### Feature Engineering
- **Method:** TF-IDF (Term Frequency–Inverse Document Frequency)
- **N-grams:** Unigrams + Bigrams
- **Max Features:** 5,000
- **Input Fields:** Name + Provider + Description + URL + Email + Fee + Amount

### Dataset
- **File:** `dataset/scholarship_dataset.csv`
- **Total Records:** 101
- **REAL:** 55 scholarships (government portals: NSP, AICTE, UGC, DST, etc.)
- **FAKE:** 46 fraudulent schemes (fee demands, guaranteed approval, suspicious domains)
- **Split:** 80% training / 20% testing

---

## Project Structure

```
ScholarShield/
├── app.py                    ← Flask web server (main entry point)
├── train_models.py           ← Train all 4 ML models & save
├── requirements.txt          ← Python dependencies
├── README.md
├── dataset/
│   └── scholarship_dataset.csv   ← Labeled dataset (101 records)
├── models/                   ← Saved model files (created after training)
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── svm.pkl
│   └── meta.json             ← Accuracy, F1, CV scores
├── templates/
│   └── index.html            ← Frontend (HTML/CSS/JS)
└── static/
    └── img/
        └── ml_results.png    ← Confusion matrices + performance charts
```

---

## Setup & Run

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the Models
```bash
python train_models.py
```
This will:
- Load the dataset from `dataset/scholarship_dataset.csv`
- Train all 4 ML models
- Save them as `.pkl` files in `models/`
- Generate the performance chart at `static/img/ml_results.png`

### Step 3 — Run the Web App
```bash
python app.py
```

### Step 4 — Open in Browser
```
http://127.0.0.1:5000
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web application |
| `/predict` | POST | Run all 4 models, returns JSON |
| `/models` | GET | Training metadata and accuracy |
| `/dataset` | GET | Dataset summary and samples |

### Example `/predict` Request
```json
POST /predict
{
  "scholarship_name": "National Scholarship Portal 2025",
  "provider": "Ministry of Education, Govt. of India",
  "description": "Government scholarship for SC/ST students...",
  "url": "https://scholarships.gov.in",
  "contact_email": "helpdesk@nsp.gov.in",
  "application_fee": "None",
  "amount": "10000 to 50000 per year"
}
```

### Example Response
```json
{
  "success": true,
  "result": {
    "ensemble_label": "REAL",
    "ensemble_conf": 100.0,
    "votes_fake": 0,
    "votes_real": 4,
    "predictions": {
      "naive_bayes":         {"label": "REAL", "confidence": 97.3, ...},
      "logistic_regression": {"label": "REAL", "confidence": 98.1, ...},
      "decision_tree":       {"label": "REAL", "confidence": 91.5, ...},
      "svm":                 {"label": "REAL", "confidence": 95.2, ...}
    },
    "flags": [
      {"type": "safe", "text": "Official .gov.in domain detected"},
      {"type": "safe", "text": "No application fee"}
    ]
  }
}
```

---

## Software Requirements

- **OS:** Windows 10/11 or Linux
- **Python:** 3.9+
- **Key Libraries:** Flask, scikit-learn, pandas, numpy, matplotlib, seaborn, joblib
- **Browser:** Any modern browser (Chrome, Firefox, Edge)

---

*ScholarShield — Protecting students from scholarship fraud using Machine Learning*
