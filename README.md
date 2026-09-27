# 🛡️ PhiUSIIL Phishing URL Detector

A complete, end-to-end Machine Learning phishing URL detection system trained on the **PhiUSIIL Phishing URL Dataset**. The project features real-time URL & content feature extraction, a 5-model ensemble classifier, an interactive Streamlit Web Dashboard, a Flask REST API backend, and a Google Chrome Browser Extension (Manifest V3).

---

## ✨ Features

- **🤖 5 ML Models Ensemble:** Combines Random Forest, Extra Trees, Gradient Boosting, Logistic Regression, and K-Nearest Neighbors (KNN) with soft-voting / majority consensus.
- **⚡ Real-Time Feature Extraction:** Automatically scrapes and extracts 19 high-impact URL structural and HTML content features live from target websites.
- **📊 Interactive Streamlit Web Dashboard:** Test URLs in real time, view detailed model predictions, breakdown features, inspect HTML elements, or manually adjust feature sliders.
- **🔌 Flask REST API Server:** Provides a high-performance REST endpoint (`/api/predict`) with CORS enabled for seamless integration.
- **🧩 Browser Extension:** Chrome/Edge Manifest V3 extension for instant 1-click phishing checks while browsing.
- **🚀 One-Click Launchers:** Windows Batch (`.bat`) and PowerShell (`.ps1`) launcher scripts for quick startup.

---

## 📁 Project Structure

```
phishing-detection/
├── train.py                  # Model training script & performance evaluation
├── app.py                    # Streamlit interactive web dashboard
├── api.py                    # Flask REST API server for external/extension requests
├── requirements.txt          # Python project dependencies
├── training_results.json     # Saved metrics summary (Accuracy, F1, AUC-ROC)
├── run.bat                   # 1-Click launcher for Streamlit Dashboard (Cmd)
├── run.ps1                   # 1-Click launcher for Streamlit Dashboard (PowerShell)
├── run_api.bat               # 1-Click launcher for REST API Server (Cmd)
├── run_api.ps1               # 1-Click launcher for REST API Server (PowerShell)
├── dataset/                  # Dataset directory
│   └── feature_dataset_clean.csv # Preprocessed PhiUSIIL dataset (20,174 rows)
├── models/                   # Saved ML model artifacts & preprocessors
│   ├── feature_columns.json
│   ├── imputer.pkl
│   ├── scaler.pkl
│   ├── random_forest.pkl
│   ├── extra_trees.pkl
│   ├── gradient_boosting.pkl
│   ├── logistic_regression.pkl
│   └── knn.pkl
├── extension/                # Chrome Browser Extension (Manifest V3)
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.css
│   ├── popup.js
│   └── icons/
└── src/
    ├── __init__.py
    ├── predictor.py          # PhishingPredictor inference engine
    └── url_feature_extractor.py # Live URL & HTML feature extraction pipeline
```

---

## 🚀 Quick Start

### 1. Installation

Clone or open the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train Models (Optional)

Pre-trained models are already saved in the `models/` directory. To retrain all 5 classifiers:
```bash
python train.py
```
*Trains Random Forest, Extra Trees, Gradient Boosting, Logistic Regression, and KNN, saving performance statistics to `training_results.json`.*

---

## 🖥️ Running the Applications

### Option A: Streamlit Web Dashboard

**Command Line:**
```bash
streamlit run app.py
```

**Windows Quick Launch:**
- Double-click `run.bat` or execute `.\run.ps1` in PowerShell.

*Access the dashboard in your web browser at `http://localhost:8501`.*

---

### Option B: Flask REST API Server

**Command Line:**
```bash
python api.py
```

**Windows Quick Launch:**
- Double-click `run_api.bat` or execute `.\run_api.ps1` in PowerShell.

*The API server will run on `http://localhost:5000`.*

---

### Option C: Chrome Browser Extension

1. Start the Flask REST API server (`python api.py` or run `run_api.bat`).
2. Open Google Chrome (or Microsoft Edge) and navigate to `chrome://extensions/`.
3. Enable **Developer mode** in the top-right corner.
4. Click **Load unpacked** and select the `extension/` directory.
5. Click the extension icon in your browser toolbar to analyze any active tab or entered URL!

---

## 📊 Dataset & Feature Extraction

The dataset consists of **20,174 balanced URLs** (10,087 phishing / 10,087 legitimate). High-correlation data leakage columns and zero-variance features were dropped to ensure realistic, highly generalizable model performance.

### 19 Extracted Feature Breakdown

| Feature Category | Features |
|---|---|
| **URL Structural (7)** | `domain_length`, `has_hyphen`, `url_shortener`, `suspicious_words`, `domain_age_category`, `dns_record`, `domain_entropy` |
| **HTML & Content (12)** | `iframe`, `popup_window`, `onmouseover`, `num_forms`, `external_scripts`, `external_images`, `hidden_elements`, `internal_links`, `external_links`, `null_links`, `external_ratio`, `script_length` |

---

## 📈 Model Performance Benchmark

Models were trained and evaluated on a held-out 20% test split:

| Model | Accuracy | F1 Score | AUC-ROC | Train Time |
|---|:---:|:---:|:---:|:---:|
| 🥇 **Gradient Boosting** | **86.69%** | **0.8300** | **0.9356** | ~2.89s |
| 🥈 **Random Forest** | **85.97%** | **0.8195** | **0.9269** | ~0.49s |
| 🥉 **KNN (k=9)** | **84.56%** | **0.8078** | **0.9123** | < 0.01s |
| 🏅 **Extra Trees** | **82.87%** | **0.7898** | **0.9037** | ~0.30s |
| 🎖️ **Logistic Regression** | **75.02%** | **0.6870** | **0.8332** | ~0.08s |

*Ensemble decision is computed via majority voting across models with confidence calculated from mean probability scores.*

---

## 📡 REST API Reference

### Health Check
`GET /`
```json
{
  "service": "PhiUSIIL Phishing URL Detector API",
  "status": "online",
  "version": "1.0.0"
}
```

### Predict Phishing Probability
`POST /api/predict` or `GET /api/predict?url=https://example.com`

**Request Body (JSON):**
```json
{
  "url": "https://example.com"
}
```

**Response (JSON):**
```json
{
  "url": "https://example.com",
  "prediction": "Legitimate",
  "confidence": 0.945,
  "ensemble_votes": {
    "Legitimate": 5,
    "Phishing": 0
  },
  "per_model": {
    "random_forest": {"label": "Legitimate", "prob": 0.96},
    "gradient_boosting": {"label": "Legitimate", "prob": 0.98},
    "knn": {"label": "Legitimate", "prob": 0.92},
    "extra_trees": {"label": "Legitimate", "prob": 0.93},
    "logistic_regression": {"label": "Legitimate", "prob": 0.93}
  }
}
```

---

## 🛡️ License & Acknowledgments

Trained on the **PhiUSIIL Phishing URL Dataset** provided by security researchers. Built with Python, Scikit-Learn, Streamlit, Flask, and BeautifulSoup4.
