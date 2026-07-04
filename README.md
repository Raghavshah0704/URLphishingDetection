# PhiUSIIL Phishing URL Detector

A complete phishing URL detection project trained from scratch on the
**PhiUSIIL_Phishing_URL_Dataset** — 235,795 labeled URLs with 53 pre-extracted
features (URL-structure + HTML/content-based).

---

## 📁 Project Structure

```
PhishDetect_PhiUSIIL/
├── train.py                # Train all 5 ML models on the dataset
├── app.py                  # Streamlit web UI for interactive detection
├── requirements.txt        # Python dependencies
├── models/                 # Saved model artifacts (auto-created after training)
│   ├── imputer.pkl
│   ├── scaler.pkl
│   ├── feature_columns.json
│   ├── random_forest.pkl
│   ├── extra_trees.pkl
│   ├── gradient_boosting.pkl
│   ├── logistic_regression.pkl
│   └── knn.pkl
└── src/
    ├── __init__.py
    └── predictor.py        # PhishingPredictor class (ensemble inference)
```

---

## 🚀 Quick Start

### 1. Install Dependencies

All required packages come from the existing venv in ProjectII:
```bash
# Already installed in the parent project venv — just activate it:
..\\.venv\\Scripts\\activate
```

Or install fresh:
```bash
pip install -r requirements.txt
```

### 2. Train the Models

```bash
cd PhishDetect_PhiUSIIL
python train.py
```

This will:
- Load `../dataset/PhiUSIIL_Phishing_URL_Dataset.csv`
- Train 5 ML models (RF, ExtraTrees, GradBoost, LogReg, KNN)
- Save all models to `models/`
- Print accuracy, F1, and AUC-ROC for each model
- Save a JSON results summary to `training_results.json`

**Expected training time:** ~5–20 minutes (depends on hardware)

### 3. Run the Web App

```bash
streamlit run app.py
```

---

## 📊 Dataset Details

| Property | Value |
|---|---|
| File | `feature_dataset_clean.csv` |
| Total rows | 20,174 |
| Phishing (1) | 10,087 |
| Legitimate (0) | 10,087 |
| Feature columns | 19 (leaky and zero-variance features dropped) |

### Feature Groups (19 Features)

**URL-based (7 features)**  
domain_length, has_hyphen, url_shortener, suspicious_words, domain_age_category, dns_record, domain_entropy

**HTML/Content-based (12 features)**  
iframe, popup_window, onmouseover, num_forms, external_scripts, external_images, hidden_elements, internal_links, external_links, null_links, external_ratio, script_length

---

## 🤖 Models

| Model | Notes |
|---|---|
| **Random Forest** | 200 trees, max depth 12, min samples leaf 4 |
| **Extra Trees** | 200 trees, max depth 12, min samples leaf 4 |
| **Gradient Boosting** | 150 estimators, learning rate 0.08, max depth 5 |
| **Logistic Regression** | Linear baseline, trained on scaled features |
| **KNN (k=9)** | Instance-based, trained on scaled features |

**Ensemble strategy:** Majority vote over predicted labels; mean probability for confidence score.

---

## 🖥️ Web App Modes

| Mode | Description |
|---|---|
| **🔗 Enter URL** | Enter a URL to automatically fetch its page and extract the 19 features in real-time |
| **📥 JSON / CSV Row** | Paste a pre-extracted JSON dict of the 19 features |
| **🎛️ Manual Sliders** | Interactively adjust the 19 feature values with sliders/inputs |

---

## 📈 Expected Performance (on held-out 20% test set)

To prevent data leakage, 9 high-correlation leakage columns (like `domain_age`, `num_dots`, `has_ip`, etc.) and 8 zero-variance columns were dropped. This yields a highly robust, generalizable model:

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| **Gradient Boosting** | 86.69% | 0.8300 | 0.9356 |
| **Random Forest** | 85.97% | 0.8195 | 0.9269 |
| **KNN (k=9)** | 84.58% | 0.8080 | 0.9126 |
| **Extra Trees** | 82.87% | 0.7898 | 0.9037 |
| **Logistic Regression** | 75.02% | 0.6870 | 0.8332 |

