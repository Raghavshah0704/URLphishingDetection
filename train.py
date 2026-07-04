# -*- coding: utf-8 -*-
"""
PhiUSIIL Phishing URL Detection - Training Script
==================================================
Trains multiple ML models on feature_dataset_clean.csv (20k+ URLs).
Dataset already has pre-extracted features -- no URL scraping needed.

Features used: 36 numeric features (URL-structure + HTML/content-based)
Label: 0 = Legitimate, 1 = Phishing
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Fix Windows console encoding for emoji output
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, f1_score
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATH = Path(__file__).parent / "dataset" / "feature_dataset_clean.csv"
MODELS_DIR   = Path(__file__).parent / "models"
RESULTS_FILE = Path(__file__).parent / "training_results.json"

# Drop high-correlation features (|corr| > 0.5 with label) to avoid leakage/overfitting
DROP_COLS = [
    "num_dots",            # corr=0.83 — near-perfect with num_subdomains
    "num_subdomains",      # corr=0.83 — near-perfect with num_dots
    "num_digits",          # corr=0.79
    "domain_age",          # corr=0.77 — highly predictive, potential leakage
    "has_ip",              # corr=0.75
    "special_char_count",  # corr=0.74
    "directory_depth",     # corr=0.70
    "tld_length",          # corr=0.58
    "url_length",          # corr=0.50
]

# Drop zero-variance / NaN-correlation columns (all same value → no signal)
ZERO_VARIANCE_COLS = [
    "has_at",
    "double_slash_redirect",
    "https_token",
    "domain_recently_registered",
    "domain_suspicious_tld",
    "right_click_disabled",
    "ssl_valid",
    "invalid_domain_chars",
]

TARGET_COL = "label"

RANDOM_STATE = 42
TEST_SIZE    = 0.20

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  PhiUSIIL Phishing URL Detector -- Training")
print("=" * 60)
print(f"\n[*] Loading dataset from:\n    {DATASET_PATH}\n")

df = pd.read_csv(DATASET_PATH)
print(f"[+] Loaded {len(df):,} rows x {len(df.columns)} columns")
print(f"    Phishing (1): {(df[TARGET_COL]==1).sum():,}")
print(f"    Legit    (0): {(df[TARGET_COL]==0).sum():,}")

# Drop metadata/text columns; keep only numeric feature columns
all_drop = DROP_COLS + ZERO_VARIANCE_COLS + [TARGET_COL]
feature_cols = [c for c in df.columns if c not in all_drop]
X = df[feature_cols].copy()
y = df[TARGET_COL].copy()

# Convert any non-numeric to NaN (safety net)
X = X.apply(pd.to_numeric, errors="coerce")

print(f"\n[*] Dropped {len(DROP_COLS)} high-corr features: {DROP_COLS}")
print(f"[*] Dropped {len(ZERO_VARIANCE_COLS)} zero-variance features: {ZERO_VARIANCE_COLS}")
print(f"[*] Remaining feature columns ({len(feature_cols)}): {feature_cols}")

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\n[*] Split -> Train: {len(X_train):,} | Test: {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()

X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
X_test_imp  = pd.DataFrame(imputer.transform(X_test),      columns=feature_cols)

X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled  = scaler.transform(X_test_imp)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE PREPROCESSORS
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(imputer, MODELS_DIR / "imputer.pkl")
joblib.dump(scaler,  MODELS_DIR / "scaler.pkl")

# Save feature column list (critical for inference alignment)
with open(MODELS_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
    json.dump(feature_cols, f, indent=2)

print(f"\n[+] Saved preprocessors -> {MODELS_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
models = {
    "random_forest": {
        "model": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=4,
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "scaled": False,
    },
    "extra_trees": {
        "model": ExtraTreesClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=4,
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "scaled": False,
    },
    "gradient_boosting": {
        "model": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=5,
            random_state=RANDOM_STATE
        ),
        "scaled": False,
    },
    "logistic_regression": {
        "model": LogisticRegression(
            max_iter=1000, C=0.5, solver="lbfgs", n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "scaled": True,
    },
    "knn": {
        "model": KNeighborsClassifier(n_neighbors=9, n_jobs=-1),
        "scaled": True,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN & EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
results = {}
print("\n" + "-" * 60)
print("  Training Models")
print("-" * 60)

for name, cfg in models.items():
    model      = cfg["model"]
    use_scaled = cfg["scaled"]

    Xtr = X_train_scaled if use_scaled else X_train_imp.values
    Xte = X_test_scaled  if use_scaled else X_test_imp.values

    print(f"\n[>] Training [{name}] ...")
    t0 = time.time()
    model.fit(Xtr, y_train)
    elapsed = time.time() - t0

    preds = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else preds.astype(float)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    print(f"    Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC-ROC: {auc:.4f}  ({elapsed:.1f}s)")
    print(classification_report(y_test, preds, target_names=["Legit", "Phishing"]))

    joblib.dump(model, MODELS_DIR / f"{name}.pkl")
    results[name] = {"accuracy": acc, "f1": f1, "auc_roc": auc, "train_sec": round(elapsed, 2)}

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("  Training Complete!")
print("=" * 60)
print(f"\nDataset: {len(df):,} URLs  |  Features used: {len(feature_cols)}")
print(f"\nSummary:")
print(f"{'Model':<25} {'Accuracy':>10} {'F1':>8} {'AUC-ROC':>10}")
print("-" * 57)
best_model = max(results, key=lambda k: results[k]["auc_roc"])
for name, r in sorted(results.items(), key=lambda x: -x[1]["auc_roc"]):
    marker = "  <-- BEST" if name == best_model else ""
    print(f"{name:<25} {r['accuracy']:>10.4f} {r['f1']:>8.4f} {r['auc_roc']:>10.4f}{marker}")
print(f"\nBest model: {best_model}")
print(f"Results saved -> {RESULTS_FILE}")
print(f"Models saved  -> {MODELS_DIR}/")
