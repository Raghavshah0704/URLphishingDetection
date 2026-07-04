"""
Predictor module for PhiUSIIL-based Phishing URL Detector.
Loads all trained models and runs ensemble prediction.

Usage:
    from src.predictor import PhishingPredictor
    p = PhishingPredictor()
    # Direct URL prediction (extracts features automatically):
    result = p.predict("https://www.google.com")
    print(result)
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

# The 19 numeric feature columns expected by the models
# (loaded from the saved JSON at runtime for alignment safety)

class PhishingPredictor:
    """Ensemble predictor wrapping all trained PhiUSIIL models."""

    MODEL_NAMES = [
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "logistic_regression",
        "knn",
    ]
    # These models used StandardScaler during training
    SCALED_MODELS = {"logistic_regression", "knn"}

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self._load_artifacts()

    def _load_artifacts(self):
        """Load preprocessors, feature list, and all models."""
        self.imputer = joblib.load(self.models_dir / "imputer.pkl")
        self.scaler  = joblib.load(self.models_dir / "scaler.pkl")

        with open(self.models_dir / "feature_columns.json") as f:
            self.feature_cols = json.load(f)

        self.models = {}
        for name in self.MODEL_NAMES:
            path = self.models_dir / f"{name}.pkl"
            if path.exists():
                self.models[name] = joblib.load(path)
            else:
                print(f"[WARNING] Model not found: {path}")

    def _preprocess(self, feature_dict: dict) -> tuple:
        """
        Turn a raw feature dict into (X_raw_imputed, X_scaled) DataFrames.
        Missing features are filled via the trained imputer.
        """
        # Build DataFrame with proper column order
        X = pd.DataFrame([feature_dict]).reindex(columns=self.feature_cols)
        X = X.apply(pd.to_numeric, errors="coerce")

        X_imp    = pd.DataFrame(self.imputer.transform(X),    columns=self.feature_cols)
        X_scaled = pd.DataFrame(self.scaler.transform(X_imp), columns=self.feature_cols)

        return X_imp, X_scaled

    def predict(self, input_data) -> dict:
        """
        Run ensemble prediction.

        Parameters
        ----------
        input_data : dict or str
            A dictionary mapping feature name → value for the 19 feature columns,
            or a raw URL string (features will be extracted from it automatically).

        Returns
        -------
        dict with keys:
            final_label   : int   (0=Legit, 1=Phishing)
            final_verdict : str
            confidence    : float (mean phishing probability across models)
            per_model     : dict  {model_name: {"label": int, "prob": float}}
        """
        if isinstance(input_data, str):
            from src.url_feature_extractor import extract_features_from_url
            feature_dict = extract_features_from_url(input_data)
        else:
            feature_dict = input_data

        X_imp, X_scaled = self._preprocess(feature_dict)

        per_model = {}
        votes = []
        probs = []

        for name, model in self.models.items():
            X_in = X_scaled.values if name in self.SCALED_MODELS else X_imp.values

            label = int(model.predict(X_in)[0])
            prob  = float(model.predict_proba(X_in)[0][1]) if hasattr(model, "predict_proba") else float(label)

            per_model[name] = {"label": label, "prob": round(prob, 4)}
            votes.append(label)
            probs.append(prob)

        # Majority vote
        final_label = 1 if sum(votes) >= (len(votes) / 2) else 0
        confidence  = round(float(np.mean(probs)), 4)

        return {
            "final_label":   final_label,
            "final_verdict": "PHISHING" if final_label == 1 else "LEGITIMATE",
            "confidence":    confidence,
            "per_model":     per_model,
        }

