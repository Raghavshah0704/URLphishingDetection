"""
PhiUSIIL Phishing URL Detector -- REST API Server
=================================================
Flask API server providing endpoint for the browser extension.
Runs on http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from src.predictor import PhishingPredictor

app = Flask(__name__)
CORS(app)  # Enable CORS for browser extension requests

print("[*] Initializing PhishingPredictor models...")
predictor = PhishingPredictor()
print("[+] Models loaded successfully!")

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "service": "PhiUSIIL Phishing URL Detector API",
        "version": "1.0.0"
    })

@app.route("/api/predict", methods=["POST", "GET"])
def predict():
    try:
        url = None
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            url = data.get("url") or request.form.get("url")
        else:
            url = request.args.get("url")

        if not url or not url.strip():
            return jsonify({"error": "No URL provided"}), 400

        url_clean = url.strip()
        if not url_clean.startswith(("http://", "https://")):
            url_clean = "http://" + url_clean

        result = predictor.predict(url_clean)
        result["url"] = url_clean

        # Convert float types to python native floats for JSON serializability
        result["confidence"] = float(result["confidence"])
        for m in result["per_model"]:
            result["per_model"][m]["prob"] = float(result["per_model"][m]["prob"])

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("[*] Starting API Server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
