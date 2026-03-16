"""
SafeType — Flask web app for content safety detection.
Run locally:  python app.py
Deploy:       gunicorn app:app
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flask import Flask, request, jsonify, send_from_directory

logger.info("Importing SafeTypePredictor ...")
from main import SafeTypePredictor, LABEL_NAMES
logger.info("Import complete.")

DIST_DIR = os.path.join(PROJECT_ROOT, 'frontend', 'dist')

app = Flask(__name__, static_folder=DIST_DIR, static_url_path='')


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        resp = app.make_default_options_response()
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

MODELS = {}

CATEGORY_META = {
    "clean":         {"icon": "✅", "color": "#10b981", "severity": "safe",   "desc": "Safe content"},
    "racism":        {"icon": "🚫", "color": "#dc2626", "severity": "danger", "desc": "Racial discrimination"},
    "sexism":        {"icon": "🚫", "color": "#e11d48", "severity": "danger", "desc": "Gender-based hate"},
    "profanity":     {"icon": "🤬", "color": "#ea580c", "severity": "warn",   "desc": "Obscene language"},
    "cyberbullying":  {"icon": "😢", "color": "#9333ea", "severity": "danger", "desc": "Personal attacks"},
    "toxicity":      {"icon": "☠️",  "color": "#7c3aed", "severity": "warn",   "desc": "Toxic language"},
    "hate_speech":   {"icon": "🔥", "color": "#b91c1c", "severity": "danger", "desc": "Hate speech"},
    "implicit_hate": {"icon": "🎭", "color": "#c2410c", "severity": "warn",   "desc": "Indirect / coded hate"},
    "threat":        {"icon": "⚠️",  "color": "#991b1b", "severity": "danger", "desc": "Threats or violence"},
    "sarcasm":       {"icon": "😏", "color": "#4f46e5", "severity": "info",   "desc": "Sarcasm / irony"},
}


SEVERE_LABELS = {"racism", "threat", "hate_speech", "sexism"}
INFO_LABELS    = {"sarcasm"}


def calibrate(result):
    """Post-processing calibration so normal messages aren't over-flagged.

    With 10 balanced classes the model only saw 10 % clean data, making it
    biased toward harmful labels even on safe text.  Three-tier thresholds:

      Severe  (racism, threat, …)   → flag if conf >= 25 %
      Info    (sarcasm)             → flag only if conf >= 75 %
      Others  (profanity, …)       → flag if conf >= 50 %

    Additionally, if clean probability > 30 %, always default to clean.
    When overriding to clean, probabilities are adjusted for consistency."""
    label = result["label"]
    conf  = result["confidence"]
    probs = result["probabilities"]
    clean_p = probs.get("clean", 0)

    if label == "clean":
        return result

    override = False

    if label in INFO_LABELS:
        override = True
    elif clean_p < 0.03:
        pass
    elif clean_p > 0.30:
        override = True
    else:
        if label in SEVERE_LABELS:
            threshold = 0.25
        elif label in INFO_LABELS:
            threshold = 0.75
        else:
            threshold = 0.50
        if conf < threshold:
            override = True

    if override:
        result["original_label"] = label
        orig_p = probs[label]
        if orig_p > clean_p:
            probs[label] = clean_p
            probs["clean"] = orig_p
        result["label"] = "clean"
        result["confidence"] = probs["clean"]

    return result


def get_predictor(model_type: str) -> SafeTypePredictor:
    if model_type not in MODELS:
        MODELS[model_type] = SafeTypePredictor(model_type)
    return MODELS[model_type]


@app.route("/")
def index():
    return send_from_directory(DIST_DIR, 'index.html')


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    model_type = data.get("model", "logistic_regression")

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if model_type not in ("naive_baseline", "logistic_regression", "distilbert"):
        return jsonify({"error": f"Unknown model: {model_type}"}), 400

    try:
        predictor = get_predictor(model_type)
        result = predictor.predict(text)
    except Exception as e:
        logger.error("Model error: %s", e, exc_info=True)
        return jsonify({"error": f"Model loading failed: {str(e)}"}), 500

    result = calibrate(result)

    result["meta"] = CATEGORY_META.get(result["label"], {})
    result["all_probs"] = [
        {"name": n, "prob": result["probabilities"][n], **CATEGORY_META.get(n, {})}
        for n in LABEL_NAMES
    ]
    result["all_probs"].sort(key=lambda x: -x["prob"])

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
