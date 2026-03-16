"""
SafeType — AI-Powered Content Safety Detection for Children

Main entry point for running inference with trained models.

Usage:
  python main.py                         # interactive mode
  python main.py --text "some message"   # single prediction
  python main.py --model distilbert      # choose model backend
  python main.py --batch input.txt       # batch file prediction

Models available:
  - naive_baseline      : Keyword-matching + majority class (Naive Baseline)
  - logistic_regression : TF-IDF + Logistic Regression (Classical ML)
  - distilbert          : Fine-tuned DistilBERT (Neural Network / Deep Learning)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import joblib

MODELS_DIR = Path(__file__).resolve().parent / "models"
LABEL_NAMES = [
    "clean", "racism", "sexism", "profanity", "cyberbullying",
    "toxicity", "hate_speech", "implicit_hate", "threat", "sarcasm",
]
SEVERITY_ICONS = {
    "clean": "✅", "racism": "🚫", "sexism": "🚫", "profanity": "⚠️",
    "cyberbullying": "🚫", "toxicity": "⚠️", "hate_speech": "🚫",
    "implicit_hate": "⚠️", "threat": "🚫", "sarcasm": "💬",
}


def clean_text(text: str) -> str:
    """Minimal cleaning for inference."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SafeTypePredictor:
    """Unified predictor that loads any of the three trained models."""

    def __init__(self, model_type: str = "logistic_regression"):
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.tokenizer = None
        self._load(model_type)

    def _load(self, model_type: str):
        if model_type == "naive_baseline":
            from scripts.model import NaiveBaseline  # noqa: F401 — needed for unpickling
            self.model = joblib.load(MODELS_DIR / "naive_baseline.pkl")

        elif model_type == "logistic_regression":
            self.model = joblib.load(MODELS_DIR / "logistic_regression.pkl")
            self.vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")

        elif model_type == "distilbert":
            import torch
            from transformers import (
                DistilBertTokenizer,
                DistilBertForSequenceClassification,
            )
            path = MODELS_DIR / "distilbert-toxicity"
            self.tokenizer = DistilBertTokenizer.from_pretrained(str(path))
            self.model = DistilBertForSequenceClassification.from_pretrained(str(path))
            self.model.eval()
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
            self.model.to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def predict(self, text: str) -> dict:
        """Return label, confidence, and per-class probabilities for one text."""
        if not text.strip():
            return self._empty_result()

        if self.model_type == "naive_baseline":
            result = self.model.predict_single(text)
            label_id = result["label"]
            probs = result["probabilities"]

        elif self.model_type == "logistic_regression":
            cleaned = clean_text(text)
            X = self.vectorizer.transform([cleaned])
            label_id = int(self.model.predict(X)[0])
            probs = self.model.predict_proba(X)[0].tolist()

        elif self.model_type == "distilbert":
            import torch
            enc = self.tokenizer(
                text, truncation=True, padding="max_length",
                max_length=128, return_tensors="pt",
            )
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids=ids, attention_mask=mask).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()
                label_id = int(torch.argmax(logits, dim=1).item())

        label = LABEL_NAMES[label_id]
        return {
            "label": label,
            "confidence": round(probs[label_id], 4),
            "probabilities": {n: round(p, 4) for n, p in zip(LABEL_NAMES, probs)},
        }

    @staticmethod
    def _empty_result() -> dict:
        probs = {name: 0.0 for name in LABEL_NAMES}
        probs["clean"] = 1.0
        return {"label": "clean", "confidence": 1.0, "probabilities": probs}


def print_result(text: str, result: dict):
    """Pretty-print a single prediction."""
    icon = SEVERITY_ICONS.get(result["label"], "❓")
    print(f"\n  Input : {text}")
    print(f"  Result: {icon}  {result['label']}  (confidence {result['confidence']:.1%})")
    top3 = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
    probs_str = "  ".join(f"{k}={v:.3f}" for k, v in top3)
    print(f"  Top 3 : {probs_str}")


def interactive_mode(predictor: SafeTypePredictor):
    """Run an interactive REPL for live text analysis."""
    print("\n" + "=" * 60)
    print("  SafeType — Interactive Mode")
    print(f"  Model: {predictor.model_type}")
    print("  Type a message and press Enter. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not text:
            continue
        result = predictor.predict(text)
        print_result(text, result)


def batch_mode(predictor: SafeTypePredictor, filepath: str):
    """Analyze every line in a text file."""
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    print(f"\nAnalyzing {len(lines)} messages from {filepath}...\n")
    for line in lines:
        result = predictor.predict(line)
        print_result(line, result)
    print(f"\n--- Done ({len(lines)} messages analyzed) ---")


def main():
    parser = argparse.ArgumentParser(
        description="SafeType — AI content safety detection",
    )
    parser.add_argument(
        "--model", type=str, default="logistic_regression",
        choices=["naive_baseline", "logistic_regression", "distilbert"],
        help="Which model to use for inference",
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Analyze a single text string",
    )
    parser.add_argument(
        "--batch", type=str, default=None,
        help="Path to a text file (one message per line) for batch analysis",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model} ...")
    predictor = SafeTypePredictor(args.model)
    print("Model loaded.\n")

    if args.text:
        result = predictor.predict(args.text)
        print_result(args.text, result)
    elif args.batch:
        batch_mode(predictor, args.batch)
    else:
        interactive_mode(predictor)


if __name__ == "__main__":
    main()
