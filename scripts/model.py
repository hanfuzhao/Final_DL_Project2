"""
Model training and evaluation script.
Implements three required approaches:
1. Naive baseline (majority class + keyword matching)
2. Classical ML (TF-IDF + Logistic Regression)
3. Deep Learning (DistilBERT fine-tuned for toxicity classification)
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("data/outputs")

LABEL_NAMES = [
    "clean", "racism", "sexism", "profanity", "cyberbullying",
    "toxicity", "hate_speech", "implicit_hate", "threat", "sarcasm",
]


# ============================================================
# Model 1: Naive Baseline
# ============================================================

class NaiveBaseline:
    """
    Naive baseline combining majority class prediction with keyword matching.
    Maps keywords to 10 content safety categories.
    """

    KEYWORD_RULES = [
        ({"nigger", "nigga", "kike", "spic", "chink", "wetback", "gook", "coon"}, 1),       # racism
        ({"bitch", "slut", "whore", "hoe", "misogyn"}, 2),                                   # sexism
        ({"kill", "die", "murder", "shoot", "bomb", "stab", "attack"}, 8),                    # threat
        ({"fuck", "shit", "damn", "ass", "dick", "pussy", "crap", "bastard"}, 3),             # profanity
        ({"stupid", "idiot", "dumb", "ugly", "fat", "loser", "moron"}, 4),                    # cyberbullying
        ({"faggot", "fag", "tranny", "retard"}, 6),                                           # hate_speech
    ]
    NUM_CLASSES = len(LABEL_NAMES)

    def __init__(self):
        self.majority_class = None

    def fit(self, X: pd.Series, y: pd.Series):
        self.majority_class = y.mode()[0]
        return self

    def _classify(self, text: str) -> int:
        words = set(str(text).lower().split())
        for keywords, label_id in self.KEYWORD_RULES:
            if words & keywords:
                return label_id
        return self.majority_class

    def predict(self, texts: pd.Series) -> np.ndarray:
        return np.array([self._classify(t) for t in texts])

    def predict_single(self, text: str) -> dict:
        label = self._classify(text)
        confidence = 0.7 if label != self.majority_class else 0.4
        probs = [(1.0 - confidence) / (self.NUM_CLASSES - 1)] * self.NUM_CLASSES
        probs[label] = confidence
        return {"label": label, "probabilities": probs}


# ============================================================
# Model 2: Classical ML (Logistic Regression with TF-IDF)
# ============================================================

def train_classical_model(X_train, y_train):
    """Train a Logistic Regression model with TF-IDF features."""
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


# ============================================================
# Model 3: Deep Learning (DistilBERT)
# ============================================================

def train_distilbert(train_df: pd.DataFrame, test_df: pd.DataFrame, epochs: int = 3):
    """Fine-tune DistilBERT for toxicity classification."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from transformers import (
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    use_device = os.environ.get("SAFETYPE_DEVICE", "auto")
    if use_device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(use_device)
    print(f"Using device: {device}", flush=True)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    class ToxicityDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.texts = texts.tolist()
            self.labels = labels.tolist()
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                str(self.texts[idx]),
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    train_dataset = ToxicityDataset(train_df["text_clean"], train_df["label_encoded"], tokenizer)
    test_dataset = ToxicityDataset(test_df["text_clean"], test_df["label_encoded"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(LABEL_NAMES)
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}", flush=True)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}", flush=True)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())

        f1 = f1_score(all_labels, all_preds, average="weighted")
        print(f"  Validation F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(MODELS_DIR / "distilbert-toxicity")
            tokenizer.save_pretrained(MODELS_DIR / "distilbert-toxicity")
            print(f"  Saved best model (F1={best_f1:.4f})")

    return model, tokenizer, all_preds, all_labels


# ============================================================
# Evaluation & Visualization
# ============================================================

def plot_confusion_matrix(y_true, y_pred, model_name: str):
    """Generate and save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()


def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    """Evaluate a model and return metrics."""
    report = classification_report(
        y_true, y_pred, target_names=LABEL_NAMES, output_dict=True
    )
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted")

    print(f"\n{'='*60}")
    print(f"Results for: {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1_w:.4f}")

    plot_confusion_matrix(y_true, y_pred, model_name)

    return {
        "model": model_name,
        "accuracy": acc,
        "weighted_f1": f1_w,
        "report": report,
    }


def error_analysis(test_df: pd.DataFrame, y_true, y_pred, model_name: str, n: int = 5):
    """Identify and analyze mispredictions."""
    errors_idx = np.where(np.array(y_true) != np.array(y_pred))[0]
    if len(errors_idx) == 0:
        print("No errors found.")
        return []

    sample_idx = np.random.choice(errors_idx, size=min(n, len(errors_idx)), replace=False)
    analysis = []
    print(f"\n--- Error Analysis: {model_name} ---")
    for i, idx in enumerate(sample_idx):
        row = test_df.iloc[idx]
        true_label = LABEL_NAMES[y_true[idx]]
        pred_label = LABEL_NAMES[y_pred[idx]]
        text = row["text"][:200]
        print(f"\n[Error {i+1}]")
        print(f"  Text: {text}")
        print(f"  True: {true_label} | Predicted: {pred_label}")
        analysis.append({
            "text": text,
            "true_label": true_label,
            "predicted_label": pred_label,
        })
    return analysis


# ============================================================
# Main Training Pipeline
# ============================================================

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train_tfidf = sparse.load_npz(PROCESSED_DIR / "X_train_tfidf.npz")
    X_test_tfidf = sparse.load_npz(PROCESSED_DIR / "X_test_tfidf.npz")

    y_train = train_df["label_encoded"].values
    y_test = test_df["label_encoded"].values

    results = []

    # --- Model 1: Naive Baseline ---
    print("\n" + "="*60)
    print("Training Model 1: Naive Baseline")
    print("="*60)
    baseline = NaiveBaseline()
    baseline.fit(train_df["text"], train_df["label_encoded"])
    y_pred_baseline = baseline.predict(test_df["text"])
    results.append(evaluate_model(y_test, y_pred_baseline, "Naive Baseline"))
    error_analysis(test_df, y_test.tolist(), y_pred_baseline.tolist(), "Naive Baseline")
    joblib.dump(baseline, MODELS_DIR / "naive_baseline.pkl")

    # --- Model 2: Classical ML (Logistic Regression) ---
    print("\n" + "="*60)
    print("Training Model 2: Logistic Regression (TF-IDF)")
    print("="*60)
    lr_model = train_classical_model(X_train_tfidf, y_train)
    y_pred_lr = lr_model.predict(X_test_tfidf)
    results.append(evaluate_model(y_test, y_pred_lr, "Logistic Regression"))
    error_analysis(test_df, y_test.tolist(), y_pred_lr.tolist(), "Logistic Regression")
    joblib.dump(lr_model, MODELS_DIR / "logistic_regression.pkl")

    # --- Model 3: Deep Learning (DistilBERT) ---
    import gc
    gc.collect()
    print("\n" + "="*60)
    print("Training Model 3: DistilBERT")
    print("="*60)
    _, _, y_pred_dl, y_true_dl = train_distilbert(train_df, test_df, epochs=3)
    results.append(evaluate_model(y_true_dl, y_pred_dl, "DistilBERT"))
    error_analysis(test_df, y_true_dl, y_pred_dl, "DistilBERT")

    # --- Summary ---
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    summary_df = pd.DataFrame([
        {"Model": r["model"], "Accuracy": r["accuracy"], "Weighted F1": r["weighted_f1"]}
        for r in results
    ])
    print(summary_df.to_string(index=False))

    summary_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    with open(OUTPUTS_DIR / "full_results.json", "w") as f:
        serializable = []
        for r in results:
            entry = {k: v for k, v in r.items() if k != "report"}
            entry["report"] = {k: v for k, v in r["report"].items()}
            serializable.append(entry)
        json.dump(serializable, f, indent=2, default=str)

    # --- Comparison Bar Chart ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models = [r["model"] for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["weighted_f1"] for r in results]

    axes[0].bar(models, accs, color=["#4CAF50", "#2196F3", "#FF9800"])
    axes[0].set_title("Accuracy Comparison")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")

    axes[1].bar(models, f1s, color=["#4CAF50", "#2196F3", "#FF9800"])
    axes[1].set_title("Weighted F1 Comparison")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Weighted F1")

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "model_comparison.png", dpi=150)
    plt.close()

    print("\nAll results saved to outputs/")


if __name__ == "__main__":
    main()
