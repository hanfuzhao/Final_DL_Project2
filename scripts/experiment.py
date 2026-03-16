"""
Experiments: Training Set Size Sensitivity, Noise Robustness, and Ablation Study.

Experiment 1: How does model performance change with different training set sizes?
Experiment 2: How robust are models to noisy/perturbed input text?
Experiment 3: Ablation study — incremental TF-IDF + Logistic Regression pipeline.
"""

import time
import numpy as np
import pandas as pd
import random
import string
from pathlib import Path
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from build_features import clean_text

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("data/outputs")


# ============================================================
# Experiment 1: Training Set Size Sensitivity
# ============================================================

def training_size_experiment(fractions: list = None):
    """Evaluate Logistic Regression performance across training set sizes."""
    if fractions is None:
        fractions = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    print("=" * 60)
    print("Experiment 1: Training Set Size Sensitivity Analysis")
    print("=" * 60)

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    train_df["text_processed"] = train_df["text_processed"].fillna("")
    test_df["text_processed"] = test_df["text_processed"].fillna("")
    y_test = test_df["label_encoded"].values

    results = []
    for frac in fractions:
        if frac < 1.0:
            sample = train_df.sample(frac=frac, random_state=42)
        else:
            sample = train_df

        vectorizer = TfidfVectorizer(
            max_features=20_000, ngram_range=(1, 2),
            min_df=3, max_df=0.95, sublinear_tf=True,
        )
        X_train = vectorizer.fit_transform(sample["text_processed"])
        X_test = vectorizer.transform(test_df["text_processed"])

        model = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, sample["label_encoded"].values)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        n_samples = len(sample)

        print(f"  Fraction: {frac:.0%} ({n_samples:,} samples) -> Acc: {acc:.4f}, F1: {f1:.4f}")
        results.append({
            "fraction": frac,
            "n_samples": n_samples,
            "accuracy": acc,
            "weighted_f1": f1,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUTS_DIR / "experiment_training_size.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(results_df["n_samples"], results_df["accuracy"], "b-o", label="Accuracy", linewidth=2)
    ax1.plot(results_df["n_samples"], results_df["weighted_f1"], "r-s", label="Weighted F1", linewidth=2)
    ax1.set_xlabel("Training Set Size", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Training Set Size vs. Model Performance", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "experiment_training_size.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {OUTPUTS_DIR}/experiment_training_size.*")
    return results_df


# ============================================================
# Experiment 2: Noise Robustness
# ============================================================

def add_typos(text: str, typo_rate: float = 0.1) -> str:
    """Simulate typos by randomly swapping/inserting/deleting characters."""
    chars = list(text)
    n_modifications = max(1, int(len(chars) * typo_rate))
    for _ in range(n_modifications):
        if not chars:
            break
        op = random.choice(["swap", "insert", "delete", "replace"])
        pos = random.randint(0, max(0, len(chars) - 1))
        if op == "swap" and pos < len(chars) - 1:
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        elif op == "insert":
            chars.insert(pos, random.choice(string.ascii_lowercase))
        elif op == "delete" and len(chars) > 1:
            chars.pop(pos)
        elif op == "replace":
            chars[pos] = random.choice(string.ascii_lowercase)
    return "".join(chars)


def noise_robustness_experiment(noise_levels: list = None):
    """Test how model performance degrades with increasing input noise."""
    if noise_levels is None:
        noise_levels = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

    print("\n" + "=" * 60)
    print("Experiment 2: Noise Robustness Testing")
    print("=" * 60)

    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    test_df["text"] = test_df["text"].fillna("")
    test_df["text_processed"] = test_df["text_processed"].fillna("")
    y_test = test_df["label_encoded"].values
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    lr_model = joblib.load(MODELS_DIR / "logistic_regression.pkl")

    results = []
    for noise_level in noise_levels:
        if noise_level > 0:
            noisy_texts = test_df["text"].apply(lambda x: add_typos(str(x), noise_level))
            noisy_processed = noisy_texts.apply(clean_text)
        else:
            noisy_processed = test_df["text_processed"]

        X_test_noisy = vectorizer.transform(noisy_processed.fillna(""))
        y_pred = lr_model.predict(X_test_noisy)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"  Noise level: {noise_level:.0%} -> Acc: {acc:.4f}, F1: {f1:.4f}")
        results.append({
            "noise_level": noise_level,
            "accuracy": acc,
            "weighted_f1": f1,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUTS_DIR / "experiment_noise_robustness.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df["noise_level"] * 100, results_df["accuracy"], "b-o", label="Accuracy", linewidth=2)
    ax.plot(results_df["noise_level"] * 100, results_df["weighted_f1"], "r-s", label="Weighted F1", linewidth=2)
    ax.set_xlabel("Noise Level (%)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Input Noise Level vs. Model Performance", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "experiment_noise_robustness.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {OUTPUTS_DIR}/experiment_noise_robustness.*")
    return results_df


# ============================================================
# Experiment 3: Ablation Study — TF-IDF + LR Feature Pipeline
# ============================================================

# Ordered ablation configs — each dict is fully self-contained
ABLATION_CONFIGS = [
    {
        "name": "1. Baseline\n(unigrams, raw text)",
        "text_col": "text_clean",
        "ngram_range": (1, 1),
        "sublinear_tf": False,
        "class_weight": None,
    },
    {
        "name": "2. +Preprocessing\n(lemmatized text)",
        "text_col": "text_processed",
        "ngram_range": (1, 1),
        "sublinear_tf": False,
        "class_weight": None,
    },
    {
        "name": "3. +Bigrams\n(unigrams+bigrams)",
        "text_col": "text_processed",
        "ngram_range": (1, 2),
        "sublinear_tf": False,
        "class_weight": None,
    },
    {
        "name": "4. +Sublinear TF\n(log-scaled TF)",
        "text_col": "text_processed",
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "class_weight": None,
    },
    {
        "name": "5. +Balanced Classes\n(full pipeline)",
        "text_col": "text_processed",
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "class_weight": "balanced",
    },
]


def ablation_experiment():
    """Incremental ablation: isolate each TF-IDF + LR design choice."""
    print("\n" + "=" * 65)
    print("Experiment 3: Ablation Study — TF-IDF + Logistic Regression")
    print("=" * 65)

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    y_test = test_df["label_encoded"].values

    for col in ("text_clean", "text_processed"):
        train_df[col] = train_df[col].fillna("")
        test_df[col] = test_df[col].fillna("")

    results = []
    for cfg in ABLATION_CONFIGS:
        label = cfg["name"].replace("\n", " ")
        print(f"\n  Running: {label}")

        t0 = time.time()

        vectorizer = TfidfVectorizer(
            max_features=20_000,
            ngram_range=cfg["ngram_range"],
            min_df=3,
            max_df=0.95,
            sublinear_tf=cfg["sublinear_tf"],
        )
        X_train = vectorizer.fit_transform(train_df[cfg["text_col"]])
        X_test = vectorizer.transform(test_df[cfg["text_col"]])

        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight=cfg["class_weight"],
            random_state=42,
        )
        model.fit(X_train, train_df["label_encoded"].values)
        y_pred = model.predict(X_test)

        elapsed = time.time() - t0
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"    Acc: {acc:.4f} | F1: {f1:.4f} | Time: {elapsed:.1f}s")
        results.append({
            "config": label,
            "accuracy": acc,
            "weighted_f1": f1,
            "time_sec": round(elapsed, 1),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUTS_DIR / "experiment_ablation.csv", index=False)

    # Bar chart
    short_labels = [cfg["name"] for cfg in ABLATION_CONFIGS]
    x = np.arange(len(short_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, results_df["accuracy"], width,
                   label="Accuracy", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, results_df["weighted_f1"], width,
                   label="Weighted F1", color="#FF9800", alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Ablation Study: Incremental TF-IDF + LR Pipeline Contributions",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "experiment_ablation.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {OUTPUTS_DIR}/experiment_ablation.*")
    return results_df


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    training_size_experiment()
    noise_robustness_experiment()
    ablation_experiment()

    print("\nAll experiments complete!")


if __name__ == "__main__":
    main()
