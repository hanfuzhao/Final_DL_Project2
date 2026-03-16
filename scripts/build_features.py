"""
Feature engineering pipeline.
Transforms raw text into features for classical ML models.
"""

import re
import string
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def download_nltk_resources():
    """Download required NLTK data."""
    for resource in ["stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
        nltk.download(resource, quiet=True)


def clean_text(text: str) -> str:
    """Apply text cleaning: lowercase, remove URLs, mentions, special chars."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"rt\s+", "", text)
    text = re.sub(r"&amp;?", "and", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """Lemmatize and remove stop words."""
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


def extract_handcrafted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hand-crafted features useful for toxicity detection."""
    features = pd.DataFrame(index=df.index)
    features["char_count"] = df["text"].str.len()
    features["word_count"] = df["text"].str.split().str.len()
    features["avg_word_len"] = features["char_count"] / features["word_count"].clip(lower=1)
    features["uppercase_ratio"] = df["text"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    features["exclamation_count"] = df["text"].str.count("!")
    features["question_count"] = df["text"].str.count(r"\?")
    features["caps_word_count"] = df["text"].apply(
        lambda x: sum(1 for w in x.split() if w.isupper() and len(w) > 1)
    )
    return features


def build_tfidf_features(
    train_texts: pd.Series,
    test_texts: pd.Series,
    max_features: int = 20_000,
) -> tuple:
    """Build TF-IDF feature matrix."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)
    return X_train_tfidf, X_test_tfidf, vectorizer


def main():
    download_nltk_resources()

    print("Loading balanced dataset...")
    df = pd.read_csv(PROCESSED_DIR / "balanced_toxicity_data.csv")
    print(f"Loaded {len(df)} samples")

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    print("Cleaning text...")
    df["text_clean"] = df["text"].apply(clean_text)
    df["text_processed"] = df["text_clean"].apply(
        lambda x: lemmatize_text(x, lemmatizer, stop_words)
    )

    label_map = {
        "clean": 0, "racism": 1, "sexism": 2, "profanity": 3,
        "cyberbullying": 4, "toxicity": 5, "hate_speech": 6,
        "implicit_hate": 7, "threat": 8, "sarcasm": 9,
    }
    df["label_encoded"] = df["label"].map(label_map)

    print("Splitting data...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label_encoded"]
    )

    print("Building TF-IDF features...")
    X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf_features(
        train_df["text_processed"], test_df["text_processed"]
    )

    print("Extracting handcrafted features...")
    train_handcrafted = extract_handcrafted_features(train_df)
    test_handcrafted = extract_handcrafted_features(test_df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    from scipy import sparse
    sparse.save_npz(PROCESSED_DIR / "X_train_tfidf.npz", X_train_tfidf)
    sparse.save_npz(PROCESSED_DIR / "X_test_tfidf.npz", X_test_tfidf)
    train_handcrafted.to_csv(PROCESSED_DIR / "train_handcrafted.csv", index=False)
    test_handcrafted.to_csv(PROCESSED_DIR / "test_handcrafted.csv", index=False)

    print(f"\nFeature pipeline complete:")
    print(f"  TF-IDF shape: {X_train_tfidf.shape}")
    print(f"  Handcrafted features: {train_handcrafted.shape[1]}")
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"  Label distribution (train):\n{train_df['label'].value_counts()}")


if __name__ == "__main__":
    main()
