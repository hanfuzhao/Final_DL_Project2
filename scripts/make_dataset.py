"""
Dataset preparation script.
Loads and combines datasets into unified 10-class format.

Classes: clean, racism, sexism, profanity, cyberbullying,
         toxicity, hate_speech, implicit_hate, threat, sarcasm

Data sources:
- Davidson et al. hate speech dataset
- Jigsaw Toxic Comment Classification Challenge
- Jigsaw Unintended Bias in Toxicity Classification
- UC Berkeley D-Lab Measuring Hate Speech
- EDOS SemEval 2023 sexism detection
- HateXplain benchmark
- SALT-NLP ImplicitHate
- TweetEval irony subset (sarcasm)
"""

import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

LABEL_NAMES = [
    "clean", "racism", "sexism", "profanity", "cyberbullying",
    "toxicity", "hate_speech", "implicit_hate", "threat", "sarcasm",
]

PROFANITY_KEYWORDS = {
    "fuck", "shit", "damn", "ass", "bitch", "dick", "pussy",
    "hell", "crap", "bastard", "slut", "whore", "hoe",
}


def load_davidson_data() -> pd.DataFrame:
    """Load Davidson hate speech dataset with 10-class mapping.
    hate_speech → hate_speech, offensive → profanity/cyberbullying, clean → clean.
    """
    path = RAW_DIR / "labeled_data.csv"
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns={"tweet": "text"})

    def map_label(row):
        cls = row["class"]
        if cls == 2:
            return "clean"
        if cls == 0:
            return "hate_speech"
        text_words = set(str(row["text"]).lower().split())
        if text_words & PROFANITY_KEYWORDS:
            return "profanity"
        return "cyberbullying"

    df["label"] = df.apply(map_label, axis=1)
    df["source"] = "davidson"
    return df[["text", "label", "source"]]


def load_jigsaw_v1() -> pd.DataFrame:
    """Load Jigsaw v1 with priority-based 10-class mapping using multi-label fields."""
    zip_path = RAW_DIR / "jigsaw-toxic-comment-classification-challenge" / "train.csv.zip"
    if not zip_path.exists():
        print(f"[WARN] Jigsaw v1 zip not found at {zip_path}, skipping.")
        return pd.DataFrame(columns=["text", "label", "source"])

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("train.csv") as f:
            df = pd.read_csv(f)

    def map_label(row):
        if row["threat"] == 1:
            return "threat"
        if row["identity_hate"] == 1:
            return "racism"
        if row["obscene"] == 1 and row["insult"] == 0:
            return "profanity"
        if row["insult"] == 1:
            return "cyberbullying"
        if row["toxic"] == 1 or row["severe_toxic"] == 1:
            return "toxicity"
        return "clean"

    df["label"] = df.apply(map_label, axis=1)
    df = df.rename(columns={"comment_text": "text"})
    df["source"] = "jigsaw_v1"
    return df[["text", "label", "source"]]


def load_jigsaw_v2(sample_size: int = 100_000) -> pd.DataFrame:
    """Load Jigsaw v2 unintended bias dataset. Maps to toxicity/clean."""
    path = RAW_DIR / "jigsaw-unintended-bias-in-toxicity-classification" / "train.csv"
    if not path.exists():
        print(f"[WARN] Jigsaw v2 not found at {path}, skipping.")
        return pd.DataFrame(columns=["text", "label", "source"])

    df = pd.read_csv(path, usecols=["comment_text", "target"], nrows=sample_size)
    df["label"] = df["target"].apply(lambda x: "toxicity" if x >= 0.5 else "clean")
    df = df.rename(columns={"comment_text": "text"})
    df["source"] = "jigsaw_v2"
    return df[["text", "label", "source"]]


def load_berkeley_hatespeech() -> pd.DataFrame:
    """Load Berkeley D-Lab dataset with target identity fields for fine-grained mapping.
    Uses target_race/target_gender to assign racism/sexism labels.
    """
    try:
        import datasets as hf_datasets
    except ImportError:
        print("[WARN] 'datasets' package not installed, skipping Berkeley data.")
        return pd.DataFrame(columns=["text", "label", "source"])

    cache_path = RAW_DIR / "berkeley_10class.csv"
    if cache_path.exists():
        print("  (loading from local cache)")
        return pd.read_csv(cache_path)

    print("  Downloading from HuggingFace...")
    ds = hf_datasets.load_dataset("ucberkeley-dlab/measuring-hate-speech")
    raw = ds["train"].to_pandas()

    agg_dict = {"hate_speech_score": "mean"}
    for col in ["target_race", "target_gender", "target_religion"]:
        if col in raw.columns:
            agg_dict[col] = "max"

    agg = raw.groupby("text").agg(agg_dict).reset_index()
    score = agg["hate_speech_score"]

    has_race = agg.get("target_race", pd.Series(0, index=agg.index)).fillna(0) > 0.5
    has_gender = agg.get("target_gender", pd.Series(0, index=agg.index)).fillna(0) > 0.5

    conditions = [
        has_race & (score > 0),
        has_gender & (score > 0),
        score >= 1.0,
        score >= 0.0,
    ]
    choices = ["racism", "sexism", "hate_speech", "toxicity"]
    agg["label"] = np.select(conditions, choices, default="clean")

    agg["source"] = "berkeley_dlab"
    result = agg[["text", "label", "source"]]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache_path, index=False)
    print(f"  Cached to {cache_path}")
    return result


def load_edos() -> pd.DataFrame:
    """Load EDOS sexism detection dataset. Maps to sexism/threat/clean."""
    path = RAW_DIR / "edos_labelled_aggregated.csv"
    if not path.exists():
        print(f"[WARN] EDOS data not found at {path}, skipping.")
        return pd.DataFrame(columns=["text", "label", "source"])

    df = pd.read_csv(path)
    category_map = {
        "none": "clean",
        "4. prejudiced discussions": "sexism",
        "3. animosity": "sexism",
        "2. derogation": "sexism",
        "1. threats, plans to harm and incitement": "threat",
    }
    df["label"] = df["label_category"].map(category_map).fillna(
        df["label_sexist"].map({"not sexist": "clean", "sexist": "sexism"})
    )
    df["source"] = "edos"
    return df[["text", "label", "source"]]


def load_hatexplain() -> pd.DataFrame:
    """Load HateXplain with target community → racism/sexism/hate_speech mapping.
    Downloads raw JSON from the HateXplain GitHub repository.
    """
    import json
    import urllib.request

    cache_path = RAW_DIR / "hatexplain_10class.csv"
    if cache_path.exists():
        print("  (loading from local cache)")
        return pd.read_csv(cache_path)

    json_path = RAW_DIR / "hatexplain_dataset.json"
    split_path = RAW_DIR / "hatexplain_splits.json"
    base_url = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/"

    for url, dest in [
        (base_url + "dataset.json", json_path),
        (base_url + "post_id_divisions.json", split_path),
    ]:
        if not dest.exists():
            print(f"  Downloading {dest.name}...")
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, dest)

    with open(json_path) as f:
        data = json.load(f)

    RACIAL_TARGETS = {"African", "Arab", "Asian", "Hispanic", "Caucasian", "Indian",
                      "Jew", "Jewish"}
    GENDER_TARGETS = {"Women"}

    rows = []
    for post_id, entry in data.items():
        tokens = entry.get("post_tokens", [])
        text = " ".join(tokens)
        annotators = entry.get("annotators", [])

        labels = [a["label"] for a in annotators]
        majority = Counter(labels).most_common(1)[0][0]

        targets = set()
        for a in annotators:
            tgt = a.get("target", [])
            if isinstance(tgt, list):
                targets.update(tgt)

        if majority == "normal":
            label = "clean"
        elif majority == "hatespeech":
            if targets & RACIAL_TARGETS:
                label = "racism"
            elif targets & GENDER_TARGETS:
                label = "sexism"
            else:
                label = "hate_speech"
        else:
            label = "profanity"

        rows.append({"text": text, "label": label, "source": "hatexplain"})

    result = pd.DataFrame(rows)
    result.to_csv(cache_path, index=False)
    print(f"  Cached to {cache_path}")
    return result


def load_implicit_hate() -> pd.DataFrame:
    """Load ImplicitHate dataset. All entries → implicit_hate."""
    import urllib.request

    cache_path = RAW_DIR / "implicit_hate_10class.csv"
    if cache_path.exists():
        print("  (loading from local cache)")
        return pd.read_csv(cache_path)

    raw_csv = RAW_DIR / "implicit_hate_raw.csv"
    if not raw_csv.exists():
        print("  Downloading from HuggingFace...")
        url = "https://huggingface.co/datasets/SALT-NLP/ImplicitHate/resolve/main/implicit_hate.csv"
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, raw_csv)

    raw = pd.read_csv(raw_csv)
    raw["label"] = "implicit_hate"
    raw = raw.rename(columns={"post": "text"})
    raw["source"] = "implicit_hate"
    result = raw[["text", "label", "source"]]

    result.to_csv(cache_path, index=False)
    print(f"  Cached to {cache_path}")
    return result


def load_sarcasm() -> pd.DataFrame:
    """Load TweetEval irony subset for sarcasm detection."""
    try:
        import datasets as hf_datasets
    except ImportError:
        print("[WARN] 'datasets' package not installed, skipping sarcasm data.")
        return pd.DataFrame(columns=["text", "label", "source"])

    cache_path = RAW_DIR / "sarcasm.csv"
    if cache_path.exists():
        print("  (loading from local cache)")
        return pd.read_csv(cache_path)

    print("  Downloading from HuggingFace...")
    ds = hf_datasets.load_dataset("tweet_eval", data_dir="irony")

    frames = []
    for split_name in ["train", "validation", "test"]:
        if split_name in ds:
            frames.append(ds[split_name].to_pandas())
    raw = pd.concat(frames, ignore_index=True)

    raw["label"] = raw["label"].apply(lambda x: "sarcasm" if x == 1 else "clean")
    raw["source"] = "tweet_eval_irony"
    result = raw[["text", "label", "source"]]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache_path, index=False)
    print(f"  Cached to {cache_path}")
    return result


def create_unified_dataset() -> pd.DataFrame:
    """Combine all data sources into a single 10-class dataset."""
    loaders = [
        ("Davidson", load_davidson_data),
        ("Jigsaw v1", load_jigsaw_v1),
        ("Jigsaw v2", load_jigsaw_v2),
        ("Berkeley D-Lab", load_berkeley_hatespeech),
        ("EDOS", load_edos),
        ("HateXplain", load_hatexplain),
        ("ImplicitHate", load_implicit_hate),
        ("Sarcasm (TweetEval)", load_sarcasm),
    ]

    frames = []
    for name, loader in loaders:
        print(f"Loading {name}...")
        df = loader()
        print(f"  -> {len(df)} samples")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["text"])
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 5]

    valid_labels = set(LABEL_NAMES)
    combined = combined[combined["label"].isin(valid_labels)]

    print(f"\nCombined dataset: {len(combined)} samples")
    print(f"Label distribution:\n{combined['label'].value_counts().to_string()}")

    return combined


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset = create_unified_dataset()

    output_path = PROCESSED_DIR / "unified_toxicity_data.csv"
    dataset.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    label_counts = dataset["label"].value_counts()
    target_per_class = 5000
    print(f"\nBalancing: target {target_per_class} per class (oversample small, downsample large).")

    frames = []
    for label in LABEL_NAMES:
        subset = dataset[dataset["label"] == label]
        n = len(subset)
        if n == 0:
            continue
        if n >= target_per_class:
            frames.append(subset.sample(n=target_per_class, random_state=42))
        else:
            frames.append(subset.sample(n=target_per_class, replace=True, random_state=42))
    balanced = pd.concat(frames, ignore_index=True)

    balanced_path = PROCESSED_DIR / "balanced_toxicity_data.csv"
    balanced.to_csv(balanced_path, index=False)
    print(f"Balanced dataset ({len(balanced)} samples) saved to {balanced_path}")
    print(f"Label distribution:\n{balanced['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
