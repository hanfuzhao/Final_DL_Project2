"""
Setup script: runs the full pipeline from raw data to trained models.
Steps:
  1. make_dataset.py  — load and combine raw datasets
  2. build_features.py — clean text, extract TF-IDF and handcrafted features
  3. model.py — train Naive Baseline, Logistic Regression, and DistilBERT
  4. experiment.py — run sensitivity analysis and noise robustness experiments
"""

import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    ("scripts/make_dataset.py", "Creating unified dataset from raw sources..."),
    ("scripts/build_features.py", "Building TF-IDF and handcrafted features..."),
    ("scripts/model.py", "Training all three models (Naive, LR, DistilBERT)..."),
    ("scripts/experiment.py", "Running experiments (training size + noise robustness)..."),
]


def run_step(script: str, description: str):
    """Run a pipeline step and stream its output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Script: {script}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        [sys.executable, script],
        cwd=Path(__file__).resolve().parent,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {script} exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    print("=" * 60)
    print("  SafeType — Full Pipeline Setup")
    print("=" * 60)

    for script, desc in SCRIPTS:
        run_step(script, desc)

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("  Models saved to:  models/")
    print("  Outputs saved to: data/outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
