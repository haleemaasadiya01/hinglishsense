"""
models/baseline.py
------------------
TF-IDF + Logistic Regression baseline sentiment classifier.

Trains on 80 % of whatever data is available, evaluates on 20 %.
If a separate test file exists (preprocessed_test.csv) it is used as
the held-out set instead.

Reports:
  • Accuracy
  • Per-class F1 (precision / recall / support)
  • Macro-F1
  • Confusion matrix (printed + saved as PNG)
"""

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

LABEL_ORDER = ["positive", "negative", "neutral"]
MODELS_DIR = Path("models/saved")
EVAL_DIR   = Path("evaluation")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    """Return (X_train, X_test, y_train, y_test) splits."""
    train_path = Path("data/processed/preprocessed_train.csv")
    test_path  = Path("data/processed/preprocessed_test.csv")
    dev_path   = Path("data/processed/preprocessed_data.csv")

    if train_path.exists() and test_path.exists():
        print("Using dedicated train / test files.")
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)
        return (
            train_df["clean_text"].fillna(""),
            test_df["clean_text"].fillna(""),
            train_df["sentiment"],
            test_df["sentiment"],
        )

    # fall back to a single file with 80/20 split
    src = train_path if train_path.exists() else dev_path
    if not src.exists():
        raise FileNotFoundError(
            "No preprocessed data found. Run preprocessing/pipeline.py first."
        )
    print(f"No separate test split found — using 80/20 split of {src.name}")
    df = pd.read_csv(src)
    X = df["clean_text"].fillna("")
    y = df["sentiment"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ── Model definition ──────────────────────────────────────────────────────────

def build_model() -> Pipeline:
    """
    TF-IDF → Logistic Regression pipeline.

    TF-IDF choices:
      ngram_range=(1,2)  captures bigrams like "nahi acha" (not good),
                         which carry sentiment in code-mixed text.
      sublinear_tf=True  dampens very high term frequencies (common in
                         short social-media posts).
      min_df=2           drops hapax legomena (one-off misspellings).

    LR choices:
      C=1.0              default regularisation — strong enough for sparse TF-IDF.
      max_iter=1000      needed for convergence on multi-class problems.
      class_weight=balanced  corrects for label imbalance (neutral is usually
                              under-represented).
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                    max_features=50_000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


# ── Evaluation helpers ────────────────────────────────────────────────────────

def print_report(y_true, y_pred, split_name="Test"):
    acc = accuracy_score(y_true, y_pred)
    sep = "-" * 55
    print(f"\n{sep}")
    print(f"  {split_name} Accuracy : {acc:.4f}  ({acc*100:.2f} %)")
    print(sep)
    print(
        classification_report(
            y_true, y_pred, labels=LABEL_ORDER, zero_division=0
        )
    )


def save_confusion_matrix(y_true, y_pred, filename="baseline_confusion.png"):
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_ORDER,
        yticklabels=LABEL_ORDER,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Baseline – Confusion Matrix")
    plt.tight_layout()
    path = EVAL_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    model = build_model()
    print("\nTraining TF-IDF + Logistic Regression ...")
    model.fit(X_train, y_train)

    # evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    print_report(y_train, y_pred_train, "Train")
    print_report(y_test,  y_pred_test,  "Test")

    save_confusion_matrix(y_test, y_pred_test)

    # persist
    model_path = MODELS_DIR / "baseline_tfidf_lr.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved -> {model_path}")
