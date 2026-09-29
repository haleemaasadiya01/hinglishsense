"""
evaluation/compare.py
---------------------
Standalone script that loads both saved models, runs them on the test split,
and prints a side-by-side comparison table + saves it to evaluation/.

Run AFTER training both models:
  python models/baseline.py
  python models/transformer_model.py
  python evaluation/compare.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocessing.pipeline import normalise_text

LABELS      = ["positive", "negative", "neutral"]
LABEL2ID    = {l: i for i, l in enumerate(LABELS)}
ID2LABEL    = {i: l for l, i in LABEL2ID.items()}
EVAL_DIR    = Path("evaluation")
CKPT_DIR    = Path("models/saved/xlmr_hinglish")
BASELINE_P  = Path("models/saved/baseline_tfidf_lr.pkl")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_data():
    test_p = Path("data/processed/preprocessed_test.csv")
    dev_p  = Path("data/processed/preprocessed_data.csv")
    if test_p.exists():
        df = pd.read_csv(test_p)
    else:
        df = pd.read_csv(dev_p)
        _, df = train_test_split(df, test_size=0.2, random_state=42,
                                 stratify=df["sentiment"])
    return df["clean_text"].fillna("").tolist(), df["sentiment"].tolist()


def baseline_predict(texts):
    model = joblib.load(BASELINE_PATH)
    return model.predict(texts).tolist()


def xlmr_predict(texts):
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT_DIR))
    model     = AutoModelForSequenceClassification.from_pretrained(str(CKPT_DIR)).to(DEVICE)
    model.eval()
    preds = []
    BATCH = 32
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        enc   = tokenizer(batch, truncation=True, padding="max_length",
                          max_length=128, return_tensors="pt")
        enc   = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return [ID2LABEL[p] for p in preds]


if __name__ == "__main__":
    texts, y_true = load_test_data()

    rows = []
    for model_name, pred_fn in [
        ("Baseline (TF-IDF + LR)", lambda t: joblib.load(BASELINE_P).predict(t).tolist()),
        ("XLM-RoBERTa", xlmr_predict),
    ]:
        y_pred = pred_fn(texts)
        acc    = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, labels=LABELS,
                                       output_dict=True, zero_division=0)
        for cls in LABELS + ["macro avg"]:
            r = report.get(cls, {})
            rows.append({
                "model": model_name,
                "class": cls,
                "precision": round(r.get("precision", 0), 3),
                "recall":    round(r.get("recall", 0), 3),
                "f1":        round(r.get("f1-score", 0), 3),
                "accuracy":  round(acc, 4) if cls == "macro avg" else "",
            })

    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    out = EVAL_DIR / "model_comparison.csv"
    tbl.to_csv(out, index=False)
    print(f"\n✅  Comparison saved → {out}")
