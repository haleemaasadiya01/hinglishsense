"""
models/transformer_model.py
----------------------------
Fine-tune XLM-RoBERTa (xlm-roberta-base) for 3-class Hinglish sentiment.

Why XLM-RoBERTa?
  • Pre-trained on 100 languages including Hindi and English — it already
    "knows" both sides of code-mixed text.
  • The BPE tokeniser handles unseen transliterated words gracefully by
    falling back to sub-word pieces, unlike word-level models.
  • IndicBERT is an alternative (smaller, India-focused), but XLM-R is
    more widely benchmarked and easier to compare against the literature.

Usage:
  python models/transformer_model.py            # fine-tune + evaluate
  python models/transformer_model.py --eval-only # evaluate saved checkpoint
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
# Switch to a lightweight distilled model for fast CPU training.
# distilbert-base-multilingual-cased is 66M params (vs 278M for xlm-roberta-base)
# and trains ~4x faster with only a small accuracy drop on code-mixed text.
# Change MODEL_NAME back to "xlm-roberta-base" if you have a GPU.
MODEL_NAME  = "distilbert-base-multilingual-cased"
LABELS      = ["positive", "negative", "neutral"]
LABEL2ID    = {l: i for i, l in enumerate(LABELS)}
ID2LABEL    = {i: l for l, i in LABEL2ID.items()}
MAX_LEN     = 96          # shorter sequences = faster; tweets rarely exceed 96 tokens
BATCH_SIZE  = 32          # larger batches = fewer steps = faster on CPU
EPOCHS      = 4           # more epochs to compensate for smaller model capacity
LR          = 3e-5
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR    = Path("models/saved/multilingual_distilbert")
EVAL_DIR    = Path("evaluation")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

class HinglishDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ── Data loading ──────────────────────────────────────────────────────────────

def load_splits():
    train_p = Path("data/processed/preprocessed_train.csv")
    test_p  = Path("data/processed/preprocessed_test.csv")
    dev_p   = Path("data/processed/preprocessed_data.csv")

    if train_p.exists() and test_p.exists():
        tr = pd.read_csv(train_p)
        te = pd.read_csv(test_p)
        X_tr, y_tr = tr["clean_text"].fillna("").tolist(), tr["sentiment"].map(LABEL2ID).tolist()
        X_te, y_te = te["clean_text"].fillna("").tolist(), te["sentiment"].map(LABEL2ID).tolist()
        return X_tr, X_te, y_tr, y_te

    df = pd.read_csv(dev_p)
    X  = df["clean_text"].fillna("").tolist()
    y  = df["sentiment"].map(LABEL2ID).tolist()
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, truths = [], []
    for batch in loader:
        batch   = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        logits  = outputs.logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        truths.extend(batch["labels"].cpu().tolist())
    return preds, truths


# ── Compare baseline vs transformer ──────────────────────────────────────────

def comparison_table(
    bl_report: dict,
    xr_report: dict,
    bl_acc: float,
    xr_acc: float,
) -> pd.DataFrame:
    rows = []
    for label in LABELS + ["macro avg"]:
        bl = bl_report.get(label, {})
        xr = xr_report.get(label, {})
        rows.append(
            {
                "class": label,
                "baseline_precision": round(bl.get("precision", 0), 3),
                "baseline_recall":    round(bl.get("recall", 0), 3),
                "baseline_f1":        round(bl.get("f1-score", 0), 3),
                "xlmr_precision":     round(xr.get("precision", 0), 3),
                "xlmr_recall":        round(xr.get("recall", 0), 3),
                "xlmr_f1":            round(xr.get("f1-score", 0), 3),
            }
        )
    tbl = pd.DataFrame(rows)
    print(f"\n{'='*70}")
    print("  MODEL COMPARISON")
    print(f"  Baseline accuracy : {bl_acc:.4f}")
    print(f"  XLM-R accuracy    : {xr_acc:.4f}")
    print(f"{'='*70}")
    print(tbl.to_string(index=False))
    print(f"{'='*70}\n")
    tbl.to_csv(EVAL_DIR / "model_comparison.csv", index=False)
    return tbl


# ── Main ──────────────────────────────────────────────────────────────────────

def main(eval_only: bool = False):
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")

    X_tr, X_te, y_tr, y_te = load_splits()
    print(f"Train: {len(X_tr)}  |  Test: {len(X_te)}")

    print(f"\nLoading tokeniser: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    test_ds  = HinglishDataset(X_te, y_te, tokenizer)
    test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    if eval_only and CKPT_DIR.exists():
        print("Loading saved checkpoint for evaluation ...")
        model = AutoModelForSequenceClassification.from_pretrained(str(CKPT_DIR))
    else:
        print(f"Loading model: {MODEL_NAME} ...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        train_ds = HinglishDataset(X_tr, y_tr, tokenizer)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        total_steps = len(train_dl) * EPOCHS
        scheduler   = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        model.to(DEVICE)
        print(f"\nFine-tuning for {EPOCHS} epochs ...")
        for epoch in range(1, EPOCHS + 1):
            loss = train(model, train_dl, optimizer, scheduler)
            preds, truths = evaluate(model, test_dl)
            acc = accuracy_score(truths, preds)
            print(f"  Epoch {epoch}/{EPOCHS}  loss={loss:.4f}  val_acc={acc:.4f}")

        model.save_pretrained(str(CKPT_DIR))
        tokenizer.save_pretrained(str(CKPT_DIR))
        print(f"\nCheckpoint saved -> {CKPT_DIR}")

    model.to(DEVICE)
    preds, truths = evaluate(model, test_dl)
    xr_acc    = accuracy_score(truths, preds)
    pred_lbls = [ID2LABEL[p] for p in preds]
    true_lbls = [ID2LABEL[t] for t in truths]

    print("\n-- XLM-RoBERTa Evaluation " + "-"*29)
    print(f"Accuracy: {xr_acc:.4f}")
    print(classification_report(true_lbls, pred_lbls, labels=LABELS, zero_division=0))

    xr_report = classification_report(
        true_lbls, pred_lbls, labels=LABELS, zero_division=0, output_dict=True
    )

    # load baseline metrics if they exist for comparison
    bl_report_path = EVAL_DIR / "baseline_report.csv"
    if bl_report_path.exists():
        bl_df  = pd.read_csv(bl_report_path, index_col=0)
        bl_acc = float(bl_df.loc["accuracy", "f1-score"]) if "accuracy" in bl_df.index else 0.0
        bl_report = bl_df.to_dict(orient="index")
        comparison_table(bl_report, xr_report, bl_acc, xr_acc)
    else:
        print("\n(Run baseline.py first to see the comparison table.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    main(eval_only=args.eval_only)
