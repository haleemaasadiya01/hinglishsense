"""
app/app.py  (also importable as  app.py  from repo root for HF Spaces)
-----------------------------------------------------------------------
HinglishSense – Streamlit sentiment analysis web app.

Features:
  • Type any Hinglish sentence → get sentiment + confidence
  • Toggle between Baseline (TF-IDF + LR) and Transformer (XLM-RoBERTa)
  • Confidence bar per class
  • Clean, minimal UI
"""

import sys
from pathlib import Path

# make project root importable when running from app/ sub-folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocessing.pipeline import normalise_text

# ── Constants ─────────────────────────────────────────────────────────────────
LABELS      = ["positive", "negative", "neutral"]
LABEL_EMOJI = {"positive": "😊", "negative": "😞", "neutral": "😐"}
LABEL_COLOR = {"positive": "#22c55e", "negative": "#ef4444", "neutral": "#f59e0b"}

BASELINE_PATH = Path("models/saved/baseline_tfidf_lr.pkl")
XLMR_PATH     = Path("models/saved/multilingual_distilbert")
MAX_LEN       = 128


# ── Model loaders (cached) ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading baseline model …")
def load_baseline():
    if not BASELINE_PATH.exists():
        return None
    return joblib.load(BASELINE_PATH)


@st.cache_resource(show_spinner="Loading XLM-RoBERTa … (first run may take a moment)")
def load_transformer():
    if not XLMR_PATH.exists():
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(str(XLMR_PATH))
    model     = AutoModelForSequenceClassification.from_pretrained(str(XLMR_PATH))
    model.eval()
    return tokenizer, model


# ── Prediction helpers ────────────────────────────────────────────────────────

def predict_baseline(text: str) -> tuple[str, dict[str, float]]:
    model = load_baseline()
    if model is None:
        st.error("Baseline model not found. Run `python models/baseline.py` first.")
        st.stop()
    clean = normalise_text(text)
    proba = model.predict_proba([clean])[0]
    classes = model.classes_
    conf   = {c: float(p) for c, p in zip(classes, proba)}
    pred   = max(conf, key=conf.get)
    return pred, conf


def predict_transformer(text: str) -> tuple[str, dict[str, float]]:
    tokenizer, model = load_transformer()
    if model is None:
        st.error("XLM-RoBERTa checkpoint not found. Run `python models/transformer_model.py` first.")
        st.stop()
    clean = normalise_text(text)
    inputs = tokenizer(
        clean,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    proba  = torch.softmax(logits, dim=-1)[0].tolist()
    id2lbl = model.config.id2label
    conf   = {id2lbl[i]: float(p) for i, p in enumerate(proba)}
    pred   = max(conf, key=conf.get)
    return pred, conf


# ── UI ────────────────────────────────────────────────────────────────────────

def render_confidence_bars(conf: dict[str, float]):
    for label in LABELS:
        score = conf.get(label, 0.0)
        color = LABEL_COLOR[label]
        bar_html = f"""
        <div style="margin-bottom:6px;">
          <div style="display:flex;justify-content:space-between;font-size:13px;color:#374151;">
            <span>{LABEL_EMOJI[label]} {label.capitalize()}</span>
            <span>{score*100:.1f}%</span>
          </div>
          <div style="background:#e5e7eb;border-radius:4px;height:10px;">
            <div style="width:{score*100:.1f}%;background:{color};
                        border-radius:4px;height:10px;transition:width 0.4s;"></div>
          </div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="HinglishSense",
        page_icon="🌐",
        layout="centered",
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style='text-align:center;font-size:2.2rem;margin-bottom:0;'>
          🌐 HinglishSense
        </h1>
        <p style='text-align:center;color:#6b7280;font-size:1rem;margin-top:4px;'>
          Sentiment Analysis for Code-Mixed Hindi-English Text
        </p>
        <hr style='border:1px solid #e5e7eb;margin:12px 0 24px;'>
        """,
        unsafe_allow_html=True,
    )

    # ── Model selector ────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col2:
        model_choice = st.radio(
            "Model",
            ["Transformer (XLM-R)", "Baseline (TF-IDF + LR)"],
            index=0,
            help="XLM-RoBERTa is more accurate; the baseline is faster.",
        )

    # ── Input area ────────────────────────────────────────────────────────────
    with col1:
        user_text = st.text_area(
            "Type a Hinglish sentence:",
            placeholder="e.g.  yaar ye movie bahut mast thi, loved it!",
            height=110,
        )

    # ── Predict ───────────────────────────────────────────────────────────────
    predict_btn = st.button("Analyse Sentiment ✨", use_container_width=True, type="primary")

    if predict_btn and user_text.strip():
        with st.spinner("Analysing …"):
            if model_choice.startswith("Transformer"):
                pred, conf = predict_transformer(user_text)
            else:
                pred, conf = predict_baseline(user_text)

        # ── Result card ───────────────────────────────────────────────────────
        color = LABEL_COLOR[pred]
        st.markdown(
            f"""
            <div style="border:2px solid {color};border-radius:12px;
                        padding:18px 24px;margin:16px 0;background:#fafafa;">
              <div style="font-size:2rem;font-weight:700;color:{color};text-align:center;">
                {LABEL_EMOJI[pred]}&nbsp;{pred.upper()}
              </div>
              <div style="text-align:center;color:#6b7280;font-size:0.85rem;margin-top:4px;">
                via {model_choice}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Confidence scores**")
        render_confidence_bars(conf)

    elif predict_btn and not user_text.strip():
        st.warning("Please enter some text first.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center;color:#9ca3af;font-size:0.75rem;
                    margin-top:48px;border-top:1px solid #e5e7eb;padding-top:12px;">
          HinglishSense · SemEval-2020 SentiMix · Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
