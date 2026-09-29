"""
preprocessing/pipeline.py
--------------------------
Hinglish text normalisation + tokenisation pipeline.

Design choices are documented inline.
"""

import ast
import re
import string
from pathlib import Path

import pandas as pd

# ── 1. Transliteration normalisation map ─────────────────────────────────────
# Hinglish has no standard spelling: the same Hindi word appears in many
# romanisation forms.  We collapse the most common variants to a canonical
# form so TF-IDF (and the transformer tokeniser) see the same token.
TRANSLIT_NORM: dict[str, str] = {
    # "good" synonyms
    "acha": "achha",
    "accha": "achha",
    "acha": "achha",
    "acha": "achha",
    "achi": "achhi",
    "acchi": "achhi",
    # "bad / rubbish"
    "bakwas": "bakwaas",
    "bakvas": "bakwaas",
    "bakwaas": "bakwaas",
    # "cool / great"
    "mast": "mast",
    # "very"
    "bohot": "bahut",
    "bhaut": "bahut",
    "bahut": "bahut",
    "bht": "bahut",
    # "what"
    "kya": "kya",
    "kyaa": "kya",
    # negation
    "nahi": "nahi",
    "nahin": "nahi",
    "nhi": "nahi",
    "nai": "nahi",
    # "heart / feeling"
    "dil": "dil",
    # elongation of "yaar" (dude)
    "yaar": "yaar",
    "yar": "yaar",
    # "like" / "love"
    "pyaar": "pyaar",
    "pyar": "pyaar",
}


# ── 2. Emoji / punctuation tables ────────────────────────────────────────────
# Social-media text is full of emoji.  We drop them because
#  a) the baseline TF-IDF model can't use them,
#  b) the transformer handles them badly (they are outside its vocab).
# A production system could map emoji → sentiment descriptors instead.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

# We keep hashtag text but strip the '#' — useful signal, not noise.
_HASHTAG = re.compile(r"#(\w+)")
_MENTION = re.compile(r"@\w+")
_URL     = re.compile(r"https?://\S+|www\.\S+")
_MULTI_SPACE = re.compile(r"\s{2,}")


# ── 3. Core normaliser ────────────────────────────────────────────────────────

def normalise_text(text: str) -> str:
    """
    Steps (order matters):
      1. Lower-case  → uniform surface form
      2. Drop URLs   → no signal for sentiment
      3. Drop @mentions → same reason
      4. Strip # from hashtags but keep the word
      5. Remove emoji
      6. Remove punctuation  → reduces sparsity in TF-IDF
      7. Collapse whitespace
      8. Transliteration normalisation  → must come AFTER lower-casing
    """
    text = text.lower()
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _HASHTAG.sub(r"\1", text)
    text = _EMOJI_PATTERN.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _MULTI_SPACE.sub(" ", text).strip()

    # word-level transliteration normalisation
    tokens = text.split()
    tokens = [TRANSLIT_NORM.get(t, t) for t in tokens]
    return " ".join(tokens)


# ── 4. Language-tag-aware feature helpers ─────────────────────────────────────
# The dataset already provides per-word language tags (Eng / Hin / O).
# Instead of re-running language ID, we expose helpers that split a tweet
# into its English and Hindi sub-strings.  These can be used as extra
# features (e.g. TF-IDF on the Hindi sub-string separately).

def split_by_language(words: list[str], tags: list[str]) -> dict[str, str]:
    """
    Returns {'eng': '...', 'hin': '...', 'other': '...'}
    """
    eng, hin, other = [], [], []
    for w, t in zip(words, tags):
        if t == "Eng":
            eng.append(w)
        elif t == "Hin":
            hin.append(w)
        else:
            other.append(w)
    return {"eng": " ".join(eng), "hin": " ".join(hin), "other": " ".join(other)}


def lang_mix_ratio(tags: list[str]) -> float:
    """Fraction of Hinglish content (Hin tokens / total)."""
    if not tags:
        return 0.0
    hin_count = sum(1 for t in tags if t == "Hin")
    return hin_count / len(tags)


# ── 5. Full pipeline ──────────────────────────────────────────────────────────

def build_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the parsed DataFrame (from parse_conll.py) and returns it
    with added columns:
      clean_text   – normalised full text  (for TF-IDF / transformer)
      lang_mix     – fraction of Hindi tokens  (auxiliary feature)
    """
    df = df.copy()

    # parse lang_tags if loaded from CSV (stored as string repr of list)
    if df["lang_tags"].dtype == object and isinstance(df["lang_tags"].iloc[0], str):
        df["lang_tags"] = df["lang_tags"].apply(ast.literal_eval)

    df["clean_text"] = df["full_text"].apply(normalise_text)
    df["lang_mix"] = df["lang_tags"].apply(lang_mix_ratio)

    return df


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/parsed_data.csv")
    print(f"Loading {src} ...")
    df = pd.read_csv(src)
    df = build_pipeline(df)

    out = Path("data/processed/preprocessed_data.csv")
    df[["tweet_id", "sentiment", "full_text", "clean_text", "lang_mix"]].to_csv(out, index=False)
    print(f"Saved {len(df)} rows -> {out}")
    print(df[["full_text", "clean_text", "lang_mix"]].head(5).to_string(index=False))
