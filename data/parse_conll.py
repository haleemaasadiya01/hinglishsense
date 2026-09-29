"""
parse_conll.py
--------------
Parse SemEval-2020 SentiMix CoNLL files into a clean pandas DataFrame.

CoNLL format recap
  <blank line>          — tweet block separator
  meta <tweet_id> <label>
  word  <lang_tag>      — one per line; lang_tag ∈ {Eng, Hin, O}
  word  <lang_tag>
  ...

Output columns:
  tweet_id      – string
  full_text     – words joined with a single space
  sentiment     – positive | negative | neutral
  lang_tags     – list[str]  (per-word language tags)
"""

import re
import sys
from pathlib import Path

import pandas as pd


VALID_LABELS = {"positive", "negative", "neutral"}


def parse_conll_file(filepath: str | Path) -> pd.DataFrame:
    """Return a DataFrame with one row per tweet."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Cannot find file: {filepath}")

    records = []
    tweet_id = None
    label = None
    words: list[str] = []
    tags: list[str] = []

    def flush(tid, lbl, ws, ts):
        if tid is not None and ws:
            records.append(
                {
                    "tweet_id": tid,
                    "full_text": " ".join(ws),
                    "sentiment": lbl,
                    "lang_tags": ts[:],
                }
            )

    with filepath.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

            # ── blank line → flush current tweet ─────────────────────────
            if line.strip() == "":
                flush(tweet_id, label, words, tags)
                tweet_id = label = None
                words, tags = [], []
                continue

            parts = line.split()

            # ── meta line ─────────────────────────────────────────────────
            if parts[0].lower() == "meta":
                # guard: flush any previous tweet that had no trailing blank
                flush(tweet_id, label, words, tags)
                words, tags = [], []

                if len(parts) < 3:
                    raise ValueError(f"Malformed meta line: {line!r}")

                tweet_id = parts[1]
                raw_label = parts[2].lower()
                if raw_label not in VALID_LABELS:
                    raise ValueError(
                        f"Unknown label {raw_label!r} on line: {line!r}"
                    )
                label = raw_label
                continue

            # ── word line ─────────────────────────────────────────────────
            if len(parts) == 2:
                word, tag = parts
                words.append(word)
                tags.append(tag)
            else:
                # edge-case: word may contain spaces — treat last token as tag
                words.append(" ".join(parts[:-1]))
                tags.append(parts[-1])

    # flush last tweet if file does not end with a blank line
    flush(tweet_id, label, words, tags)

    df = pd.DataFrame(records, columns=["tweet_id", "full_text", "sentiment", "lang_tags"])
    return df


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Hinglish_dev_3k_split_conll.txt")

    print(f"\nLoading: {data_file}\n" + "-"*55)
    df = parse_conll_file(data_file)

    print(f"Shape            : {df.shape}")
    print(f"Columns          : {list(df.columns)}\n")

    print("-- Sample rows " + "-"*40)
    print(df[["tweet_id", "sentiment", "full_text"]].head(5).to_string(index=False))
    print()

    print("-- Label distribution " + "-"*33)
    counts = df["sentiment"].value_counts()
    pct = df["sentiment"].value_counts(normalize=True).mul(100).round(1)
    dist = pd.concat([counts, pct], axis=1)
    dist.columns = ["count", "pct %"]
    print(dist.to_string())
    print()

    # save processed file next to source
    out = Path("data/processed/parsed_data.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved to {out}")
