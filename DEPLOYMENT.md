# Deploying HinglishSense to HuggingFace Spaces

## What you need
- A free account at https://huggingface.co
- `git` installed
- `git-lfs` installed (for large model files)
- The fine-tuned XLM-RoBERTa checkpoint under `models/saved/xlmr_hinglish/`

---

## Step 1 – Install git-lfs (once per machine)

```bash
# Windows (winget)
winget install Git.LFS

# Then enable it globally
git lfs install
```

---

## Step 2 – Create a new Space on HuggingFace

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `hinglishsense`
   - **License**: MIT
   - **SDK**: Streamlit
   - **SDK version**: 1.32.0
3. Click **Create Space** — this creates a git repo at
   `https://huggingface.co/spaces/<YOUR_USERNAME>/hinglishsense`

---

## Step 3 – Prepare your local repo

```bash
# From your project root:
git init
git lfs track "*.pkl"
git lfs track "*.bin"
git lfs track "*.pt"
git lfs track "*.safetensors"
```

The `.gitattributes` file is created automatically by git-lfs.

---

## Step 4 – requirements.txt (already in place)

HuggingFace Spaces reads `requirements.txt` directly. The file already
pinned in this repo includes:

```
torch>=2.1.0
transformers>=4.37.0
sentencepiece>=0.1.99
streamlit>=1.32.0
scikit-learn>=1.3.0
joblib>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```

> **Tip on Space resources:** The free tier has ~16 GB RAM and no GPU.
> XLM-RoBERTa runs fine on CPU for inference.
> If the Space times out on first load, bump the "Sleep time" in Settings.

---

## Step 5 – Connect and push

```bash
# Add the HF remote (replace YOUR_USERNAME)
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/hinglishsense

# Add all files
git add .
git commit -m "initial: HinglishSense Streamlit app"

# Push — enter your HF token when prompted (or set it in HTTPS credentials)
git push hf main
```

> **Authentication:** HuggingFace uses your **User Access Token** (not your password).
> Create one at https://huggingface.co/settings/tokens (role: write).
> Either use it as the HTTPS password, or run:
> ```bash
> huggingface-cli login
> ```

---

## Step 6 – Verify the Space is live

- Open `https://huggingface.co/spaces/YOUR_USERNAME/hinglishsense`
- First build takes ~3–5 minutes (pip install)
- Once green, the public URL is permanent and shareable

---

## File structure that HF Spaces requires

```
repo root/
├── app.py                  <-- entry point (must be named app.py)
├── requirements.txt
├── README.md               <-- YAML front-matter configures the Space card
├── app/
│   └── app.py              <-- real Streamlit code
├── preprocessing/
│   └── pipeline.py
├── models/
│   └── saved/
│       ├── baseline_tfidf_lr.pkl
│       └── xlmr_hinglish/
│           ├── config.json
│           ├── tokenizer_config.json
│           ├── sentencepiece.bpe.model
│           └── model.safetensors   (tracked by git-lfs)
```

The `README.md` already has the correct YAML front-matter:

```yaml
---
title: HinglishSense
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
---
```

---

## Updating the Space after changes

```bash
git add .
git commit -m "update: describe your change"
git push hf main
```

HF Spaces rebuilds automatically on every push.

---

## Secrets / environment variables

If you ever need API keys, add them in
**Space Settings → Variables and Secrets** — never commit them to git.
Access them in Python via `os.environ["MY_SECRET"]`.
