"""
app.py  –  HuggingFace Spaces entry point
This is a thin shim that just imports and runs the real app in app/app.py.
HF Spaces requires the Streamlit entry-point to be a file named `app.py`
in the repository root.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.app import main

if __name__ == "__main__":
    main()
