"""Loads the trained TF-IDF + Logistic Regression sentiment model once."""
from functools import lru_cache
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def sentiment_model():
    return joblib.load(ROOT / "models" / "sentiment_tfidf_lr.joblib")
