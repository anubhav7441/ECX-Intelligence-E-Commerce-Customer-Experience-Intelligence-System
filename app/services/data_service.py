"""Loads the pre-computed analysis results once and serves them to the routes. No numbers are hard-coded here."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def clean(obj):
    """Make nested structures JSON-safe (NaN -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def df_records(df):
    return clean(df.to_dict(orient="records"))


@lru_cache(maxsize=1)
def bi():
    return _json(ROOT / "research" / "results" / "bi_results.json")


@lru_cache(maxsize=1)
def stats():
    return _json(ROOT / "research" / "statistical_analysis" / "results" / "stats_results.json")


@lru_cache(maxsize=1)
def nlp():
    return _json(ROOT / "ml" / "sentiment_analysis" / "results" / "nlp_results.json")


@lru_cache(maxsize=1)
def nlp_eval():
    return _json(ROOT / "ml" / "evaluation" / "nlp_eval_results.json")


@lru_cache(maxsize=1)
def ml():
    return _json(ROOT / "ml" / "prediction" / "results" / "ml_results.json")


@lru_cache(maxsize=1)
def shap_global():
    return {"S1_R": _json(ROOT / "ml" / "explainability" / "results" / "shap_S1_R.json"),
            "S0_R": _json(ROOT / "ml" / "explainability" / "results" / "shap_S0_R.json")}


@lru_cache(maxsize=1)
def prep_report():
    return _json(P / "preprocessing_report.json")


@lru_cache(maxsize=None)
def table(name):
    return pd.read_csv(P / "bi" / f"{name}.csv")


def monthly():
    return df_records(table("monthly"))


def sellers(segment=None, limit=None):
    d = table("sellers").sort_values("revenue", ascending=False)
    if segment:
        d = d[d.segment == segment]
    d = d.assign(seller_short=d.primary_seller_id.str[:8] + "...")
    return df_records(d.head(limit) if limit else d)


def categories(segment=None):
    d = table("categories").sort_values("revenue", ascending=False)
    if segment:
        d = d[d.segment == segment]
    return df_records(d)


def states():
    return df_records(table("states"))


def regions():
    return df_records(table("regions"))


def db_status():
    """Check the SQLite database through SQLAlchemy (used by the Data Management page and /api/health)."""
    from flask import current_app
    try:
        eng = create_engine(current_app.config["DATABASE_URL"])
        with eng.connect() as c:
            tables = ["customers", "sellers", "products", "orders", "order_items", "order_payments", "order_reviews", "geolocation_zip"]
            counts = {t: c.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar() for t in tables}  # nosec B608
        return {"ok": True, "tables": counts}
    except Exception as e:  # database missing / locked / corrupt
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300], "tables": {}}
