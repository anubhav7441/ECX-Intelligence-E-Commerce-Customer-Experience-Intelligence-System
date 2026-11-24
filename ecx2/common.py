"""Shared paths and cached data loaders for ECX-Intelligence 2.0 (additive layer; v1 code is untouched)."""
import json
from functools import lru_cache
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V2_RES = ROOT / "v2_results"
V2_MODELS = ROOT / "models" / "v2"
V2_DATA = ROOT / "data" / "processed" / "v2"
for _p in (V2_RES, V2_MODELS, V2_DATA):
    _p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def orders():
    """Analysed orders (delivered, with a review) incl. leakage-safe history columns; review text included."""
    p = ROOT / "data" / "processed" / "orders_with_history.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run the v1 pipeline (scripts/run_all.py) or extract the data ZIP.")
    d = pd.read_parquet(p)
    d["purchase_ym"] = pd.to_datetime(d["order_purchase_timestamp"]).dt.to_period("M").astype(str)
    return d


@lru_cache(maxsize=1)
def nlp_flags():
    p = ROOT / "data" / "processed" / "review_nlp.parquet"
    return pd.read_parquet(p) if p.exists() else None


def load_json(name, default=None):
    p = V2_RES / name
    return json.loads(p.read_text()) if p.exists() else default


def save_json(name, obj):
    (V2_RES / name).write_text(json.dumps(obj, indent=2, default=str))


def clean(o):
    """Make numpy/pandas objects JSON-safe (NaN -> None)."""
    import math
    import numpy as np
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(float(o)) or math.isinf(float(o))) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o)
    return o
