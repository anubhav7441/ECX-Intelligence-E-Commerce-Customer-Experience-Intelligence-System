"""Dataset registry & isolation. Every dataset lives in its own folder  data/datasets/<dataset_id>/ :
   meta.json (source, adapter, mapping, dimensions)  history.json (processing log)  data.parquet (cleaned working copy)
   results.json (analysis results)  report/ (PDF + CSV exports).
Nothing is ever shared between datasets. The built-in 'olist' entry is READ-ONLY and only points at existing v1 files."""
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID_RE = re.compile(r"^[a-z0-9_]{3,40}$")
OLIST_ID = "olist"
MAX_ROWS = 2_000_000


def base_dir():
    p = Path(os.environ.get("MULTIDS_DIR") or ROOT / "data" / "datasets")
    p.mkdir(parents=True, exist_ok=True)
    return p


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def valid_id(did):
    return isinstance(did, str) and bool(ID_RE.match(did))


def dpath(did):
    if not valid_id(did):
        raise ValueError("invalid dataset id")
    return base_dir() / did


def new_id(prefix="ds"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _w(p, obj):
    p.write_text(json.dumps(obj, indent=2, default=str))


def _r(p, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def create(did, meta, df, history):
    d = dpath(did)
    if d.exists():
        raise FileExistsError(f"dataset {did} already exists")
    (d / "report").mkdir(parents=True)
    if len(df) > MAX_ROWS:
        raise ValueError(f"dataset has more than {MAX_ROWS:,} rows")
    # parquet needs homogeneous object columns; stringify mixed object columns
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v if (isinstance(v, str) or pd.isna(v)) else str(v))
    df.to_parquet(d / "data.parquet", index=False)
    meta = {**meta, "dataset_id": did, "created_at": now(), "n_rows": int(len(df)), "n_cols": int(df.shape[1]),
            "columns": [str(c) for c in df.columns], "mapping": {}, "mapping_confirmed": False, "read_only": False}
    _w(d / "meta.json", meta)
    _w(d / "history.json", history)
    _w(d / "results.json", {})
    return meta


def _hist_path(did):
    return base_dir() / "_olist_history.json" if did == OLIST_ID else dpath(did) / "history.json"


def log(did, step, detail):
    p = _hist_path(did)
    h = _r(p, [])
    h.append({"time": now(), "step": step, "detail": detail})
    _w(p, h)


def meta(did):
    if did == OLIST_ID:
        return olist_meta()
    return _r(dpath(did) / "meta.json")


def save_meta(did, m):
    if did == OLIST_ID:
        raise PermissionError("The Olist dataset is read-only here.")
    _w(dpath(did) / "meta.json", m)


def history(did):
    if did == OLIST_ID:
        return [{"time": None, "step": "read-only registration", "detail": "Uses existing v1 file orders_with_history.parquet; v1 workflow and models are not modified."}] + _r(_hist_path(did), [])
    return _r(dpath(did) / "history.json", [])


def results(did):
    return _r(dpath(did) / "results.json", {}) if did != OLIST_ID else _r(base_dir() / "_olist_results.json", {})


def save_results(did, res):
    _w(dpath(did) / "results.json" if did != OLIST_ID else base_dir() / "_olist_results.json", res)


def olist_meta():
    p = ROOT / "data" / "processed" / "orders_with_history.parquet"
    return {"dataset_id": OLIST_ID, "name": "Olist Brazilian E-Commerce (Dataset A)", "adapter": "olist", "read_only": True,
            "available": p.exists(), "source": "Kaggle: Olist Brazilian E-Commerce Public Dataset (CC BY-NC-SA 4.0)",
            "source_url": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
            "description": "Primary research dataset. Full research workflow = the original ECX-Intelligence pages. This entry only allows the generic multi-dataset report on the processed order table; it never changes v1 results or models.",
            "sample_note": "The file used is the v1 modelling table (orders_with_history.parquet): 95,446 of the 95,824 analysed orders, because v1 removed customers that appear in both training and test periods. Figures can therefore differ slightly from the v1 report (e.g. 95,824 orders).",
            "mapping": OLIST_MAPPING, "mapping_confirmed": True}


OLIST_MAPPING = {"order_id": "order_id", "customer_id": "customer_unique_id", "product_id": "primary_product_id", "date": "order_purchase_timestamp",
                 "revenue": "items_price", "region": "customer_state", "rating": "review_score", "review_text": "review_comment_message",
                 "seller_id": "primary_seller_id", "delivery_days": "delivery_days", "delay_days": "delay_days", "category": "primary_category"}


def report_dir(did):
    d = (base_dir() / "_olist_report") if did == OLIST_ID else dpath(did) / "report"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_df(did):
    if did == OLIST_ID:
        p = ROOT / "data" / "processed" / "orders_with_history.parquet"
        if not p.exists():
            raise FileNotFoundError("Olist processed data not found (extract the data ZIP).")
        return pd.read_parquet(p)
    p = dpath(did) / "data.parquet"
    if not p.exists():
        raise FileNotFoundError("dataset not found")
    return pd.read_parquet(p)


def list_all():
    out = [olist_meta()]
    for d in sorted(base_dir().iterdir()) if base_dir().exists() else []:
        if d.is_dir() and (d / "meta.json").exists():
            m = _r(d / "meta.json")
            out.append({k: m.get(k) for k in ("dataset_id", "name", "adapter", "n_rows", "n_cols", "created_at", "mapping_confirmed", "source", "read_only", "owner_id")})
    return out


def delete(did):
    if did == OLIST_ID:
        raise PermissionError("The Olist dataset cannot be deleted here.")
    d = dpath(did)
    if not d.exists():
        raise FileNotFoundError("dataset not found")
    shutil.rmtree(d)
