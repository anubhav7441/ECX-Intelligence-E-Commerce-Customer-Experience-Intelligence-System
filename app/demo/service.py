"""Guest demo logic. Everything here reads ONLY the capped sample file; the row cap and usage limits are enforced here (backend), not in the browser."""
from functools import lru_cache
import numpy as np
import pandas as pd
from flask import current_app, request, session
from app.auth import db as adb

HARD_MAX_ROWS = 2000          # even an admin cannot raise the demo cap above this


def limits():
    c = current_app.config
    def g(key, default, lo, hi):
        try:
            v = int(adb.get_setting(key, default))
        except Exception:
            v = default
        return max(lo, min(hi, v))
    return {"rows": g("demo_row_limit", c["DEMO_ROW_LIMIT"], 50, HARD_MAX_ROWS),
            "predictions": g("demo_max_predictions", c["DEMO_MAX_PREDICTIONS"], 0, 100),
            "investigations": g("demo_max_investigations", c["DEMO_MAX_INVESTIGATIONS"], 0, 50)}


@lru_cache(maxsize=2)
def _load(path):
    return pd.read_parquet(path)


def sample():
    df = _load(str(current_app.config["DEMO_SAMPLE_FILE"]))
    return df.head(limits()["rows"]).copy()               # <- the backend cap


# usage counters: signed session cookie + per-IP memory counter (so clearing cookies does not trivially reset it)
_ip_use = {}


def used(kind):
    return max(int(session.get(f"demo_{kind}", 0)), _ip_use.get((request.remote_addr, kind), 0))


def use(kind):
    session[f"demo_{kind}"] = int(session.get(f"demo_{kind}", 0)) + 1
    k = (request.remote_addr, kind)
    _ip_use[k] = _ip_use.get(k, 0) + 1


def reset_state():
    _ip_use.clear()


def summary():
    d = sample()
    d["order_purchase_timestamp"] = pd.to_datetime(d["order_purchase_timestamp"])
    m = d.groupby(d.order_purchase_timestamp.dt.to_period("M").astype(str)).agg(orders=("order_id", "size"), late=("is_late", "mean")).reset_index()
    st = d.groupby("customer_state").agg(orders=("order_id", "size"), late=("is_late", "mean")).sort_values("orders", ascending=False).head(8).reset_index()
    sc = d.review_score.value_counts().sort_index()
    cat = d.groupby("primary_category").agg(orders=("order_id", "size"), score=("review_score", "mean")).sort_values("orders", ascending=False).head(8).reset_index()
    L = limits()
    return {"n_rows": int(len(d)), "row_limit": L["rows"],
            "kpis": {"orders": int(len(d)), "late_rate": float(d.is_late.mean()), "mean_review": float(d.review_score.mean()),
                     "dissatisfied_rate": float(d.dissatisfied.mean()), "mean_delivery_days": float(d.delivery_days.mean())},
            "monthly": m.to_dict("records"), "states": st.to_dict("records"), "scores": {int(k): int(v) for k, v in sc.items()},
            "categories": cat.to_dict("records"),
            "usage": {"predictions": {"used": used("predictions"), "max": L["predictions"]}, "investigations": {"used": used("investigations"), "max": L["investigations"]}},
            "orders_preview": d[["order_id", "customer_state", "primary_category", "review_score", "is_late"]].head(25).assign(order_id=lambda x: x.order_id.str[:12]).to_dict("records"),
            "note": f"Demo sample: {len(d)} real Olist orders (random sample, fixed seed). Not the full dataset."}


def sample_order_ids():
    return set(sample().order_id)


def seller_investigation(seller_prefix):
    """Limited investigation: descriptive profile of one seller INSIDE the sample, with an honest small-sample warning."""
    d = sample()
    t = (seller_prefix or "").strip().lower()
    if len(t) < 6:
        return {"ok": False, "error": "Enter at least 6 characters of a seller id from the sample."}
    ids = [i for i in d.primary_seller_id.dropna().unique() if i.startswith(t)]
    if len(ids) != 1:
        return {"ok": False, "error": "No unique seller in the demo sample matches that id. Pick one from the list."}
    x = d[d.primary_seller_id == ids[0]]
    n = len(x)
    P = d
    out = {"ok": True, "seller": ids[0][:12] + "…", "orders_in_sample": n, "late_rate": float(x.is_late.mean()), "mean_review": float(x.review_score.mean()),
           "sample_late_rate": float(P.is_late.mean()), "sample_mean_review": float(P.review_score.mean()), "mode": "demo (descriptive only)"}
    out["findings"] = [f"This seller has {n} order(s) in the {len(P)}-row demo sample; late rate {out['late_rate']:.0%} vs {out['sample_late_rate']:.0%} overall.",
                       f"Mean review {out['mean_review']:.2f} vs {out['sample_mean_review']:.2f} overall."]
    out["limitation"] = ("With so few orders, these numbers are NOT reliable. Registered users get the full multi-agent investigation with confidence intervals, "
                         "evidence ledger and human-approved actions on the complete dataset.")
    return out


def top_sellers():
    d = sample()
    return [{"id": s[:12], "full": s[:12], "orders": int(n)} for s, n in d.primary_seller_id.value_counts().head(6).items()]


# ---------------------------------------------------------------- own-data check (guest demo, capped at the demo row limit)
ALLOWED_EXT = (".csv", ".xlsx")
MAX_COLS = 100


def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if (np.isnan(v) or np.isinf(v)) else float(v)
    return str(v)[:80]


def check_own_data(filename, data):
    """Profile a guest's own file using ONLY its first `rows` rows (backend cap). Nothing is saved to disk.
    Returns {"ok": False, "error": ...} for invalid input or the result dict with an explicit 'limited to N rows' notice."""
    from multids.profile import LoadError, load_table, profile
    L = limits()
    name = (filename or "").strip()
    if not name or len(name) > 150:
        return {"ok": False, "error": "Choose a file first (the file name must be 1-150 characters)."}
    if not name.lower().endswith(ALLOWED_EXT):
        return {"ok": False, "error": "Only .csv or .xlsx files can be checked in demo mode."}
    if not data:
        return {"ok": False, "error": "The file is empty."}
    max_bytes = int(current_app.config.get("DEMO_UPLOAD_MAX_BYTES", 5 * 1024 * 1024))
    if len(data) > max_bytes:
        return {"ok": False, "error": f"Demo mode accepts files up to {max_bytes // (1024 * 1024)} MB. Register a free account for larger files."}
    try:
        df, notes = load_table(name, data)
    except LoadError as e:
        return {"ok": False, "error": str(e)}
    total = int(len(df))
    notes = list(notes)
    if df.shape[1] > MAX_COLS:
        notes.append(f"The file has {df.shape[1]} columns; only the first {MAX_COLS} were used.")
        df = df.iloc[:, :MAX_COLS]
    cap = L["rows"]
    truncated = total > cap
    df = df.head(cap).copy()                              # <- the backend cap: nothing beyond this row is ever analysed
    prof = profile(df)
    notice = (f"Data provided is limited to {cap} rows only. Your file has {total:,} rows; only the first {cap} rows were used for the information below."
              if truncated else f"Your file has {total:,} rows (demo mode reads at most {cap} rows).")
    num = [c for c in prof["columns"] if c["kind"] == "numeric"][:8]
    cat = None
    for c in prof["by_kind"]["categorical"][:1]:
        vc = df[c].value_counts(dropna=True).head(5)
        cat = {"column": c, "top": [{"value": str(k)[:60], "count": int(v)} for k, v in vc.items()]}
    preview = [[_clean(v) for v in row] for row in df.head(10).itertuples(index=False, name=None)]
    return {"ok": True, "file_name": name[:150], "rows_in_file": total, "rows_used": int(len(df)), "row_limit": cap, "truncated": truncated,
            "notice": notice, "notes": notes, "n_cols": int(df.shape[1]), "quality_score": prof["quality_score"], "duplicate_rows": prof["duplicate_rows"],
            "columns": [{k: _clean(v) for k, v in c.items()} for c in prof["columns"]], "issues": prof["issues"][:12],
            "by_kind": {k: v for k, v in prof["by_kind"].items() if v}, "numeric_summary": [{k: _clean(v) for k, v in c.items()} for c in num],
            "top_categories": cat, "preview_columns": [str(c)[:60] for c in df.columns], "preview_rows": preview,
            "limitation": "Demo check: descriptive profile of the first rows only. Register a free account to analyse the whole file, run predictions and download reports."}
