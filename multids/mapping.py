"""Column-role mapping: automatic suggestion (names + value checks) and validation of the user's confirmed mapping."""
import re
import pandas as pd
from multids.profile import column_kind

ROLES = {   # role: (label, expected kinds, synonyms)
    "order_id": ("Order / invoice ID", {"id", "categorical", "numeric"}, ["order id", "orderid", "invoice", "invoiceno", "invoice no", "transaction id", "order number", "order_id", "order no", "order ref", "ref no", "bill no", "receipt", "txn", "txn id", "transaction"]),
    "customer_id": ("Customer ID", {"id", "categorical", "numeric"}, ["customer id", "customerid", "customer_unique_id", "user id", "user_id", "client id", "buyer id", "reviewer id", "reviewerid", "cust", "customer", "client", "user", "buyer", "member", "email", "customer name"]),
    "product_id": ("Product ID / SKU", {"id", "categorical", "numeric"}, ["product id", "productid", "stockcode", "stock code", "sku", "asin", "parent_asin", "item id", "item_id", "product_id", "item code", "product code", "article", "model no", "upc", "barcode"]),
    "product_name": ("Product name / description", {"text", "categorical"}, ["description", "product name", "product_name", "item name", "title", "product", "item", "name of product", "goods"]),
    "category": ("Product category", {"categorical"}, ["category", "categories", "department", "product type", "segment", "type", "group", "genre", "class", "brand"]),
    "date": ("Transaction / review date", {"date"}, ["date", "invoicedate", "order date", "purchase", "timestamp", "time", "created", "reviewtime", "day", "month", "invoice date", "sale date", "order dt", "dt", "posted"]),
    "delivery_date": ("Delivered date", {"date"}, ["delivered", "delivery date", "delivered_customer", "arrival"]),
    "estimated_date": ("Estimated delivery date", {"date"}, ["estimated", "promised", "expected delivery"]),
    "delivery_days": ("Delivery time (days)", {"numeric"}, ["delivery days", "delivery_days", "days to deliver", "transit days", "shipping days"]),
    "delay_days": ("Delay vs estimate (days, >0 = late)", {"numeric"}, ["delay", "delay_days", "days late"]),
    "quantity": ("Quantity", {"numeric"}, ["quantity", "qty", "units", "count", "pieces", "pcs", "items sold", "volume", "no of units"]),
    "unit_price": ("Unit price", {"numeric"}, ["unit price", "unitprice", "price", "unit_price", "cost", "rate", "mrp", "selling price", "sale price"]),
    "revenue": ("Revenue / line total", {"numeric"}, ["revenue", "total", "amount", "sales", "line total", "payment_value", "gmv", "income", "turnover", "net sales", "grand total", "bill amount", "value"]),
    "region": ("Country / state / region", {"categorical"}, ["country", "state", "region", "province", "territory", "customer_state", "city", "location", "zone", "area", "market", "district", "nation"]),
    "rating": ("Rating / score", {"numeric"}, ["rating", "score", "stars", "review_score", "star", "grade", "satisfaction", "rank", "points", "stars given"]),
    "review_text": ("Review text", {"text", "categorical"}, ["review text", "review", "text", "comment", "message", "body", "feedback", "reviewtext", "remarks", "remark", "opinion", "notes", "description of review", "summary"]),
    "seller_id": ("Seller / vendor ID", {"id", "categorical", "numeric"}, ["seller", "vendor", "merchant", "store", "shop", "supplier", "branch", "outlet"]),
}
_tok = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _score(col, role, kind, s):
    label, kinds, syn = ROLES[role]
    if kind == "id" and pd.api.types.is_numeric_dtype(s) and "numeric" in kinds:
        kind = "numeric"   # sequential-looking integers can still be quantities
    c = _tok(col)
    best = 0.0
    for w in syn:
        w = _tok(w)
        if c == w:
            best = max(best, 1.0)
        elif w and (f" {w} " in f" {c} "):
            best = max(best, 0.8)
        elif w and len(w) > 3 and w in c:
            best = max(best, 0.55)
    if best == 0:
        return 0.0, ""
    if kind not in kinds:
        # tolerate: date role with strings that parse; numeric id columns used as categorical ids
        if not (role in ("customer_id", "order_id", "product_id", "seller_id") and kind in ("id", "categorical", "numeric", "text")):
            return 0.0, ""
        best *= 0.8
    if role == "rating" and kind == "numeric":
        v = s.dropna()
        if v.empty or v.min() < 0 or v.max() > 10 or v.nunique() > 11:
            return 0.0, ""
    if role == "region" and s.nunique() > 400:
        best *= 0.5
    if role in ("delivery_days", "delay_days") and kind == "numeric":
        pass
    return best, f"name matches '{col}'" + ("" if kind in kinds else " (type slightly different)")


def suggest(df, prof=None):
    kinds = {c: column_kind(df[c]) for c in df.columns}
    cand = []
    for role in ROLES:
        for c in df.columns:
            sc, why = _score(c, role, kinds[c], df[c])
            if sc > 0:
                cand.append((sc, role, c, why))
    cand.sort(key=lambda x: -x[0])
    used_roles, used_cols, out = set(), set(), {}
    for sc, role, c, why in cand:
        if role in used_roles or (c in used_cols and not (role == "revenue" and False)):
            continue
        used_roles.add(role); used_cols.add(c)
        out[role] = {"column": c, "confidence": round(sc, 2), "reason": why}
    _fallbacks(df, kinds, out, used_cols)
    return {"suggested": out, "kinds": kinds, "roles": {r: v[0] for r, v in ROLES.items()}}


def _fallbacks(df, kinds, out, used_cols):
    """Low-confidence suggestions from the values themselves, used only when the names gave nothing."""
    if len(df) < 30:
        return
    free = [c for c in df.columns if c not in used_cols]
    def add(role, c, why):
        if role not in out and c not in used_cols:
            out[role] = {"column": c, "confidence": 0.3, "reason": why}; used_cols.add(c)
    dates = [c for c in free if kinds[c] == "date"]
    if len(dates) == 1:
        add("date", dates[0], "the only column that holds dates (please confirm)")
    texts = [c for c in free if kinds[c] == "text"]
    if len(texts) == 1:
        add("review_text", texts[0], "the only column with long free text (please confirm)")
    nums = [c for c in free if kinds[c] == "numeric" and c not in used_cols]
    if "revenue" not in out and "unit_price" not in out and len(nums) == 1 and df[nums[0]].dropna().ge(0).mean() > 0.95 and df[nums[0]].nunique() > 10:
        add("revenue", nums[0], "the only numeric column; assumed to be an amount (please confirm)")


def validate(df, mapping):
    """mapping: {role: column}. Returns (clean_mapping, errors, warnings)."""
    errs, warns, clean = [], [], {}
    if not isinstance(mapping, dict):
        return {}, ["Mapping must be an object {role: column}."], []
    used = {}
    for role, col in mapping.items():
        if col in (None, ""):
            continue
        if role not in ROLES:
            errs.append(f"Unknown role '{role}'."); continue
        if col not in df.columns:
            errs.append(f"Column '{col}' (for {ROLES[role][0]}) does not exist."); continue
        if col in used:
            errs.append(f"Column '{col}' is used for both '{used[col]}' and '{role}'. One column can have one meaning."); continue
        k = column_kind(df[col])
        if k == "id" and pd.api.types.is_numeric_dtype(df[col]) and "numeric" in ROLES[role][1]:
            k = "numeric"
        exp = ROLES[role][1]
        if k not in exp:
            ok_soft = (role in ("customer_id", "order_id", "product_id", "seller_id")) and k in ("id", "categorical", "numeric", "text")
            if role == "date" or role.endswith("_date"):
                p = pd.to_datetime(df[col].dropna().head(500), errors="coerce", format="mixed")
                ok_soft = p.notna().mean() > 0.9
            if role in ("quantity", "unit_price", "revenue", "rating", "delivery_days", "delay_days"):
                ok_soft = pd.to_numeric(df[col].dropna().head(500), errors="coerce").notna().mean() > 0.9
            if not ok_soft:
                errs.append(f"Column '{col}' looks like '{k}' data, but '{ROLES[role][0]}' needs {sorted(exp)} data."); continue
            warns.append(f"Column '{col}' detected as '{k}'; accepted for '{ROLES[role][0]}' after conversion.")
        used[col] = role
        clean[role] = col
    return clean, errs, warns


def prepare(df, mapping):
    """Build a standardised frame with canonical role names. Returns (std_df, notes). Never mutates df."""
    notes, out = [], pd.DataFrame(index=df.index)
    for role, col in mapping.items():
        s = df[col]
        if role == "date" or role.endswith("_date"):
            p = pd.to_datetime(s, errors="coerce", format="mixed")
            bad = int(p.isna().sum() - s.isna().sum())
            if bad:
                notes.append(f"{role}: {bad:,} values could not be parsed as dates and were ignored.")
            out[role] = p
        elif role in ("quantity", "unit_price", "revenue", "rating", "delivery_days", "delay_days"):
            n = pd.to_numeric(s, errors="coerce")
            bad = int(n.isna().sum() - s.isna().sum())
            if bad:
                notes.append(f"{role}: {bad:,} non-numeric values were ignored.")
            out[role] = n
        else:
            out[role] = s
    if "revenue" not in out and "quantity" in out and "unit_price" in out:
        out["revenue"] = out["quantity"] * out["unit_price"]
        notes.append("Revenue was computed as quantity x unit price.")
    if "delivery_days" not in out and "date" in out and "delivery_date" in out:
        out["delivery_days"] = (out["delivery_date"] - out["date"]).dt.total_seconds() / 86400
        notes.append("Delivery days = delivered date minus order date.")
    if "delay_days" not in out and "delivery_date" in out and "estimated_date" in out:
        out["delay_days"] = (out["delivery_date"] - out["estimated_date"]).dt.total_seconds() / 86400
        notes.append("Delay = delivered date minus estimated date (>0 means late).")
    return out, notes
