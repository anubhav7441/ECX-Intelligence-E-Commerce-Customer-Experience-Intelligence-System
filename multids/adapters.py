"""Dataset adapters: detection and careful cleaning for UCI Online Retail II and Amazon Reviews 2023 files.
Every cleaning step is counted and returned as a processing-history entry. Original raw values are never silently altered."""
import re
import pandas as pd

RETAIL_COLS = {"invoice", "stockcode", "description", "quantity", "invoicedate", "price", "customer id", "country"}
INFO = {
    "online_retail_ii": {
        "name": "UCI Online Retail II (Dataset B)", "source": "UCI Machine Learning Repository - Online Retail II",
        "source_url": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
        "description": "Transactions of a UK-based online retailer (gift-ware, many wholesale customers), 2009-2011. Cite as shown on the official dataset page; check its license terms before redistributing."},
    "amazon_reviews": {
        "name": "Amazon Reviews 2023 (Dataset C)", "source": "Amazon Reviews 2023 (McAuley Lab, UCSD)",
        "source_url": "https://amazon-reviews-2023.github.io/",
        "description": "Product reviews from Amazon. Only a single category-specific sample is used (see sample note). Cite as shown on the official site; check its terms of use."},
}
PRESET = {
    "online_retail_ii": {"order_id": "Invoice", "customer_id": "Customer ID", "product_id": "StockCode", "product_name": "Description", "date": "InvoiceDate",
                         "quantity": "Quantity", "unit_price": "Price", "revenue": "Revenue", "region": "Country"},
    "amazon_reviews": {"rating": "rating", "review_text": "text", "product_id": "parent_asin", "customer_id": "user_id", "date": "timestamp"},
}


def detect(df):
    low = {str(c).strip().lower() for c in df.columns}
    if len(RETAIL_COLS & low) >= 7:
        return "online_retail_ii"
    if {"rating", "text"} <= low and ({"asin", "parent_asin"} & low) and ({"timestamp", "user_id"} & low):
        return "amazon_reviews"
    return None


def clean_online_retail(df):
    log = []
    ren = {c: next((k for k in ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"] if k.lower() == str(c).strip().lower()), c) for c in df.columns}
    d = df.rename(columns=ren).copy()
    n0 = len(d)
    log.append({"step": "adapter: Online Retail II", "detail": f"{n0:,} raw rows, {d.shape[1]} columns."})
    dup = int(d.duplicated().sum())
    d = d.drop_duplicates()
    log.append({"step": "removed exact duplicate rows", "detail": f"{dup:,} removed (the source file contains repeated lines)."})
    d["Invoice"] = d["Invoice"].astype(str).str.strip()
    d["is_cancellation"] = d["Invoice"].str.upper().str.startswith("C")
    log.append({"step": "flagged cancellations", "detail": f"{int(d.is_cancellation.sum()):,} rows have an invoice starting with 'C'; KEPT as negative sales/returns and flagged in column is_cancellation."})
    price = pd.to_numeric(d["Price"], errors="coerce")
    bad_price = int((price.isna() | (price <= 0)).sum())
    d = d[~(price.isna() | (price <= 0))].copy()
    d["Price"] = pd.to_numeric(d["Price"], errors="coerce")
    log.append({"step": "removed invalid prices", "detail": f"{bad_price:,} rows with price <= 0 or non-numeric removed (stock adjustments / bad debt entries)."})
    sc = d["StockCode"].astype(str).str.strip()
    nonprod = ~sc.str.match(r"^\d{5}")
    d = d[~nonprod].copy()
    log.append({"step": "removed non-product stock codes", "detail": f"{int(nonprod.sum()):,} rows (e.g. POST, DOT, M, BANK CHARGES, fees) are not merchandise and were removed from product analysis."})
    qty = pd.to_numeric(d["Quantity"], errors="coerce")
    neg_nocancel = int(((qty < 0) & ~d.is_cancellation).sum())
    log.append({"step": "checked negative quantities", "detail": f"{int((qty < 0).sum()):,} negative-quantity rows remain ({neg_nocancel:,} without a 'C' invoice; kept, treated as returns/adjustments)."})
    miss = int(d["Customer ID"].isna().sum())
    d["Customer ID"] = d["Customer ID"].map(lambda v: None if pd.isna(v) else str(int(float(v))) if re.fullmatch(r"\d+(\.0+)?", str(v)) else str(v))
    log.append({"step": "customer IDs", "detail": f"{miss:,} rows have no Customer ID; kept for sales totals but excluded from customer segmentation automatically."})
    d["Revenue"] = pd.to_numeric(d["Quantity"], errors="coerce") * d["Price"]
    log.append({"step": "derived Revenue", "detail": "Revenue = Quantity x Price (negative for cancellations)."})
    d["InvoiceDate"] = pd.to_datetime(d["InvoiceDate"], errors="coerce")
    log.append({"step": "final size", "detail": f"{len(d):,} rows kept of {n0:,} ({len(d) / n0:.1%})."})
    return d.reset_index(drop=True), log


def clean_amazon(df, category="unspecified", cap=100_000):
    log = [{"step": "adapter: Amazon Reviews 2023", "detail": f"{len(df):,} reviews read, columns: {', '.join(map(str, df.columns))}."}]
    d = df.copy()
    for c in list(d.columns):
        if d[c].map(lambda v: isinstance(v, (list, dict))).any():
            d = d.drop(columns=[c]); log.append({"step": f"dropped nested column '{c}'", "detail": "Lists/dicts are not analysed."})
    if "timestamp" in d.columns:
        ts = pd.to_numeric(d["timestamp"], errors="coerce")
        d["timestamp"] = pd.to_datetime(ts, unit="ms", errors="coerce")
        log.append({"step": "converted timestamp", "detail": f"Unix milliseconds -> datetime; {int(d.timestamp.isna().sum()):,} unparsable."})
    d["rating"] = pd.to_numeric(d["rating"], errors="coerce")
    bad = int((~d.rating.between(1, 5)).sum())
    d = d[d.rating.between(1, 5)]
    log.append({"step": "validated ratings", "detail": f"{bad:,} rows with rating outside 1-5 removed."})
    keys = [c for c in ["user_id", "parent_asin", "timestamp", "text"] if c in d.columns]
    dup = int(d.duplicated(keys).sum())
    d = d.drop_duplicates(keys)
    log.append({"step": "removed duplicate reviews", "detail": f"{dup:,} duplicates (same user, product, time, text)."})
    for c in ("text", "title"):
        if c in d.columns:
            d[c] = d[c].fillna("").astype(str).str.replace(r"<[^>]+>", " ", regex=True).str.strip()
    log.append({"step": "text cleaning", "detail": "HTML tags stripped, whitespace trimmed; text otherwise unchanged."})
    note = f"Category: {category}. {len(d):,} reviews after cleaning (sample cap {cap:,} lines per upload). Not the full dataset."
    log.append({"step": "sample documented", "detail": note})
    return d.reset_index(drop=True), log, note
