"""CSV upload validation (Data Management module). Files are validated in memory and never written to disk or the database."""
import io
import pandas as pd

KNOWN = {
    "orders": ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at",
               "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"],
    "order_items": ["order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_date", "price", "freight_value"],
    "order_reviews": ["review_id", "order_id", "review_score", "review_comment_title", "review_comment_message",
                      "review_creation_date", "review_answer_timestamp"],
    "order_payments": ["order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"],
    "customers": ["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"],
    "sellers": ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"],
    "products": ["product_id", "product_category_name", "product_name_lenght", "product_description_lenght", "product_photos_qty",
                 "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"],
    "geolocation": ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng", "geolocation_city", "geolocation_state"],
}
KEYS = {"orders": ["order_id"], "customers": ["customer_id"], "sellers": ["seller_id"], "products": ["product_id"],
        "order_items": ["order_id", "order_item_id"], "order_payments": ["order_id", "payment_sequential"]}
MAX_BYTES = 20 * 1024 * 1024


class UploadError(ValueError):
    pass


def validate_csv(filename, data):
    if not filename or not filename.lower().endswith(".csv"):
        raise UploadError("Only .csv files are accepted.")
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise UploadError("File is larger than 20 MB.")
    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            raise UploadError(f"Could not parse CSV: {e}"[:200]) from e
    if df is None:
        raise UploadError("Could not decode the file (tried utf-8 and latin-1).")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise UploadError("The CSV has no data rows or no columns.")

    cols = set(df.columns)
    best, best_overlap = None, 0
    for name, req in KNOWN.items():
        ov = len(cols & set(req))
        if ov > best_overlap:
            best, best_overlap = name, ov
    detected = best if best and best_overlap >= max(2, 0.6 * len(KNOWN[best])) else None
    missing_required = [c for c in KNOWN[detected] if c not in cols] if detected else []

    miss = df.isna().sum()
    n = len(df)
    issues = []
    if detected and missing_required:
        issues.append(f"Missing expected columns for '{detected}': {', '.join(missing_required)}")
    dup_rows = int(df.duplicated().sum())
    if dup_rows:
        issues.append(f"{dup_rows} fully duplicated rows")
    key_dups = None
    if detected in KEYS and all(k in df.columns for k in KEYS[detected]):
        key_dups = int(df.duplicated(subset=KEYS[detected]).sum())
        if key_dups:
            issues.append(f"{key_dups} duplicated primary-key values ({', '.join(KEYS[detected])})")
    date_checks = {}
    for c in df.columns:
        if "date" in c.lower() or "timestamp" in c.lower() or c.lower().endswith("_at"):
            parsed = pd.to_datetime(df[c], errors="coerce")
            bad = int((parsed.isna() & df[c].notna()).sum())
            date_checks[c] = {"unparseable": bad}
            if bad:
                issues.append(f"Column '{c}': {bad} values are not valid dates")
    if detected == "order_reviews" and "review_score" in df.columns:
        sc = pd.to_numeric(df.review_score, errors="coerce")
        bad = int(((sc < 1) | (sc > 5) | sc.isna()).sum())
        if bad:
            issues.append(f"review_score: {bad} values outside 1-5 or non-numeric")
    for c in ("price", "freight_value", "payment_value"):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            if (v < 0).any():
                issues.append(f"Column '{c}': {int((v < 0).sum())} negative values")

    return {
        "filename": filename, "rows": int(n), "columns": int(df.shape[1]), "detected_table": detected,
        "missing_required_columns": missing_required,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": [{"column": c, "missing": int(m), "pct": round(float(m) / n * 100, 2)} for c, m in miss.items() if m > 0],
        "duplicate_rows": dup_rows, "duplicate_keys": key_dups, "date_checks": date_checks, "issues": issues,
        "quality": "OK" if not issues else "ISSUES FOUND",
        "preview_columns": [str(c) for c in df.columns],
        "preview": [[("" if pd.isna(x) else str(x)[:80]) for x in row] for row in df.head(10).itertuples(index=False, name=None)],
    }
