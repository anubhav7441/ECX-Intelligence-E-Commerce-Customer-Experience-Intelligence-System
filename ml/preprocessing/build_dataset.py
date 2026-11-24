"""Build the order-level analytical dataset (ONE row per order) from the SQLite database.

Key design decisions (all verified against the real data):
  * Items, payments and reviews are aggregated to order level BEFORE joining -> no join inflation.
  * Orders with several reviews keep only the latest review (by answer timestamp).
  * Lateness is date-based: late if delivered DATE > estimated DATE (estimated dates have no time part).
  * Analytical population = delivered orders that have a delivery date, >=1 item and a review.

Run from project root:  python -m ml.preprocessing.build_dataset
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[2]
DB_URL = f"sqlite:///{ROOT / 'database' / 'ecx.db'}"
OUT_DIR = ROOT / "data" / "processed"


def haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lng2) - np.radians(lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# Brazilian macro-regions (IBGE) for state -> region mapping
REGION = {
    **dict.fromkeys(["AC", "AP", "AM", "PA", "RO", "RR", "TO"], "North"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Northeast"),
    **dict.fromkeys(["DF", "GO", "MT", "MS"], "Central-West"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Southeast"),
    **dict.fromkeys(["PR", "RS", "SC"], "South"),
}


def build(engine=None):
    engine = engine or create_engine(DB_URL)
    q = lambda sql: pd.read_sql(sql, engine)
    report = {}

    orders = q("SELECT * FROM orders")
    for c in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
              "order_delivered_customer_date", "order_estimated_delivery_date"]:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")
    report["orders_total"] = len(orders)

    # ---- items -> order level
    items = q("""SELECT i.*, p.product_category_name, p.product_weight_g, p.product_length_cm,
                        p.product_height_cm, p.product_width_cm, p.product_photos_qty,
                        p.product_description_lenght, p.product_name_lenght
                 FROM order_items i JOIN products p ON p.product_id = i.product_id""")
    trans = q("SELECT * FROM category_translation")
    items = items.merge(trans, on="product_category_name", how="left")
    items["category"] = items["product_category_name_english"].fillna("unknown")
    items["volume_cm3"] = items.product_length_cm * items.product_height_cm * items.product_width_cm
    first = items.sort_values(["order_id", "order_item_id"]).drop_duplicates("order_id")[
        ["order_id", "seller_id", "product_id", "category", "product_photos_qty",
         "product_description_lenght", "product_name_lenght"]
    ].rename(columns={"seller_id": "primary_seller_id", "product_id": "primary_product_id",
                      "category": "primary_category"})
    agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "size"),
        n_sellers=("seller_id", "nunique"),
        n_categories=("category", "nunique"),
        items_price=("price", "sum"),
        freight_value=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        total_volume_cm3=("volume_cm3", "sum"),
        max_shipping_limit=("shipping_limit_date", "max"),
    ).reset_index()
    agg["freight_ratio"] = np.where(agg.items_price > 0, agg.freight_value / agg.items_price, np.nan)
    agg["max_shipping_limit"] = pd.to_datetime(agg["max_shipping_limit"], errors="coerce")

    # ---- payments -> order level
    pay = q("SELECT * FROM order_payments")
    pay_agg = pay.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        n_payment_rows=("payment_sequential", "size"),
        max_installments=("payment_installments", "max"),
    ).reset_index()
    main_type = (pay.sort_values("payment_value", ascending=False).drop_duplicates("order_id")
                 [["order_id", "payment_type"]].rename(columns={"payment_type": "main_payment_type"}))
    pay_agg = pay_agg.merge(main_type, on="order_id")

    # ---- reviews -> latest review per order
    rev = q("SELECT * FROM order_reviews")
    report["review_rows"] = len(rev)
    rev["review_answer_timestamp"] = pd.to_datetime(rev["review_answer_timestamp"], errors="coerce")
    rev["review_creation_date"] = pd.to_datetime(rev["review_creation_date"], errors="coerce")
    rev = rev.sort_values(["order_id", "review_answer_timestamp", "review_pk"])
    report["orders_with_multiple_reviews"] = int((rev.groupby("order_id").size() > 1).sum())
    rev = rev.drop_duplicates("order_id", keep="last")
    rev["has_comment"] = rev.review_comment_message.notna() & (rev.review_comment_message.str.strip() != "")

    # ---- customers + geo
    cust = q("SELECT * FROM customers")
    sell = q("SELECT seller_id, seller_zip_code_prefix, seller_city, seller_state FROM sellers")
    geo = q("SELECT zip_code_prefix, lat, lng FROM geolocation_zip")

    df = (orders.merge(cust, on="customer_id", how="left")
                .merge(agg, on="order_id", how="left")
                .merge(first, on="order_id", how="left")
                .merge(pay_agg, on="order_id", how="left")
                .merge(rev[["order_id", "review_id", "review_score", "review_comment_title", "review_comment_message",
                            "review_creation_date", "review_answer_timestamp", "has_comment"]],
                       on="order_id", how="left")
                .merge(sell.rename(columns={"seller_id": "primary_seller_id"}), on="primary_seller_id", how="left"))

    cg = geo.rename(columns={"zip_code_prefix": "customer_zip_code_prefix", "lat": "cust_lat", "lng": "cust_lng"})
    sg = geo.rename(columns={"zip_code_prefix": "seller_zip_code_prefix", "lat": "sell_lat", "lng": "sell_lng"})
    df = df.merge(cg, on="customer_zip_code_prefix", how="left").merge(sg, on="seller_zip_code_prefix", how="left")
    df["distance_km"] = haversine_km(df.sell_lat, df.sell_lng, df.cust_lat, df.cust_lng)
    df["customer_region"] = df.customer_state.map(REGION)
    df["seller_region"] = df.seller_state.map(REGION)
    df["same_state"] = (df.customer_state == df.seller_state).astype(int)

    # ---- time / delivery features
    P, A, C, D, E = ("order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                     "order_delivered_customer_date", "order_estimated_delivery_date")
    df["purchase_month"] = df[P].dt.to_period("M").astype(str)
    df["purchase_dow"] = df[P].dt.dayofweek
    df["purchase_hour"] = df[P].dt.hour
    df["delivery_days"] = (df[D] - df[P]).dt.total_seconds() / 86400
    df["estimated_days"] = (df[E] - df[P]).dt.total_seconds() / 86400
    df["delay_days"] = (df[D].dt.normalize() - df[E].dt.normalize()).dt.days  # >0 means late
    df["is_late"] = np.where(df[D].notna(), (df.delay_days > 0).astype(float), np.nan)
    df["approval_hours"] = (df[A] - df[P]).dt.total_seconds() / 3600
    df["seller_handling_days"] = (df[C] - df[A]).dt.total_seconds() / 86400
    df["carrier_transit_days"] = (df[D] - df[C]).dt.total_seconds() / 86400
    df["review_after_delivery"] = df.review_creation_date >= df[D].dt.normalize()

    # ---- customer repeat-purchase info (customer_unique_id level, all orders)
    o_sorted = df[["customer_unique_id", "order_id", P]].sort_values(["customer_unique_id", P])
    o_sorted["order_rank"] = o_sorted.groupby("customer_unique_id").cumcount() + 1
    o_sorted["n_orders_customer"] = o_sorted.groupby("customer_unique_id")["order_id"].transform("size")
    o_sorted["has_later_order"] = (o_sorted.order_rank < o_sorted.n_orders_customer).astype(int)
    df = df.merge(o_sorted[["order_id", "order_rank", "n_orders_customer", "has_later_order"]], on="order_id", how="left")

    # ---- target
    df["dissatisfied"] = np.where(df.review_score.notna(), (df.review_score <= 2).astype(float), np.nan)

    # ---- analytical population funnel
    report["funnel"] = {
        "all_orders": len(df),
        "delivered_status": int((df.order_status == "delivered").sum()),
        "delivered_with_date": int(((df.order_status == "delivered") & df[D].notna()).sum()),
        "...and_has_items": int(((df.order_status == "delivered") & df[D].notna() & df.n_items.notna()).sum()),
        "...and_has_review": int(((df.order_status == "delivered") & df[D].notna() & df.n_items.notna()
                                  & df.review_score.notna()).sum()),
    }
    ana = df[(df.order_status == "delivered") & df[D].notna() & df.n_items.notna() & df.review_score.notna()].copy()

    # ---- data-quality checks on the analytical set
    report["quality"] = {
        "delivered_before_purchase": int((ana.delivery_days < 0).sum()),
        "review_created_before_delivery": int((~ana.review_after_delivery).sum()),
        "missing_distance": int(ana.distance_km.isna().sum()),
        "unknown_category": int((ana.primary_category == "unknown").sum()),
        "zero_price_orders": int((ana.items_price <= 0).sum()),
        "payment_vs_items+freight_mismatch_gt_1": int(
            ((ana.payment_value - (ana.items_price + ana.freight_value)).abs() > 1).sum()),
    }
    for col in ["delivery_days", "freight_ratio", "items_price", "distance_km"]:
        s = ana[col].dropna()
        q1, q3 = s.quantile([.25, .75])
        report.setdefault("iqr_outliers", {})[col] = int(((s < q1 - 3 * (q3 - q1)) | (s > q3 + 3 * (q3 - q1))).sum())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "orders_all.parquet", index=False)
    ana.to_parquet(OUT_DIR / "orders_analytic.parquet", index=False)
    ana.drop(columns=["review_comment_title", "review_comment_message"]).to_csv(
        OUT_DIR / "orders_analytic.csv", index=False)
    items.to_parquet(OUT_DIR / "items_enriched.parquet", index=False)
    report["analytic_rows"] = len(ana)
    report["analytic_unique_orders"] = int(ana.order_id.nunique())
    with open(OUT_DIR / "preprocessing_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return ana, report


if __name__ == "__main__":
    ana, rep = build()
    print(json.dumps(rep, indent=2, default=str))
    print(ana.shape)
