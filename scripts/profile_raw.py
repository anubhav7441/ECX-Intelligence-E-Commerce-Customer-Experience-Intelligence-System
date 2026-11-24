"""Profile every raw Olist CSV: shape, dtypes, missing values, duplicates, key uniqueness."""
import sys
from pathlib import Path
import pandas as pd

RAW = Path(sys.argv[1])
FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

dfs = {}
for name, fn in FILES.items():
    df = pd.read_csv(RAW / fn)
    dfs[name] = df
    print("=" * 70)
    print(f"{name}: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print("columns/dtypes:", dict(df.dtypes.astype(str)))
    miss = df.isna().sum()
    miss = miss[miss > 0]
    print("missing:", {k: f"{v} ({v/len(df):.1%})" for k, v in miss.items()} or "none")
    print("full-row duplicates:", int(df.duplicated().sum()))

print("=" * 70)
print("KEY CHECKS")
o, c, i, r, p, s = (dfs[k] for k in ["orders", "customers", "order_items", "reviews", "products", "sellers"])
print("orders.order_id unique:", o.order_id.is_unique)
print("customers.customer_id unique:", c.customer_id.is_unique)
print("unique customer_unique_id:", c.customer_unique_id.nunique())
print("orders with >1 item rows:", int((i.groupby("order_id").size() > 1).sum()))
print("unique orders in items:", i.order_id.nunique())
print("payments rows per order >1:", int((dfs["payments"].groupby("order_id").size() > 1).sum()))
print("reviews: unique review_id", r.review_id.nunique(), "| unique order_id", r.order_id.nunique(), "| rows", len(r))
print("orders with >1 review:", int((r.groupby("order_id").size() > 1).sum()))
print("review_id appears >1 time:", int((r.review_id.value_counts() > 1).sum()))
print("orders without review:", int((~o.order_id.isin(r.order_id)).sum()))
print("items.order_id not in orders:", int((~i.order_id.isin(o.order_id)).sum()))
print("items.product_id not in products:", int((~i.product_id.isin(p.product_id)).sum()))
print("items.seller_id not in sellers:", int((~i.seller_id.isin(s.seller_id)).sum()))
print("orders.customer_id not in customers:", int((~o.customer_id.isin(c.customer_id)).sum()))
print("orders without items:", int((~o.order_id.isin(i.order_id)).sum()))

print("=" * 70)
print("ORDERS: status counts"); print(o.order_status.value_counts().to_string())
for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
    d = pd.to_datetime(o[col], errors="coerce")
    print(col, "min", d.min(), "max", d.max())
print("REVIEWS: score distribution"); print(r.review_score.value_counts(normalize=True).sort_index().round(4).to_string())
print("reviews with comment text:", int(r.review_comment_message.notna().sum()),
      f"({r.review_comment_message.notna().mean():.1%})")
print("reviews with title:", int(r.review_comment_title.notna().sum()))
print("review date range:", pd.to_datetime(r.review_creation_date).min(), pd.to_datetime(r.review_creation_date).max())
print("PRODUCTS: null category", int(p.product_category_name.isna().sum()),
      "| categories not in translation:",
      sorted(set(p.product_category_name.dropna()) - set(dfs['category_translation'].product_category_name)))
g = dfs["geolocation"]
print("GEO: unique zip prefixes", g.geolocation_zip_code_prefix.nunique())
print("customer zips missing in geo:", int((~c.customer_zip_code_prefix.isin(g.geolocation_zip_code_prefix)).sum()),
      "| seller zips missing in geo:", int((~s.seller_zip_code_prefix.isin(g.geolocation_zip_code_prefix)).sum()))
