"""Phase 1: build the customer-product interaction table from the raw Olist CSVs.
Every decision is written to data/processed/recsys/prep_log.json. Run: python -m recsys.data"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "recsys"
BAD_STATUS = ["canceled", "unavailable"]


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    log = []
    o = pd.read_csv(RAW / "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
    it = pd.read_csv(RAW / "olist_order_items_dataset.csv")
    cu = pd.read_csv(RAW / "olist_customers_dataset.csv")
    pr = pd.read_csv(RAW / "olist_products_dataset.csv")
    tr = pd.read_csv(RAW / "product_category_name_translation.csv")
    log.append(f"Raw rows: orders {len(o)}, items {len(it)}, customers {len(cu)}, products {len(pr)}.")
    m = it.merge(o[["order_id", "customer_id", "order_status", "order_purchase_timestamp"]], on="order_id", how="inner")
    m = m.merge(cu[["customer_id", "customer_unique_id"]], on="customer_id", how="inner")
    n0 = len(m)
    m = m[~m.order_status.isin(BAD_STATUS)]
    log.append(f"Excluded orders with status {BAD_STATUS}: {n0 - len(m)} item rows removed (the product was never delivered to the customer).")
    n1 = len(m)
    m = m.dropna(subset=["customer_unique_id", "product_id", "order_purchase_timestamp"])
    log.append(f"Dropped rows with missing customer/product/time: {n1 - len(m)}.")
    n2 = len(m)
    m = m[m.price > 0]
    log.append(f"Dropped rows with non-positive price: {n2 - len(m)}.")
    log.append("customer_unique_id (not customer_id) identifies a person across orders; olist's customer_id changes with every order.")
    log.append("Each order_item row is one unit; repeated units of a product in one order are counted as one purchase event with a quantity.")
    pr = pr.merge(tr, on="product_category_name", how="left")
    pr["category"] = pr.product_category_name_english.fillna(pr.product_category_name).fillna("unknown")
    log.append(f"Products with no category: {int((pr.category == 'unknown').sum())}; labelled 'unknown' (never guessed).")
    inter = (m.groupby(["customer_unique_id", "product_id"])
             .agg(n_orders=("order_id", "nunique"), qty=("order_item_id", "count"), first_time=("order_purchase_timestamp", "min"),
                  last_time=("order_purchase_timestamp", "max"), spend=("price", "sum")).reset_index())
    log.append(f"Interaction table: one row per customer-product pair (implicit feedback, 1 = purchased): {len(inter)} rows, "
               f"{inter.customer_unique_id.nunique()} customers, {inter.product_id.nunique()} products.")
    per = inter.groupby("customer_unique_id").product_id.nunique()
    log.append(f"Customers with >=2 distinct products: {int((per >= 2).sum())} ({(per >= 2).mean() * 100:.1f}%). Sparsity: "
               f"{100 * (1 - len(inter) / (inter.customer_unique_id.nunique() * inter.product_id.nunique())):.4f}%.")
    # basket table: one row per (order, product)
    bask = m[["order_id", "customer_unique_id", "product_id", "order_purchase_timestamp"]].drop_duplicates(["order_id", "product_id"])
    log.append(f"Baskets (distinct products per order): {bask.order_id.nunique()} orders, {int((bask.groupby('order_id').size() >= 2).sum())} with >=2 distinct products.")
    prod = pr[["product_id", "category", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
               "product_photos_qty", "product_name_lenght", "product_description_lenght"]].copy()
    price = m.groupby("product_id").price.median().rename("median_price")
    prod = prod.merge(price, on="product_id", how="left")
    log.append("The dataset has no product names or description text, only category, size, weight and counts; content features use these only.")
    inter.to_parquet(OUT / "interactions.parquet"); bask.to_parquet(OUT / "baskets.parquet"); prod.to_parquet(OUT / "products.parquet")
    (OUT / "prep_log.json").write_text(json.dumps({"decisions": log}, indent=1))
    for l in log: print("-", l)


if __name__ == "__main__":
    build()
