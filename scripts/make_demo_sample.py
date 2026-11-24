"""Build the guest-demo sample: a fixed, reproducible random sample of 500 Olist orders.
Usage (from the project folder):  python scripts/make_demo_sample.py
Needs data/processed/orders_with_history.parquet (from the data ZIP). The sample is a real subset - nothing is invented."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "orders_with_history.parquet"
OUT = ROOT / "data" / "demo" / "demo_sample.parquet"
COLS = ["order_id", "order_purchase_timestamp", "customer_state", "primary_seller_id", "primary_category", "items_price", "freight_value",
        "payment_value", "main_payment_type", "review_score", "delivery_days", "estimated_days", "delay_days", "is_late", "dissatisfied", "purchase_ym"]


def main(n=500, seed=42):
    if not SRC.exists():
        raise SystemExit(f"Missing {SRC}. Extract the data ZIP first.")
    d = pd.read_parquet(SRC)
    d = d[[c for c in COLS if c in d.columns]]
    s = d.sample(n=n, random_state=seed).sort_values("order_purchase_timestamp").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} ({len(s)} rows, {OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
