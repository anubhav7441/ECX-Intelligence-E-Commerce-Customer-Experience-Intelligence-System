"""Create the SQLite database from schema.sql and load all Olist CSVs with SQLAlchemy.

Run from project root:  python scripts/01_load_database.py
"""
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, event, text

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB_PATH = ROOT / "database" / "ecx.db"
SCHEMA = ROOT / "database" / "schema.sql"

# Two categories exist in products.csv but not in the official translation file.
EXTRA_TRANSLATIONS = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_food_preparers",
}


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()
    engine = create_engine(f"sqlite:///{DB_PATH}")

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    # Create tables from schema.sql
    raw_conn = engine.raw_connection()
    raw_conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    raw_conn.commit()
    raw_conn.close()

    def load(table, df):
        df.to_sql(table, engine, if_exists="append", index=False, chunksize=20000)
        print(f"loaded {table:<22} {len(df):>9,} rows")

    trans = pd.read_csv(RAW / "product_category_name_translation.csv")
    extra = pd.DataFrame(
        {"product_category_name": list(EXTRA_TRANSLATIONS), "product_category_name_english": list(EXTRA_TRANSLATIONS.values())}
    )
    load("category_translation", pd.concat([trans, extra], ignore_index=True))
    load("customers", pd.read_csv(RAW / "olist_customers_dataset.csv"))
    load("sellers", pd.read_csv(RAW / "olist_sellers_dataset.csv"))
    load("products", pd.read_csv(RAW / "olist_products_dataset.csv"))
    load("orders", pd.read_csv(RAW / "olist_orders_dataset.csv"))
    load("order_items", pd.read_csv(RAW / "olist_order_items_dataset.csv"))
    load("order_payments", pd.read_csv(RAW / "olist_order_payments_dataset.csv"))
    load("order_reviews", pd.read_csv(RAW / "olist_order_reviews_dataset.csv"))

    geo = pd.read_csv(RAW / "olist_geolocation_dataset.csv")
    geo_zip = (
        geo.groupby("geolocation_zip_code_prefix")
        .agg(lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median"),
             city=("geolocation_city", "first"), state=("geolocation_state", "first"),
             raw_row_count=("geolocation_lat", "size"))
        .reset_index()
        .rename(columns={"geolocation_zip_code_prefix": "zip_code_prefix"})
    )
    load("geolocation_zip", geo_zip)

    # Validation: row counts and foreign-key integrity
    with engine.connect() as c:
        print("\nforeign_key_check violations:", len(c.execute(text("PRAGMA foreign_key_check")).fetchall()))
        for t in ["customers", "sellers", "products", "orders", "order_items", "order_payments", "order_reviews", "geolocation_zip"]:
            print(f"  {t:<18}", c.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar())


if __name__ == "__main__":
    main()
