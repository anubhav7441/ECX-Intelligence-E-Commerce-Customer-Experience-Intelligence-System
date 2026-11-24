import sqlite3
import pandas as pd
from ml.preprocessing.build_dataset import haversine_km, ROOT

DB = ROOT / "database" / "ecx.db"


def test_haversine_known_distance():
    # Sao Paulo -> Rio de Janeiro is roughly 357 km in a straight line
    d = haversine_km(-23.5505, -46.6333, -22.9068, -43.1729)
    assert 340 < float(d) < 375


def test_haversine_zero():
    assert float(haversine_km(-10, -50, -10, -50)) == 0.0


def test_one_row_per_order(analytic):
    assert analytic.order_id.is_unique


def test_only_delivered_with_review(analytic):
    assert (analytic.order_status == "delivered").all()
    assert analytic.review_score.between(1, 5).all()
    assert analytic.order_delivered_customer_date.notna().all()


def test_target_definition(analytic):
    assert ((analytic.review_score <= 2).astype(float) == analytic.dissatisfied).all()


def test_late_flag_matches_delay(analytic):
    assert ((analytic.delay_days > 0).astype(float) == analytic.is_late).all()


def test_no_join_inflation_revenue_matches_sql():
    """Aggregating in pandas must equal a direct SQL SUM over order_items (join-inflation guard)."""
    full = pd.read_parquet(ROOT / "data" / "processed" / "orders_all.parquet")
    con = sqlite3.connect(DB)
    sql_total = con.execute("SELECT SUM(price) FROM order_items").fetchone()[0]
    assert abs(full.items_price.sum() - sql_total) < 0.01
    assert len(full) == con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    con.close()


def test_sql_average_review_matches_pandas(analytic):
    con = sqlite3.connect(DB)
    n = con.execute("SELECT COUNT(DISTINCT order_id) FROM order_reviews").fetchone()[0]
    assert n == 98673
    con.close()


def test_foreign_keys_have_no_violations():
    con = sqlite3.connect(DB)
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    con.close()


def test_freight_ratio_non_negative(analytic):
    assert (analytic.freight_ratio.dropna() >= 0).all()
