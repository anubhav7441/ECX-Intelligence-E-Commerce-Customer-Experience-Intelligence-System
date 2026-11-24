import numpy as np
import pandas as pd
from ml.prediction.features import (FEATURE_SETS, NUM_HIST, TARGET, add_basic_features, add_history, temporal_split)

FORBIDDEN = {"review_score", "dissatisfied", "review_comment_message", "review_comment_title", "review_creation_date",
             "review_answer_timestamp", "has_comment", "review_id", "sent_p_negative"}


def test_no_target_or_review_columns_in_any_feature_set():
    for name, (num, cat) in FEATURE_SETS.items():
        assert not (set(num) | set(cat)) & FORBIDDEN, name


def test_checkout_set_has_no_delivery_outcome():
    num, _ = FEATURE_SETS["S0 checkout-time"]
    assert not {"delivery_days", "delay_days", "is_late", "carrier_transit_days"} & set(num)


def test_temporal_split_is_ordered_and_customer_disjoint(analytic):
    df = add_basic_features(analytic)
    tr, va, te, info = temporal_split(df)
    assert tr.order_purchase_timestamp.max() <= va.order_purchase_timestamp.min() <= te.order_purchase_timestamp.min()
    assert not set(tr.customer_unique_id) & set(te.customer_unique_id)
    assert not set(va.customer_unique_id) & set(te.customer_unique_id)


def test_history_features_use_only_earlier_reviews(analytic):
    """Brute-force check: for sampled orders, history equals the shrunk mean of reviews created strictly before purchase."""
    df = add_basic_features(analytic).sort_values("order_purchase_timestamp").head(15000).reset_index(drop=True)
    pm, pn = float(df.review_score.mean()), float(df[TARGET].mean())
    h = add_history(df, pd.Series(dtype=float), pm, pn)
    rng = np.random.default_rng(1)
    for i in rng.choice(len(df), 25, replace=False):
        r = df.iloc[i]
        past = df[(df.primary_seller_id == r.primary_seller_id) & (df.review_creation_date < r.order_purchase_timestamp)]
        assert h.loc[i, "seller_hist_n"] == len(past)
        assert abs(h.loc[i, "seller_hist_mean"] - (past.review_score.sum() + 10 * pm) / (len(past) + 10)) < 1e-9
        assert r.order_id not in set(past.order_id)


def test_history_columns_listed():
    assert "seller_hist_mean" in NUM_HIST
