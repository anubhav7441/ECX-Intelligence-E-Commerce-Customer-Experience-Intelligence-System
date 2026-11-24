"""Feature definitions, leakage-safe historical review features, temporal split and preprocessing.

PREDICTION TARGET: dissatisfied = 1 if review_score <= 2 (about 13% of orders -> imbalanced).

FEATURE SETS (what is known WHEN):
  S0  checkout-time   : order, product, price, freight, distance, estimated delivery, payment info
  S1  delivery-time   : S0 + fulfilment outcome (approval time, handling time, transit time, delivery days, delay, late flag)
  R   review-derived  : HISTORICAL seller / category / state review statistics computed ONLY from reviews created
                        strictly BEFORE the order's purchase time. The order's own review score/text is never used.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "dissatisfied"

CAT = ["customer_state", "seller_state", "primary_category", "main_payment_type"]
NUM_S0 = [
    "log_items_price", "log_freight_value", "freight_ratio", "n_items", "n_sellers", "n_categories",
    "log_total_weight_g", "log_total_volume_cm3", "product_photos_qty", "product_description_lenght",
    "product_name_lenght", "estimated_days", "distance_km", "same_state", "log_payment_value", "max_installments",
    "purchase_dow", "purchase_hour",
]
NUM_FULFIL = ["approval_hours", "seller_handling_days", "carrier_transit_days", "delivery_days", "delay_days", "is_late"]
NUM_HIST = ["seller_hist_n", "seller_hist_mean", "seller_hist_neg", "seller_hist_late_cmp",
            "cat_hist_mean", "cat_hist_neg", "state_hist_neg"]

FEATURE_SETS = {
    "S0 checkout-time": (NUM_S0, CAT),
    "S0+R checkout-time + history": (NUM_S0 + NUM_HIST, CAT),
    "S1 delivery-time": (NUM_S0 + NUM_FULFIL, CAT),
    "S1+R delivery-time + history": (NUM_S0 + NUM_FULFIL + NUM_HIST, CAT),
}

# feature -> group names used for grouped SHAP importance
FEATURE_GROUPS = {
    "Delivery performance": ["delivery_days", "delay_days", "is_late", "carrier_transit_days", "seller_handling_days",
                             "approval_hours", "estimated_days"],
    "Price & freight": ["log_items_price", "log_freight_value", "freight_ratio", "log_payment_value", "max_installments"],
    "Product": ["log_total_weight_g", "log_total_volume_cm3", "product_photos_qty", "product_description_lenght",
                "product_name_lenght", "primary_category"],
    "Order structure": ["n_items", "n_sellers", "n_categories"],
    "Geography": ["distance_km", "same_state", "customer_state", "seller_state"],
    "Payment type": ["main_payment_type"],
    "Time of purchase": ["purchase_dow", "purchase_hour"],
    "Historical reviews": NUM_HIST,
}


def add_basic_features(df):
    d = df.copy()
    for src, dst in [("items_price", "log_items_price"), ("freight_value", "log_freight_value"),
                     ("total_weight_g", "log_total_weight_g"), ("total_volume_cm3", "log_total_volume_cm3"),
                     ("payment_value", "log_payment_value")]:
        d[dst] = np.log1p(d[src].clip(lower=0))
    d["freight_ratio"] = d["freight_ratio"].clip(upper=d["freight_ratio"].quantile(0.99))
    return d


def temporal_split(df, train_frac=0.70, val_frac=0.15):
    """Time-ordered split on purchase timestamp. Customers who appear in the test period are removed from
    train/val so no customer is shared across splits."""
    d = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    n = len(d)
    i1, i2 = int(n * train_frac), int(n * (train_frac + val_frac))
    tr, va, te = d.iloc[:i1].copy(), d.iloc[i1:i2].copy(), d.iloc[i2:].copy()
    test_customers = set(te.customer_unique_id)
    n_tr0, n_va0 = len(tr), len(va)
    tr = tr[~tr.customer_unique_id.isin(test_customers)]
    va = va[~va.customer_unique_id.isin(test_customers)]
    info = {"n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "dropped_train_shared_customers": n_tr0 - len(tr), "dropped_val_shared_customers": n_va0 - len(va),
            "train_period": [str(tr.order_purchase_timestamp.min().date()), str(tr.order_purchase_timestamp.max().date())],
            "val_period": [str(va.order_purchase_timestamp.min().date()), str(va.order_purchase_timestamp.max().date())],
            "test_period": [str(te.order_purchase_timestamp.min().date()), str(te.order_purchase_timestamp.max().date())],
            "dissatisfied_rate_train": float(tr[TARGET].mean()), "dissatisfied_rate_val": float(va[TARGET].mean()),
            "dissatisfied_rate_test": float(te[TARGET].mean())}
    return tr, va, te, info


def add_history(df, late_flag_by_order, prior_mean, prior_neg, k=10):
    """Leakage-safe historical features. For every order, use only reviews of the same seller/category/state whose
    review_creation_date is STRICTLY BEFORE this order's purchase timestamp (so its own review can never be included).
    Means are Bayesian-shrunk toward the (training-set) prior with strength k."""
    d = df.reset_index(drop=True).copy()
    d["_late_cmp"] = d.order_id.map(late_flag_by_order).fillna(0).astype(float)
    rd = d.review_creation_date.values.astype("datetime64[ns]").astype("int64")
    pt = d.order_purchase_timestamp.values.astype("datetime64[ns]").astype("int64")
    score = d.review_score.values.astype(float)
    neg = d[TARGET].values.astype(float)
    late = d["_late_cmp"].values

    def run(key, outs):
        res = {name: np.full(len(d), np.nan) for name in outs}
        for _, idx in d.groupby(key).indices.items():
            order = idx[np.argsort(rd[idx], kind="stable")]
            dates = rd[order]
            cs = lambda v: np.concatenate([[0.0], np.cumsum(v[order])])
            c_s, c_n, c_l = cs(score), cs(neg), cs(late)
            pos = np.searchsorted(dates, pt[idx], side="left")   # reviews strictly before purchase
            n = pos.astype(float)
            if "n" in res:
                res["n"][idx] = n
            if "mean" in res:
                res["mean"][idx] = (c_s[pos] + k * prior_mean) / (n + k)
            if "neg" in res:
                res["neg"][idx] = (c_n[pos] + k * prior_neg) / (n + k)
            if "late_cmp" in res:
                res["late_cmp"][idx] = (c_l[pos] + k * 0.1) / (n + k)
        return res

    s = run("primary_seller_id", ["n", "mean", "neg", "late_cmp"])
    c = run("primary_category", ["mean", "neg"])
    st = run("customer_state", ["neg"])
    d["seller_hist_n"], d["seller_hist_mean"], d["seller_hist_neg"], d["seller_hist_late_cmp"] = s["n"], s["mean"], s["neg"], s["late_cmp"]
    d["cat_hist_mean"], d["cat_hist_neg"] = c["mean"], c["neg"]
    d["state_hist_neg"] = st["neg"]
    return d.drop(columns=["_late_cmp"])


def build_preprocessor(num_cols, cat_cols, scale=False):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=100, sparse_output=False))]), cat_cols),
    ], verbose_feature_names_out=False)
