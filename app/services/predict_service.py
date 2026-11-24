"""Prediction + SHAP explanation service.

Two models are served (both XGBoost, selected on validation PR-AUC in the training run):
  * checkout-time model  (S0+R): uses only information known when the order is placed + seller/category/state HISTORY
  * delivery-time model  (S1+R): additionally uses the delivery outcome (days, delay, late flag)
Historical features for a seller/category/state are computed from ALL reviews in the dataset, with the same shrinkage as in
training. Anything the user leaves blank is imputed with the training median by the model pipeline.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

from ml.explainability.explain import load_artifact, explain_rows
from ml.prediction.features import TARGET
import json

ROOT = Path(__file__).resolve().parents[2]
K = 10


class InputError(ValueError):
    pass


@lru_cache(maxsize=1)
def calibration():
    with open(ROOT / "ml" / "evaluation" / "calibration.json", encoding="utf-8") as f:
        return json.load(f)["models"]


def attach_band(res, tag):
    """Translate the raw (uncalibrated) score into what was actually observed on the held-out test period."""
    cal = calibration()[tag]
    p = res["probability_dissatisfied"]
    res["risk_score"] = p
    band = cal["bands"][-1]
    for b in cal["bands"]:
        if p < b["hi"]:
            band = b
            break
    if p < cal["bands"][0]["lo"]:
        band = cal["bands"][0]
    res["observed_rate_in_band"] = band["observed_dissatisfied_rate"]
    res["band_range"] = [band["lo"], band["hi"]]
    res["band_n_test_orders"] = band["n"]
    res["test_base_rate"] = cal["test_base_rate"]
    return res


@lru_cache(maxsize=1)
def reference():
    full = pd.read_parquet(ROOT / "data" / "processed" / "orders_with_history.parquet")
    nlp = pd.read_parquet(ROOT / "data" / "processed" / "review_nlp.parquet")
    late = nlp.set_index("order_id").cat_late_delivery
    full["late_cmp"] = full.order_id.map(late).fillna(0)
    mu, neg0 = float(full.review_score.mean()), float(full[TARGET].mean())

    def grp(key, with_late=False):
        g = full.groupby(key).agg(n=("review_score", "size"), s=("review_score", "sum"), k=(TARGET, "sum"), l=("late_cmp", "sum"))
        out = pd.DataFrame({"n": g.n, "mean": (g.s + K * mu) / (g.n + K), "neg": (g.k + K * neg0) / (g.n + K),
                            "late_cmp": (g.l + K * 0.1) / (g.n + K)})
        return out

    seller = grp("primary_seller_id"); cat = grp("primary_category"); state = grp("customer_state")
    seller_state = full.groupby("primary_seller_id").seller_state.agg(lambda s: s.mode().iloc[0])
    dist = full.groupby(["seller_state", "customer_state"]).distance_km.median()
    sellers = full.groupby("primary_seller_id").size()
    return {"seller": seller, "cat": cat, "state": state, "seller_state": seller_state, "dist": dist,
            "eligible_sellers": sellers[sellers >= 50].sort_values(ascending=False).index.tolist(),
            "categories": sorted(full.primary_category.dropna().unique().tolist()),
            "states": sorted(full.customer_state.dropna().unique().tolist()),
            "payment_types": ["credit_card", "boleto", "voucher", "debit_card"],
            "medians": full[["total_weight_g", "total_volume_cm3"]].median().to_dict(),
            "full": full}


def options():
    r = reference()
    return {"sellers": r["eligible_sellers"][:300], "categories": r["categories"], "states": r["states"], "payment_types": r["payment_types"]}


def _num(d, key, lo, hi, required=False, default=None):
    v = d.get(key, None)
    if v is None or v == "":
        if required:
            raise InputError(f"'{key}' is required.")
        return default
    try:
        v = float(v)
    except (TypeError, ValueError) as err:
        raise InputError(f"'{key}' must be a number.") from err
    if not np.isfinite(v) or v < lo or v > hi:
        raise InputError(f"'{key}' must be between {lo} and {hi}.")
    return v


def build_row(d):
    """Validate form/JSON input and build ONE order row with all columns the models expect."""
    r = reference()
    price = _num(d, "items_price", 0.01, 50000, required=True)
    freight = _num(d, "freight_value", 0, 5000, required=True)
    n_items = int(_num(d, "n_items", 1, 50, default=1))
    est_days = _num(d, "estimated_days", 1, 200, required=True)
    seller_id = (d.get("seller_id") or "").strip()
    cust_state = (d.get("customer_state") or "").strip().upper()
    category = (d.get("category") or "").strip()
    pay = (d.get("payment_type") or "credit_card").strip()
    if cust_state not in r["states"]:
        raise InputError("Choose a valid customer state.")
    if category not in r["categories"]:
        raise InputError("Choose a valid product category.")
    if pay not in r["payment_types"]:
        raise InputError("Choose a valid payment type.")
    if seller_id not in r["seller"].index:
        raise InputError("Choose a seller from the list.")
    sell_state = r["seller_state"].get(seller_id)
    dist = _num(d, "distance_km", 0, 9000)
    if dist is None:
        dist = r["dist"].get((sell_state, cust_state), np.nan)
    weight = _num(d, "total_weight_g", 0, 200000, default=r["medians"]["total_weight_g"])
    volume = _num(d, "total_volume_cm3", 0, 5e6, default=r["medians"]["total_volume_cm3"])
    installments = int(_num(d, "max_installments", 1, 24, default=1))
    n_sellers = int(_num(d, "n_sellers", 1, 20, default=1))
    delivery_days = _num(d, "delivery_days", 0, 400)  # optional -> delivery-time model
    sh, ch, sth = r["seller"].loc[seller_id], r["cat"].loc[category], r["state"].loc[cust_state]
    row = {
        "items_price": price, "freight_value": freight, "freight_ratio": min(freight / price, float(r["full"].freight_ratio.quantile(.99))),
        "n_items": n_items, "n_sellers": n_sellers, "n_categories": 1, "total_weight_g": weight, "total_volume_cm3": volume,
        "product_photos_qty": np.nan, "product_description_lenght": np.nan, "product_name_lenght": np.nan,
        "estimated_days": est_days, "distance_km": dist, "same_state": int(sell_state == cust_state),
        "payment_value": price + freight, "max_installments": installments, "purchase_dow": np.nan, "purchase_hour": np.nan,
        "customer_state": cust_state, "seller_state": sell_state, "primary_category": category, "main_payment_type": pay,
        "seller_hist_n": sh.n, "seller_hist_mean": sh["mean"], "seller_hist_neg": sh.neg, "seller_hist_late_cmp": sh.late_cmp,
        "cat_hist_mean": ch["mean"], "cat_hist_neg": ch.neg, "state_hist_neg": sth.neg,
        "approval_hours": np.nan, "seller_handling_days": np.nan, "carrier_transit_days": np.nan,
        "delivery_days": np.nan, "delay_days": np.nan, "is_late": np.nan,
    }
    mode = "checkout"
    if delivery_days is not None:
        mode = "delivery"
        row["delivery_days"] = delivery_days
        row["delay_days"] = float(np.ceil(delivery_days - est_days))   # approximation: day-level difference
        row["is_late"] = float(row["delay_days"] > 0)
    df = pd.DataFrame([row])
    # basic features (log transforms) - must match training
    for src, dst in [("items_price", "log_items_price"), ("freight_value", "log_freight_value"), ("total_weight_g", "log_total_weight_g"),
                     ("total_volume_cm3", "log_total_volume_cm3"), ("payment_value", "log_payment_value")]:
        df[dst] = np.log1p(df[src].clip(lower=0))
    return df, mode


def predict(d, top=8):
    df, mode = build_row(d)
    tag = "S1_R" if mode == "delivery" else "S0_R"
    art = load_artifact(tag)
    res = explain_rows(art, df, top=top)[0]
    attach_band(res, tag)
    res["mode"] = mode
    res["model"] = f"{art['model_name']} ({art['feature_set']})"
    res["note"] = ("Estimated from order data and the seller's/category's historical review statistics. "
                   "It is a risk score (association learned from 2016-2018 data), not a certain outcome.")
    # make values JSON-safe
    for c in res["contributions"]:
        v = c["value"]
        c["value"] = None if v is None else (round(v, 3) if isinstance(v, float) else v)
    return res


def explain_order(order_id, top=8):
    r = reference()
    full = r["full"]
    row = full[full.order_id == order_id]
    if row.empty:
        raise InputError("Order id not found in the analytic dataset.")
    delivered = True
    art = load_artifact("S1_R")
    res = explain_rows(art, row, top=top)[0]
    res["model"] = f"{art['model_name']} ({art['feature_set']})"
    attach_band(res, "S1_R")
    res["order_id"] = order_id
    res["actual_review_score"] = int(row.review_score.iloc[0])
    res["actual_dissatisfied"] = int(row.dissatisfied.iloc[0])
    for c in res["contributions"]:
        v = c["value"]
        c["value"] = None if v is None else (round(v, 3) if isinstance(v, float) else v)
    return res
