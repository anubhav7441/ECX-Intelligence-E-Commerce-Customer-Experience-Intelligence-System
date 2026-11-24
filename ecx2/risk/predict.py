"""Runtime early late-delivery risk scoring (Workflow A). Pre-delivery information only."""
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from ecx2.common import V2_MODELS, load_json

RANGES = {"items_price": (0.5, 20000), "freight_value": (0, 2000), "estimated_days": (1, 120), "distance_km": (0, 8000),
          "n_items": (1, 30), "total_weight_g": (0, 200000), "approval_hours": (0, 2000), "seller_handling_days": (0, 120)}
STAGE_FILE = {"A0": "delivery_risk_A0.joblib", "A1": "delivery_risk_A1.joblib"}


@lru_cache(maxsize=2)
def _art(stage):
    p = V2_MODELS / STAGE_FILE[stage]
    if not p.exists():
        raise FileNotFoundError("Risk model missing - run: python -m ecx2.risk.train")
    return joblib.load(p)


def validate(inp):
    errs = []
    for k, (lo, hi) in RANGES.items():
        if inp.get(k) not in (None, ""):
            try:
                v = float(inp[k])
                if not lo <= v <= hi:
                    errs.append(f"{k} must be between {lo} and {hi}")
            except (TypeError, ValueError):
                errs.append(f"{k} must be a number")
    return errs


def _row(inp, ref):
    med = ref["num_median"]
    g = lambda k, d=None: float(inp[k]) if inp.get(k) not in (None, "") else (med.get(k) if d is None else d)
    seller = inp.get("seller_id") or None
    price, freight = g("items_price", 80.0), g("freight_value", 20.0)
    sstats = ref["seller"].get(seller) if seller else None
    prior, k = ref["prior_late"], ref["k"]
    dist = g("distance_km", ref["seller_dist_median"].get(seller, med["distance_km"]) if seller else med["distance_km"])
    weight = g("total_weight_g", 1500.0)
    row = {"log_items_price": np.log1p(price), "log_freight_value": np.log1p(freight),
           "freight_ratio": min(freight / max(price, 0.5), 3.0), "n_items": g("n_items", 1), "n_sellers": 1, "n_categories": 1,
           "log_total_weight_g": np.log1p(weight), "log_total_volume_cm3": med.get("log_total_volume_cm3", np.log1p(8000)),
           "product_photos_qty": med.get("product_photos_qty", 1), "product_description_lenght": med.get("product_description_lenght", 500),
           "product_name_lenght": med.get("product_name_lenght", 50), "estimated_days": g("estimated_days", med["estimated_days"]),
           "distance_km": dist, "same_state": float(bool(inp.get("seller_state") and inp.get("seller_state") == inp.get("customer_state"))),
           "log_payment_value": np.log1p(price + freight), "max_installments": g("max_installments", 1),
           "purchase_dow": g("purchase_dow", 2), "purchase_hour": g("purchase_hour", 14),
           "seller_late_hist": ((sstats["rate"] * sstats["n"] + k * prior) / (sstats["n"] + k)) if sstats else prior,
           "seller_late_hist_n": sstats["n"] if sstats else 0,
           "cat_late_hist": ref["category"].get(inp.get("category"), prior),
           "state_late_hist": ref["state"].get(inp.get("customer_state"), prior),
           "platform_late_30d": g("platform_late_30d", ref["platform_late_30d_latest"]),
           "platform_orders_7d": g("platform_orders_7d", ref["platform_orders_7d_latest"]),
           "approval_hours": g("approval_hours", med["approval_hours"]), "seller_handling_days": g("seller_handling_days", med["seller_handling_days"]),
           "customer_state": inp.get("customer_state") or "missing", "seller_state": inp.get("seller_state") or ref["seller_state"].get(seller, "missing") if seller else (inp.get("seller_state") or "missing"),
           "primary_category": inp.get("category") or "missing", "main_payment_type": inp.get("main_payment_type") or "credit_card"}
    return row, bool(sstats)


def predict_risk(inp):
    """inp: dict. If approval_hours AND seller_handling_days are both given -> stage A1, else stage A0."""
    errs = validate(inp)
    if errs:
        return {"ok": False, "errors": errs}
    ref = load_json("risk_reference.json")
    res = load_json("risk_results.json")
    if ref is None or res is None:
        return {"ok": False, "errors": ["Risk results missing - run: python -m ecx2.risk.train"]}
    stage = "A1" if all(inp.get(k) not in (None, "") for k in ("approval_hours", "seller_handling_days")) else "A0"
    art = _art(stage)
    row, seller_known = _row(inp, ref)
    cols = art["num_cols"] + art["cat_cols"]
    X = pd.DataFrame([row])[cols]
    Xt = art["pre"].transform(X)
    score = float(art["model"].predict_proba(Xt)[:, 1][0])
    c0, c1 = art["bands"]
    band = "Low" if score < c0 else ("Medium" if score < c1 else "High")
    sname = "A0 checkout-time" if stage == "A0" else "A1 carrier hand-off"
    obs = res["selected"][sname]["bands"]["observed_test_late_rate"][band]
    # explanation for the linear model: coefficient x transformed value, grouped to original features
    factors = []
    if hasattr(art["model"], "coef_"):
        names = list(art["pre"].get_feature_names_out())
        contrib = art["model"].coef_[0] * np.asarray(Xt)[0]
        grp = {}
        for n, c in zip(names, contrib):
            base = next((cc for cc in art["cat_cols"] if n.startswith(cc + "_")), n)
            grp[base] = grp.get(base, 0.0) + float(c)
        factors = [{"feature": k, "contribution": round(v, 3), "direction": "raises risk" if v > 0 else "lowers risk"}
                   for k, v in sorted(grp.items(), key=lambda kv: -abs(kv[1]))[:6]]
    return {"ok": True, "stage": sname, "model": art["model_name"], "risk_score": round(score, 4), "band": band,
            "observed_late_rate_in_band": round(obs["late_rate"], 4), "band_n_test_orders": obs["n"],
            "seller_history_used": seller_known, "top_factors": factors,
            "test_roc_auc": round(res["selected"][sname]["test"]["roc_auc"], 3),
            "note": "Score is a ranking signal, NOT a probability. 'observed_late_rate_in_band' is the share of test-period orders in this band that were actually late. Historical data only; one marketplace."}
