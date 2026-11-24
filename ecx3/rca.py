"""Root-cause breakdown for an alert.

For an alert (metric, entity, windows) it asks: WHERE did the extra problem events come from?
  * "excess events" of a level = events in the current window minus the events expected if that level had kept the whole
    entity's baseline rate. The share of the total excess that sits in one level shows how concentrated the problem is.
  * a level is a driver candidate when it holds a meaningful share of the excess AND its rate is clearly above the rest of the current window (Fisher exact).
  * for late deliveries it also checks which delivery stage got slower (seller handling vs carrier transit). Transit time is not offered as a
    driver band, because slow transit is largely the same thing as being late (an outcome, not a cause).
Findings are ASSOCIATIONS from observed data. They are leads for people to verify, not proven causes."""
import pandas as pd
from scipy import stats

from ecx3.monitor import METRICS, windows

DIMENSIONS = [  # (key, label, column or bucket-builder)
    ("state", "Customer state", "customer_state"),
    ("category", "Product category", "primary_category"),
    ("seller", "Seller", "primary_seller_id"),
    ("seller_state", "Seller state", "seller_state"),
    ("distance", "Delivery distance", "_distance_band"),
    ("price", "Order value", "_price_band"),
    ("weight", "Parcel weight", "_weight_band"),
    ("freight", "Freight share of price", "_freight_band"),
    ("handling", "Seller handling time", "_handling_band"),
    ("weekday", "Purchase weekday", "_weekday"),
]
MIN_LEVEL_N = 12
MIN_SHARE = 0.08


def _bands(d):
    d = d.copy()
    def cut(col, bins, labels):
        return pd.cut(d[col], bins=bins, labels=labels, include_lowest=True).astype("object") if col in d else None
    d["_distance_band"] = cut("distance_km", [-1, 100, 500, 1000, 2000, 1e9], ["<100 km", "100-500 km", "500-1000 km", "1000-2000 km", ">2000 km"])
    d["_price_band"] = cut("items_price", [-1, 50, 100, 200, 500, 1e9], ["<50", "50-100", "100-200", "200-500", ">500"])
    d["_weight_band"] = cut("total_weight_g", [-1, 500, 2000, 5000, 1e9], ["<0.5 kg", "0.5-2 kg", "2-5 kg", ">5 kg"])
    d["_freight_band"] = cut("freight_ratio", [-1, 0.15, 0.3, 0.6, 1e9], ["<15%", "15-30%", "30-60%", ">60%"])
    d["_handling_band"] = cut("seller_handling_days", [-1, 1, 3, 7, 1e9], ["<=1 day", "2-3 days", "4-7 days", ">7 days"])
    d["_weekday"] = pd.to_datetime(d["order_purchase_timestamp"]).dt.day_name()
    return d


def _subset(d, etype, eid):
    col = {"state": "customer_state", "category": "primary_category", "seller": "primary_seller_id"}.get(etype)
    return d if col is None else d[d[col].astype(str) == str(eid)]


def analyse(d, alert, window_days=28, baseline_days=84):
    """Return the root-cause breakdown for a stored alert dict."""
    metric = alert["metric"]
    if metric not in METRICS:
        raise ValueError(f"unknown metric '{metric}'")
    col = METRICS[metric]["col"]
    sub = _subset(d, alert["entity_type"], alert["entity_id"])
    cur, base, _ = windows(sub, window_days, baseline_days, as_of=alert.get("as_of"))
    n1, n0 = len(cur), len(base)
    if n1 == 0 or n0 == 0:
        raise ValueError("not enough orders in the windows to analyse this alert")
    r1, r0 = cur[col].mean(), base[col].mean()
    total_excess = float(cur[col].sum() - n1 * r0)
    cur, base = _bands(cur), _bands(base)
    # a dimension that equals the alert's own entity cannot explain it
    skip = {"platform": set(), "state": {"state"}, "category": {"category"}, "seller": {"seller", "seller_state"}}[alert["entity_type"]]
    cands = []
    for key, label, c in DIMENSIONS:
        if key in skip or c not in cur.columns or cur[c].isna().all():
            continue
        for lvl, g in cur.groupby(c, observed=True):
            n, k = len(g), int(g[col].sum())
            if n < MIN_LEVEL_N:
                continue
            excess = float(k - n * r0)
            if total_excess <= 0 or excess <= 0:
                continue
            share = min(excess / total_excess, 1.0)          # other groups can offset, so a single group may exceed 100%; show at most 100%
            rest = cur[cur[c] != lvl]
            if len(rest) < MIN_LEVEL_N:
                continue
            p = stats.fisher_exact([[k, n - k], [int(rest[col].sum()), len(rest) - int(rest[col].sum())]], alternative="greater")[1]
            bl = base[base[c] == lvl]
            cands.append({"dimension": key, "dimension_label": label, "level": str(lvl), "n_current": n, "events_current": k,
                          "rate_current": k / n, "rate_rest_current": float(rest[col].mean()),
                          "rate_baseline": float(bl[col].mean()) if len(bl) >= 5 else None,
                          "excess_events": excess, "share_of_excess": share, "p_vs_rest": float(p)})
    drivers = [c for c in cands if c["share_of_excess"] >= MIN_SHARE and c["p_vs_rest"] < 0.05 and c["rate_current"] > c["rate_rest_current"]]
    drivers.sort(key=lambda c: -c["excess_events"])
    drivers = drivers[:8]
    stage = None
    if metric == "late_rate" and {"seller_handling_days", "carrier_transit_days"} <= set(cur.columns):
        stage = {}
        for name, lab in [("seller_handling_days", "Seller handling time (days)"), ("carrier_transit_days", "Carrier transit time (days)")]:
            a, b = cur[name].dropna(), base[name].dropna()
            if len(a) > 10 and len(b) > 10:
                stage[name] = {"label": lab, "median_current": float(a.median()), "median_baseline": float(b.median()),
                               "p_slower": float(stats.mannwhitneyu(a, b, alternative="greater").pvalue)}
        if stage:
            slow = [v["label"] for v in stage.values() if v["p_slower"] < 0.05 and v["median_current"] > v["median_baseline"]]
            stage["slower_stages"] = slow
    top = drivers[0] if drivers else None
    conc = sum(d["share_of_excess"] for d in drivers[:3]) if drivers else 0.0
    confidence = "High" if (n1 >= 200 and conc >= 0.6) else "Medium" if (n1 >= 60 and drivers) else "Low"
    return {"alert_id": alert.get("alert_id"), "metric": metric, "entity_type": alert["entity_type"], "entity_id": alert["entity_id"],
            "n_current": n1, "n_baseline": n0, "rate_current": float(r1), "rate_baseline": float(r0), "total_excess_events": total_excess,
            "drivers": drivers, "stage_check": stage, "top_driver": top, "top_share": float(top["share_of_excess"]) if top else 0.0, "top3_share": float(conc), "confidence": confidence,
            "limitations": ["Observed associations only, not proven causes. Verify with the owner before acting.",
                            "Small windows are noisy, and recent orders may still be missing late reviews.",
                            "Several drivers can overlap (for example one remote state served by one seller)."]}
