"""Data-drift check: has the kind of orders the models see changed?

Population Stability Index (PSI) compares the distribution of a feature in the current window with the baseline window.
Common reading: below 0.10 stable, 0.10 to 0.25 some shift, above 0.25 a large shift. A large shift means the saved models
may be less reliable and should be checked or retrained. This checks the INPUTS only. Checking prediction quality needs the
real outcomes (reviews), which arrive late, so it is a separate, slower check."""
import numpy as np
import pandas as pd

from ecx3.monitor import windows

FEATURES = {"distance_km": "Delivery distance", "items_price": "Order value", "freight_ratio": "Freight share of price",
            "total_weight_g": "Parcel weight", "estimated_days": "Promised delivery time (days)", "seller_handling_days": "Seller handling time"}
MIN_N = 200


def psi(base, cur, bins=10):
    """PSI of `cur` against `base` using quantile bins of the baseline. Returns None if there is too little data."""
    b, c = pd.Series(base).dropna(), pd.Series(cur).dropna()
    if len(b) < MIN_N or len(c) < MIN_N or b.nunique() < 3:
        return None
    edges = np.unique(np.quantile(b, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    pb = np.histogram(b, edges)[0] / len(b)
    pc = np.histogram(c, edges)[0] / len(c)
    pb, pc = np.clip(pb, 1e-4, None), np.clip(pc, 1e-4, None)
    return float(np.sum((pc - pb) * np.log(pc / pb)))


def check(d, window_days=28, baseline_days=84, as_of=None):
    cur, base, (cs, end, bs) = windows(d, window_days, baseline_days, as_of)
    rows = []
    for col, label in FEATURES.items():
        if col in d.columns:
            v = psi(base[col], cur[col])
            if v is not None:
                rows.append({"feature": col, "label": label, "psi": v, "level": "large" if v > 0.25 else "some" if v > 0.10 else "stable"})
    rows.sort(key=lambda r: -r["psi"])
    return {"window": f"{cs} to {end} vs {bs} to {cs}", "features": rows, "max_psi": rows[0]["psi"] if rows else None}


def drift_alerts(d, window_days=28, baseline_days=84, as_of=None):
    """Alert dicts for features with a large shift (warning) or some shift (info). Uses the same shape as monitor alerts."""
    res = check(d, window_days, baseline_days, as_of)
    cur, base, (cs, end, bs) = windows(d, window_days, baseline_days, as_of)
    out = []
    for r in res["features"]:
        if r["level"] == "stable":
            continue
        out.append({"dedupe_key": f"data_drift:{r['feature']}", "kind": "data_drift", "metric": f"drift_{r['feature']}", "entity_type": "platform", "entity_id": r["feature"],
                    "severity": "warning" if r["level"] == "large" else "info",
                    "title": f"Data drift: {r['label'].lower()} has shifted ({r['level']}, PSI {r['psi']:.2f})",
                    "detail": ("The mix of orders differs from the previous period (PSI is a distribution-shift score; above 0.25 is a large shift). "
                               "The saved models may be less reliable. Check whether this is seasonal, then consider retraining. Note: the newest window only holds orders already delivered and reviewed, which alone can create a shift."),
                    "as_of": (pd.Timestamp(end) - pd.Timedelta(days=1)).date().isoformat(), "window_label": res["window"],
                    "baseline_value": None, "current_value": r["psi"], "n_current": int(len(cur)), "n_baseline": int(len(base)), "p_value": None, "effect": r["psi"]})
    return out
